use crate::ClipperPlugin;
use nih_plug::nih_log;
use num_traits::{FloatConst, Zero};
use std::cell::{OnceCell, RefCell};
use std::f64::consts::TAU;
use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::{
    HasParameters, ParamId, ParamName, RemoteControl, RemoteControlled, SmoothedParam,
};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, DSPProcessBlock};
use valib::filters::biquad::Biquad;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::clippers::DiodeClipperModel;
use valib::saturators::Linear;
use valib::simd::{AutoF32x2, AutoF64x2, SimdComplexField};
use valib::wdf::adapters::Parallel;
use valib::wdf::diode::{DiodeLambert, DiodeModel};
use valib::wdf::dsl::{node, node_mut, voltage};
use valib::wdf::leaves::{Capacitor, ResistiveVoltageSource};
use valib::wdf::module::WdfModule;
use valib::wdf::{Wave, Wdf};
use valib::{wdf, Scalar, SimdCast};

struct DcBlocker<T>(Biquad<T, Linear>);

impl<T> DcBlocker<T> {
    const CUTOFF_HZ: f32 = 5.0;
    const Q: f32 = 0.707;
    fn new(samplerate: f32) -> Self
    where
        T: Scalar,
    {
        Self(Biquad::highpass(
            T::from_f64((Self::CUTOFF_HZ / samplerate) as f64),
            T::from_f64(Self::Q as f64),
        ))
    }
}

impl<T: Scalar> DSPMeta for DcBlocker<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.0.set_samplerate(samplerate);
        self.0.update_coefficients(&Biquad::highpass(
            T::from_f64((Self::CUTOFF_HZ / samplerate) as f64),
            T::from_f64(Self::Q as f64),
        ));
    }

    fn latency(&self) -> usize {
        self.0.latency()
    }

    fn reset(&mut self) {
        self.0.reset()
    }
}

impl<T: Scalar> DSPProcess<1, 1> for DcBlocker<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.0.process(x)
    }
}

type Sample = AutoF32x2;
type Sample64 = AutoF64x2;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ParamName)]
pub enum DspParams {
    Drive,
    Cutoff,
    ForceReset,
}

pub struct DspInner {
    drive: SmoothedParam,
    cutoff: SmoothedParam,
    model: WdfModule<
        DiodeLambert<Sample64>,
        Parallel<ResistiveVoltageSource<Sample64>, Capacitor<Sample64>>,
    >,
    rvs: wdf::Node<ResistiveVoltageSource<Sample64>>,
}

impl DspInner {
    const C: f64 = 33e-9;
    fn new(samplerate: f32) -> Self {
        let rvs = node(ResistiveVoltageSource::new(
            Self::resistance_for_cutoff(3000.),
            Sample64::zero(),
        ));
        let module = WdfModule::new(
            node(DiodeLambert::germanium(1)),
            node(Parallel::new(
                rvs.clone(),
                node(Capacitor::new(
                    Sample64::from_f64(samplerate as _),
                    Sample64::from_f64(Self::C),
                )),
            )),
        );
        Self {
            drive: SmoothedParam::exponential(1.0, samplerate, 10.0),
            cutoff: SmoothedParam::exponential(3000., samplerate, 10.),
            model: module,
            rvs,
        }
    }

    fn resistance_for_cutoff(cutoff: f32) -> Sample64 {
        Sample64::simd_recip(Sample64::from_f64(TAU * Self::C * cutoff as f64))
    }
}

impl HasParameters for DspInner {
    type Name = DspParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        nih_log!("DspInner::set_parameter {param:?} {value}");
        match param {
            DspParams::Drive => {
                self.drive.param = value;
            }
            DspParams::Cutoff => {
                self.cutoff.param = value;
            }
            DspParams::ForceReset => {
                self.reset();
            }
        }
    }
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.drive.set_samplerate(samplerate);
        self.model.set_samplerate(samplerate as _);
    }

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.drive.reset();
        self.model.reset();
    }
}

impl DSPProcess<1, 1> for DspInner {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let drive = self.drive.next_sample_as::<Sample64>();
        {
            let x: Sample64 = x[0].cast();
            let mut rvs = node_mut(&self.rvs);
            rvs.r = Self::resistance_for_cutoff(self.cutoff.next_sample());
            rvs.vs = drive * x;
        }
        self.model.process_sample();
        let out = voltage(&self.model.root) / drive.simd_tanh();
        [out.cast()]
    }
}

pub struct Dsp {
    inner: RemoteControlled<Oversampled<Sample, BlockAdapter<DspInner>>>,
    dc_blocker: DcBlocker<Sample>,
    pub rc: RemoteControl<DspParams>,
}

impl DSPMeta for Dsp {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner.set_samplerate(samplerate);
        self.dc_blocker.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.inner.latency() + self.dc_blocker.latency()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.dc_blocker.reset();
    }
}

impl DSPProcessBlock<1, 1> for Dsp {
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, 1>,
        mut outputs: AudioBufferMut<Self::Sample, 1>,
    ) {
        self.inner.process_block(inputs, outputs.as_mut());
        for i in 0..outputs.samples() {
            outputs.set_frame(i, self.dc_blocker.process(outputs.get_frame(i)));
        }
    }

    fn max_block_size(&self) -> Option<usize> {
        self.inner.max_block_size()
    }
}

impl HasParameters for Dsp {
    type Name = DspParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        self.inner.inner.inner.0.set_parameter(param, value); // lol
    }
}

pub fn create_dsp(
    samplerate: f32,
    oversample: usize,
    max_block_size: usize,
) -> RemoteControlled<Dsp> {
    nih_log!("dsp::create_dsp {:?}", std::thread::current().id());
    let inner = DspInner::new(samplerate * oversample as f32);
    let os = Oversample::new(oversample, max_block_size).with_dsp(samplerate, BlockAdapter(inner));
    let inner = RemoteControlled::new(samplerate, 1e3, os);
    let rc = inner.proxy.clone();
    let dsp = Dsp {
        inner,
        dc_blocker: DcBlocker::new(samplerate),
        rc,
    };
    RemoteControlled::new(samplerate, 1e3, dsp)
}
