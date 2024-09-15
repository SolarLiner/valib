use nih_plug::{nih_log, prelude::Enum};
use num_traits::Zero;
use std::f64::consts::TAU;
use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::ParamId;
use valib::dsp::parameter::{HasParameters, ParamName, RemoteControlled, SmoothedParam};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, DSPProcessBlock};
use valib::filters::specialized::DcBlocker;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::clippers::DiodeClipper;
use valib::simd::{AutoF32x2, AutoF64x2, SimdComplexField};
use valib::wdf;
use valib::wdf::adapters::Parallel;
use valib::wdf::dsl::*;
use valib::wdf::leaves::{Capacitor, ResistiveVoltageSource};
use valib::wdf::module::WdfModule;
use valib::wdf::DiodeNR;
use valib::{Scalar, SimdCast};

type Sample = AutoF32x2;
type Sample64 = AutoF64x2;

#[derive(Debug, Copy, Clone, Eq, PartialEq, ParamName)]
pub enum DspParams {
    Drive,
    Cutoff,
    NumForward,
    NumBackward,
    DiodeType,
    ForceReset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum, ParamName)]
pub enum DiodeType {
    Silicon,
    Germanium,
    Led,
}

pub struct DspInner {
    drive: SmoothedParam,
    cutoff: SmoothedParam,
    model: WdfModule<
        DiodeNR<Sample64>,
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
        let diode = {
            let data = DiodeClipper::new_germanium(1, 1, Sample64::zero());
            //diode_lambert(data.isat, data.vt)
            diode_nr(data)
        };
        let module = module(
            diode,
            parallel(
                rvs.clone(),
                capacitor(
                    Sample64::from_f64(samplerate as _),
                    Sample64::from_f64(Self::C),
                ),
            ),
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
            DspParams::NumForward => {
                node_mut(&self.model.root).set_num_forward(value as usize);
            }
            DspParams::NumBackward => {
                node_mut(&self.model.root).set_num_backward(value as usize);
            }
            DspParams::DiodeType => {
                let mut root_node = node_mut(&self.model.root);
                let data = match DiodeType::from_id(value as _) {
                    DiodeType::Silicon => DiodeClipper::new_silicon(1, 1, Sample64::zero()),
                    DiodeType::Germanium => DiodeClipper::new_germanium(1, 1, Sample64::zero()),
                    DiodeType::Led => DiodeClipper::new_led(1, 1, Sample64::zero()),
                };
                root_node.set_configuration(data);
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
    inner: Oversampled<Sample, BlockAdapter<DspInner>>,
    dc_blocker: DcBlocker<Sample>,
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
        self.inner.inner.0.set_parameter(param, value); // lol
    }
}

pub fn create_dsp(
    samplerate: f32,
    oversample: usize,
    max_block_size: usize,
) -> RemoteControlled<Dsp> {
    nih_log!("dsp::create_dsp {:?}", std::thread::current().id());
    let inner = DspInner::new(samplerate * oversample as f32);
    let os = Oversample::new(oversample, max_block_size);
    let dsp = Dsp {
        inner: os.with_dsp(samplerate, BlockAdapter(inner)),
        dc_blocker: DcBlocker::new(samplerate),
    };
    RemoteControlled::new(samplerate, 1e3, dsp)
}
