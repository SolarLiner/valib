use nih_plug::nih_log;
use nih_plug::prelude::Enum;
use num_traits::Zero;

use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::{HasParameters, ParamId, ParamName, RemoteControlled, SmoothedParam};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, DSPProcessBlock};
use valib::filters::biquad::Biquad;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::clippers::{DiodeClipper, DiodeClipperModel};
use valib::saturators::Linear;
use valib::simd::{AutoF32x2, AutoF64x2, SimdComplexField};
use valib::{Scalar, SimdCast};

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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum, ParamName)]
pub enum DiodeType {
    Silicon,
    Germanium,
    #[name = "LED"]
    Led,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ParamName)]
pub enum DspParams {
    Drive,
    ModelSwitch,
    DiodeType,
    NumForward,
    NumBackward,
    ForceReset,
}

pub struct DspInner {
    drive: SmoothedParam,
    model_switch: bool,
    num_forward: u8,
    num_backward: u8,
    diode_type: DiodeType,
    force_reset: bool,
    nr_model: DiodeClipperModel<Sample64>,
    nr_nr: DiodeClipper<Sample64>,
}

impl DspInner {
    fn new(samplerate: f32) -> Self {
        Self {
            drive: SmoothedParam::exponential(1.0, samplerate, 10.0),
            model_switch: false,
            num_forward: 1,
            num_backward: 1,
            diode_type: DiodeType::Silicon,
            force_reset: false,
            nr_model: DiodeClipperModel::new_silicon(1, 1),
            nr_nr: DiodeClipper::new_silicon(1, 1, Sample64::zero()),
        }
    }

    fn update_from_params(&mut self) {
        let num_fwd = self.num_forward;
        let num_bck = self.num_backward;
        self.nr_model = match self.diode_type {
            DiodeType::Silicon => DiodeClipperModel::new_silicon(num_fwd, num_bck),
            DiodeType::Germanium => DiodeClipperModel::new_germanium(num_fwd, num_bck),
            DiodeType::Led => DiodeClipperModel::new_led(num_fwd, num_bck),
        };
        let last_vout = self.nr_nr.last_output();
        self.nr_nr = match self.diode_type {
            DiodeType::Silicon => {
                DiodeClipper::new_silicon(num_fwd as usize, num_bck as usize, last_vout)
            }
            DiodeType::Germanium => {
                DiodeClipper::new_germanium(num_fwd as usize, num_bck as usize, last_vout)
            }
            DiodeType::Led => DiodeClipper::new_led(num_fwd as usize, num_bck as usize, last_vout),
        };
    }
}

impl HasParameters for DspInner {
    type Name = DspParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        let mut do_update = false;
        match param {
            DspParams::Drive => {
                self.drive.param = value;
            }
            DspParams::ModelSwitch => {
                self.model_switch = value > 0.5;
            }
            DspParams::DiodeType => {
                self.diode_type =
                    DiodeType::from_id(value.clamp(0.0, DiodeType::variants().len() as _) as _);
                do_update = true;
            }
            DspParams::NumForward => {
                self.num_forward = value.clamp(1.0, 5.0) as _;
                do_update = true;
            }
            DspParams::NumBackward => {
                self.num_backward = value.clamp(1.0, 5.0) as _;
                do_update = true;
            }
            DspParams::ForceReset => {
                self.reset();
            }
        }
        if do_update {
            self.update_from_params();
        }
    }
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.drive.set_samplerate(samplerate);
        self.nr_model.set_samplerate(samplerate);
        self.nr_nr.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        if self.model_switch {
            self.nr_model.latency()
        } else {
            self.nr_nr.latency()
        }
    }

    fn reset(&mut self) {
        self.drive.reset();
        self.nr_model.reset();
        self.nr_nr.reset();
        self.force_reset = false;
    }
}

impl DSPProcess<1, 1> for DspInner {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let drive = self.drive.next_sample_as::<Sample64>();
        let x64 = x.map(|x| x.cast() * drive);
        if self.model_switch {
            self.nr_model.process(x64)
        } else {
            self.nr_nr.process(x64)
        }
        .map(|x| x / drive.simd_asinh())
        .map(|x| x.cast())
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
        self.inner.set_parameter(param, value);
    }
}

pub fn create_dsp(
    samplerate: f32,
    oversample: usize,
    max_block_size: usize,
) -> RemoteControlled<Dsp> {
    let inner = Oversample::new(oversample, max_block_size).with_dsp(
        samplerate,
        BlockAdapter(DspInner::new(samplerate * oversample as f32)),
    );
    let dsp = Dsp {
        inner,
        dc_blocker: DcBlocker::new(samplerate),
    };
    RemoteControlled::new(samplerate, 1e3, dsp)
}
