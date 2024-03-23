use std::fmt;
use std::fmt::Formatter;

use enum_map::Enum;
use num_traits::Zero;

use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::{HasParameters, Parameter, SmoothedParam};
use valib::dsp::{DSPMeta, DSPProcess, DSPProcessBlock};
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
pub enum DiodeType {
    Silicon,
    Germanium,
    Led,
}

impl fmt::Display for DiodeType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Silicon => write!(f, "Silicon"),
            Self::Germanium => write!(f, "Germanium"),
            Self::Led => write!(f, "LED"),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
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
    model_switch: Parameter,
    num_forward: Parameter,
    num_backward: Parameter,
    diode_type: Parameter,
    force_reset: Parameter,
    nr_model: DiodeClipperModel<Sample64>,
    nr_nr: DiodeClipper<Sample64>,
}

impl DspInner {
    fn new(samplerate: f32) -> Self {
        Self {
            drive: Parameter::new(1.0).smoothed_exponential(samplerate, 10.0),
            model_switch: Parameter::new(0.0),
            num_forward: Parameter::new(1.0),
            num_backward: Parameter::new(1.0),
            diode_type: Parameter::new(0.0),
            force_reset: Parameter::new(0.0),
            nr_model: DiodeClipperModel::new_silicon(1, 1),
            nr_nr: DiodeClipper::new_silicon(1, 1, Sample64::zero()),
        }
    }

    fn update_from_params(&mut self) {
        if self.num_forward.has_changed()
            || self.num_backward.has_changed()
            || self.diode_type.has_changed()
        {
            let num_fwd = self.num_forward.get_value() as _;
            let num_bck = self.num_backward.get_value() as _;
            self.nr_model = match self.diode_type.get_enum::<DiodeType>() {
                DiodeType::Silicon => DiodeClipperModel::new_silicon(num_fwd, num_bck),
                DiodeType::Germanium => DiodeClipperModel::new_germanium(num_fwd, num_bck),
                DiodeType::Led => DiodeClipperModel::new_led(num_fwd, num_bck),
            };
            let last_vout = self.nr_nr.last_output();
            self.nr_nr = match self.diode_type.get_enum::<DiodeType>() {
                DiodeType::Silicon => {
                    DiodeClipper::new_silicon(num_fwd as usize, num_bck as usize, last_vout)
                }
                DiodeType::Germanium => {
                    DiodeClipper::new_germanium(num_fwd as usize, num_bck as usize, last_vout)
                }
                DiodeType::Led => {
                    DiodeClipper::new_led(num_fwd as usize, num_bck as usize, last_vout)
                }
            };
        }
    }
}

impl HasParameters for DspInner {
    type Enum = DspParams;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        match param {
            DspParams::Drive => &self.drive.param,
            DspParams::ModelSwitch => &self.model_switch,
            DspParams::DiodeType => &self.diode_type,
            DspParams::NumForward => &self.num_forward,
            DspParams::NumBackward => &self.num_backward,
            DspParams::ForceReset => &self.force_reset,
        }
    }
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.nr_model.set_samplerate(samplerate);
        self.nr_nr.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        if self.model_switch.get_bool() {
            self.nr_model.latency()
        } else {
            self.nr_nr.latency()
        }
    }

    fn reset(&mut self) {
        self.nr_model.reset();
        self.nr_nr.reset();
    }
}

impl DSPProcess<1, 1> for DspInner {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        if self.force_reset.has_changed() {
            self.reset()
        }
        self.update_from_params();

        let drive = Sample64::from_f64(self.drive.next_sample() as _);
        let x64 = x.map(|x| x.cast() * drive);
        if self.model_switch.get_bool() {
            self.nr_model.process(x64)
        } else {
            self.nr_nr.process(x64)
        }
        .map(|x| x / drive.simd_asinh())
        .map(|x| x.cast())
    }
}

pub struct Dsp {
    inner: Oversampled<Sample, DspInner>,
    max_oversampling: usize,
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
    type Enum = DspParams;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        self.inner.get_parameter(param)
    }
}

pub fn create_dsp(samplerate: f32, oversample: usize, max_block_size: usize) -> Dsp {
    let mut inner =
        Oversample::new(oversample, max_block_size).with_dsp(samplerate, DspInner::new(samplerate));
    Dsp {
        inner,
        max_oversampling: oversample,
        dc_blocker: DcBlocker::new(samplerate),
    }
}
