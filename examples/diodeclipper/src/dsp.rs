use enum_map::Enum;
use num_traits::Zero;

use valib::dsp::parameter::{HasParameters, Parameter, SmoothedParam};
use valib::dsp::{PerSampleBlockAdapter, DSP};
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

impl<T: Scalar> DSP<1, 1> for DcBlocker<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.0.process(x)
    }

    fn reset(&mut self) {
        self.0.reset()
    }

    fn latency(&self) -> usize {
        self.0.latency()
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        self.0.set_samplerate(samplerate);
        self.0.update_coefficients(&Biquad::highpass(
            T::from_f64((Self::CUTOFF_HZ / samplerate) as f64),
            T::from_f64(Self::Q as f64),
        ));
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

impl DiodeType {
    pub fn name(&self) -> String {
        match self {
            Self::Silicon => "Silicon",
            Self::Germanium => "Germanium",
            Self::Led => "LED",
        }
        .to_string()
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

impl DSP<1, 1> for DspInner {
    type Sample = Sample;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        if self.force_reset.has_changed() {
            DSP::reset(self)
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

    fn set_samplerate(&mut self, samplerate: f32) {
        DSP::set_samplerate(&mut self.nr_model, samplerate);
        DSP::set_samplerate(&mut self.nr_nr, samplerate);
    }

    fn latency(&self) -> usize {
        if self.model_switch.get_bool() {
            DSP::latency(&self.nr_model)
        } else {
            DSP::latency(&self.nr_nr)
        }
    }

    fn reset(&mut self) {
        DSP::reset(&mut self.nr_model);
        DSP::reset(&mut self.nr_nr);
    }
}

pub struct Dsp {
    inner: PerSampleBlockAdapter<Oversampled<Sample, DspInner>, 1, 1>,
    dc_blocker: DcBlocker<Sample>,
}

impl DSP<1, 1> for Dsp {
    type Sample = Sample;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.dc_blocker.process(self.inner.process(x))
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        DSP::set_samplerate(&mut self.inner, samplerate);
        DSP::set_samplerate(&mut self.dc_blocker, samplerate);
    }

    fn latency(&self) -> usize {
        DSP::latency(&self.inner) + DSP::latency(&self.dc_blocker)
    }

    fn reset(&mut self) {
        DSP::reset(&mut self.inner);
        DSP::reset(&mut self.dc_blocker);
    }
}

impl HasParameters for Dsp {
    type Enum = DspParams;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        self.inner.get_parameter(param)
    }
}

pub fn create_dsp(samplerate: f32, oversample: usize, max_block_size: usize) -> Dsp {
    let inner = PerSampleBlockAdapter::new(
        Oversample::new(oversample, max_block_size).with_dsp(DspInner::new(samplerate)),
    );
    Dsp {
        inner,
        dc_blocker: DcBlocker::new(samplerate),
    }
}
