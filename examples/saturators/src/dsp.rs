use enum_map::{Enum, EnumArray};
use num_traits::Zero;

use valib::dsp::parameter::{HasParameters, Parameter, SmoothedParam};
use valib::dsp::utils::slice_to_mono_block_mut;
use valib::dsp::{DSPBlock, DSP};
use valib::filters::biquad::Biquad;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::adaa::{Adaa, Antiderivative, Antiderivative2};
use valib::saturators::clippers::DiodeClipperModel;
use valib::saturators::{Asinh, Clipper, Linear, Saturator, Tanh};
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
        DSP::reset(&mut self.0);
    }

    fn latency(&self) -> usize {
        DSP::latency(&self.0)
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        DSP::set_samplerate(&mut self.0, samplerate);
        self.0.update_coefficients(&Biquad::highpass(
            T::from_f64((Self::CUTOFF_HZ / samplerate) as f64),
            T::from_f64(Self::Q as f64),
        ));
    }
}

type Sample = AutoF32x2;
type Sample64 = AutoF64x2;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
pub enum SaturatorType {
    HardClip,
    Tanh,
    Asinh,
    DiodeSymmetric,
    DiodeAssymetric,
}

enum DspSaturatorDirect {
    HardClip,
    Tanh,
    Asinh,
    Diode(DiodeClipperModel<Sample64>),
}

impl DSP<1, 1> for DspSaturatorDirect {
    type Sample = Sample64;

    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = match self {
            Self::HardClip => Clipper.saturate(x),
            Self::Tanh => Tanh.saturate(x),
            Self::Asinh => Asinh.saturate(x),
            Self::Diode(clipper) => clipper.saturate(x),
        };
        [y]
    }
}

enum DspSaturatorAdaa1 {
    HardClip,
    Tanh,
    Asinh,
}

impl Antiderivative<Sample64> for DspSaturatorAdaa1 {
    fn evaluate(&self, x: Sample64) -> Sample64 {
        match self {
            Self::HardClip => Clipper.evaluate(x),
            Self::Tanh => Tanh.evaluate(x),
            Self::Asinh => Asinh.evaluate(x),
        }
    }

    fn antiderivative(&self, x: Sample64) -> Sample64 {
        match self {
            DspSaturatorAdaa1::HardClip => Clipper.antiderivative(x),
            DspSaturatorAdaa1::Tanh => Tanh.antiderivative(x),
            DspSaturatorAdaa1::Asinh => Asinh.antiderivative(x),
        }
    }
}

enum DspSaturatorAdaa2 {
    HardClip,
    Asinh,
}

impl Antiderivative<Sample64> for DspSaturatorAdaa2 {
    fn evaluate(&self, x: Sample64) -> Sample64 {
        match self {
            Self::HardClip => Clipper.evaluate(x),
            Self::Asinh => Asinh.evaluate(x),
        }
    }

    fn antiderivative(&self, x: Sample64) -> Sample64 {
        match self {
            Self::HardClip => Clipper.antiderivative(x),
            Self::Asinh => Asinh.antiderivative(x),
        }
    }
}

impl Antiderivative2<Sample64> for DspSaturatorAdaa2 {
    fn antiderivative2(&self, x: Sample64) -> Sample64 {
        match self {
            Self::HardClip => Clipper.antiderivative2(x),
            Self::Asinh => Asinh.antiderivative2(x),
        }
    }
}

enum DspSaturator {
    Direct(DspSaturatorDirect),
    Adaa1(Adaa<Sample64, DspSaturatorAdaa1, 1>),
    Adaa2(Adaa<Sample64, DspSaturatorAdaa2, 2>),
}

impl DSP<1, 1> for DspSaturator {
    type Sample = Sample64;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        match self {
            DspSaturator::Direct(sat) => sat.process(x),
            DspSaturator::Adaa1(adaa) => adaa.process(x),
            DspSaturator::Adaa2(adaa) => adaa.process(x),
        }
    }
}

impl SaturatorType {
    pub fn name(&self) -> String {
        match self {
            SaturatorType::HardClip => "Hard clip",
            SaturatorType::Tanh => "Tanh",
            SaturatorType::Asinh => "Asinh",
            SaturatorType::DiodeSymmetric => "Diode (symmetric)",
            SaturatorType::DiodeAssymetric => "Diode (assymetric)",
        }
        .to_string()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
pub enum DspInnerParams {
    Drive,
    Saturator,
    Feedback,
    AdaaLevel,
    AdaaEpsilon,
}

pub struct DspInner {
    drive: SmoothedParam,
    model_switch: Parameter,
    feedback: SmoothedParam,
    adaa_level: Parameter,
    adaa_epsilon: SmoothedParam,
    cur_saturator: DspSaturator,
    last_out: Sample64,
}

impl DspInner {
    fn new(samplerate: f32) -> Self {
        Self {
            drive: Parameter::new(1.0).smoothed_exponential(samplerate, 10.0),
            model_switch: Parameter::new(0.0),
            feedback: Parameter::new(0.0).smoothed_linear(samplerate, 100.0),
            adaa_level: Parameter::new(2.0),
            adaa_epsilon: Parameter::new(1e-4).smoothed_linear(samplerate, 100.0),
            cur_saturator: DspSaturator::Adaa2(Adaa::new(DspSaturatorAdaa2::HardClip)),
            last_out: Sample64::zero(),
        }
    }

    fn update_from_params(&mut self) {
        if self.model_switch.has_changed() || self.adaa_level.has_changed() {
            self.cur_saturator = match (
                self.model_switch.get_enum(),
                self.adaa_level.get_value() as u8,
            ) {
                (SaturatorType::HardClip, 0) => DspSaturator::Direct(DspSaturatorDirect::HardClip),
                (SaturatorType::HardClip, 1) => {
                    DspSaturator::Adaa1(Adaa::new(DspSaturatorAdaa1::HardClip))
                }
                (SaturatorType::HardClip, _) => {
                    DspSaturator::Adaa2(Adaa::new(DspSaturatorAdaa2::HardClip))
                }
                (SaturatorType::Tanh, 0) => DspSaturator::Direct(DspSaturatorDirect::Tanh),
                (SaturatorType::Tanh, _) => DspSaturator::Adaa1(Adaa::new(DspSaturatorAdaa1::Tanh)),
                (SaturatorType::Asinh, 0) => DspSaturator::Direct(DspSaturatorDirect::Asinh),
                (SaturatorType::Asinh, 1) => {
                    DspSaturator::Adaa1(Adaa::new(DspSaturatorAdaa1::Asinh))
                }
                (SaturatorType::Asinh, _) => {
                    DspSaturator::Adaa2(Adaa::new(DspSaturatorAdaa2::Asinh))
                }
                (SaturatorType::DiodeSymmetric, _) => DspSaturator::Direct(
                    DspSaturatorDirect::Diode(DiodeClipperModel::new_silicon(1, 1)),
                ),
                (SaturatorType::DiodeAssymetric, _) => DspSaturator::Direct(
                    DspSaturatorDirect::Diode(DiodeClipperModel::new_germanium(1, 2)),
                ),
            }
        }
        let adaa_epsilon = Sample64::from_f64(self.adaa_epsilon.next_sample() as _);
        match &mut self.cur_saturator {
            DspSaturator::Adaa1(adaa) => adaa.epsilon = adaa_epsilon,
            DspSaturator::Adaa2(adaa) => adaa.epsilon = adaa_epsilon,
            _ => {}
        }
    }
}

impl HasParameters for DspInner {
    type Enum = DspInnerParams;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        match param {
            DspInnerParams::Drive => &self.drive.param,
            DspInnerParams::Saturator => &self.model_switch,
            DspInnerParams::Feedback => &self.feedback.param,
            DspInnerParams::AdaaLevel => &self.adaa_level,
            DspInnerParams::AdaaEpsilon => &self.adaa_epsilon.param,
        }
    }
}

impl DSP<1, 1> for DspInner {
    type Sample = Sample;

    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.update_from_params();

        let drive = Sample64::from_f64(self.drive.next_sample() as _);
        let feedback = Sample64::from_f64(self.feedback.next_sample() as _);
        let x64 = x.cast() * drive + self.last_out * feedback;
        let [y] = self.cur_saturator.process([x64]);
        self.last_out = y;
        let yout = y / drive.simd_asinh();
        [yout.cast()]
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        DSP::set_samplerate(&mut self.drive, samplerate);
        DSP::set_samplerate(&mut self.feedback, samplerate);
    }

    fn latency(&self) -> usize {
        DSP::latency(&self.cur_saturator)
    }

    fn reset(&mut self) {
        DSP::reset(&mut self.cur_saturator);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DspParams {
    InnerParam(DspInnerParams),
    DcBlocker,
    Oversampling,
}

impl Enum for DspParams {
    const LENGTH: usize = DspInnerParams::LENGTH + 2;

    fn from_usize(value: usize) -> Self {
        if value < DspInnerParams::LENGTH {
            DspParams::InnerParam(DspInnerParams::from_usize(value))
        } else {
            match value - DspInnerParams::LENGTH {
                0 => Self::DcBlocker,
                1 => Self::Oversampling,
                _ => unreachable!(),
            }
        }
    }

    fn into_usize(self) -> usize {
        match self {
            Self::InnerParam(dsp_param) => dsp_param.into_usize(),
            Self::DcBlocker => DspInnerParams::LENGTH,
            Self::Oversampling => DspInnerParams::LENGTH + 1,
        }
    }
}

impl<T> EnumArray<T> for DspParams {
    type Array = [T; Self::LENGTH];
}

pub struct Dsp {
    use_dc_blocker: Parameter,
    oversample_amount: Parameter,
    inner: Oversampled<Sample, DspInner>,
    dc_blocker: DcBlocker<Sample>,
    dc_blocker_staging: Box<[Sample]>,
}

impl DSPBlock<1, 1> for Dsp {
    type Sample = Sample;

    fn process_block(&mut self, inputs: &[[Self::Sample; 1]], outputs: &mut [[Self::Sample; 1]]) {
        let staging = slice_to_mono_block_mut(&mut self.dc_blocker_staging[..inputs.len()]);

        self.inner
            .set_oversampling_amount(self.oversample_amount.get_value() as _);
        self.inner.process_block(inputs, staging);
        if self.use_dc_blocker.get_bool() {
            self.dc_blocker.process_block(staging, outputs);
        } else {
            outputs.copy_from_slice(staging);
        }
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        DSPBlock::set_samplerate(&mut self.inner, samplerate);
        DSPBlock::set_samplerate(&mut self.dc_blocker, samplerate);
    }

    fn latency(&self) -> usize {
        let inner_latency = DSPBlock::latency(&self.inner);
        if self.use_dc_blocker.get_bool() {
            inner_latency + DSPBlock::latency(&self.dc_blocker)
        } else {
            inner_latency
        }
    }

    fn reset(&mut self) {
        DSPBlock::reset(&mut self.inner);
        DSPBlock::reset(&mut self.dc_blocker);
    }
}

impl HasParameters for Dsp {
    type Enum = DspParams;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        match param {
            DspParams::InnerParam(inner) => self.inner.get_parameter(inner),
            DspParams::DcBlocker => &self.use_dc_blocker,
            DspParams::Oversampling => &self.oversample_amount,
        }
    }
}

pub fn create_dsp(samplerate: f32, oversample: usize, max_block_size: usize) -> Dsp {
    let inner = Oversample::new(oversample, max_block_size).with_dsp(DspInner::new(samplerate));
    Dsp {
        inner,
        oversample_amount: Parameter::new(2.0),
        use_dc_blocker: Parameter::new(1.0),
        dc_blocker: DcBlocker::new(samplerate),
        dc_blocker_staging: vec![Sample::zero(); max_block_size].into_boxed_slice(),
    }
}
