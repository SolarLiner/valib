use nih_plug::prelude::Enum;
use num_traits::{One, Zero};
use std::borrow::Cow;

use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::{HasParameters, ParamId, ParamName, RemoteControlled, SmoothedParam};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, DSPProcessBlock};
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
        self.0.reset();
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
pub enum SaturatorType {
    #[name = "Hard clip"]
    HardClip,
    Tanh,
    Asinh,
    #[name = "Diode (sym.)"]
    DiodeSymmetric,
    #[name = "Diode (asym.)"]
    DiodeAssymetric,
}

enum DspSaturatorDirect {
    HardClip,
    Tanh,
    Asinh,
    Diode(DiodeClipperModel<Sample64>),
}

impl DSPMeta for DspSaturatorDirect {
    type Sample = Sample64;
}

impl DSPProcess<1, 1> for DspSaturatorDirect {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = match self {
            Self::HardClip => Clipper::default().saturate(x),
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
            Self::HardClip => Clipper {
                min: -Sample64::one(),
                max: Sample64::one(),
            }
            .evaluate(x),
            Self::Tanh => Tanh.evaluate(x),
            Self::Asinh => Asinh.evaluate(x),
        }
    }

    fn antiderivative(&self, x: Sample64) -> Sample64 {
        match self {
            DspSaturatorAdaa1::HardClip => Clipper {
                min: -Sample64::one(),
                max: Sample64::one(),
            }
            .antiderivative(x),
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
            Self::HardClip => Clipper {
                min: -Sample64::one(),
                max: Sample64::one(),
            }
            .evaluate(x),
            Self::Asinh => Asinh.evaluate(x),
        }
    }

    fn antiderivative(&self, x: Sample64) -> Sample64 {
        match self {
            Self::HardClip => Clipper {
                min: -Sample64::one(),
                max: Sample64::one(),
            }
            .antiderivative(x),
            Self::Asinh => Asinh.antiderivative(x),
        }
    }
}

impl Antiderivative2<Sample64> for DspSaturatorAdaa2 {
    fn antiderivative2(&self, x: Sample64) -> Sample64 {
        match self {
            Self::HardClip => Clipper {
                min: -Sample64::one(),
                max: Sample64::one(),
            }
            .antiderivative2(x),
            Self::Asinh => Asinh.antiderivative2(x),
        }
    }
}

enum DspSaturator {
    Direct(DspSaturatorDirect),
    Adaa1(Adaa<Sample64, DspSaturatorAdaa1, 1>),
    Adaa2(Adaa<Sample64, DspSaturatorAdaa2, 2>),
}

impl DSPMeta for DspSaturator {
    type Sample = Sample64;
}

impl DSPProcess<1, 1> for DspSaturator {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, ParamName)]
pub enum DspInnerParams {
    Drive,
    Saturator,
    Feedback,
    AdaaLevel,
    AdaaEpsilon,
}

pub struct DspInner {
    drive: SmoothedParam,
    model_switch: SaturatorType,
    feedback: SmoothedParam,
    adaa_level: u8,
    adaa_epsilon: SmoothedParam,
    cur_saturator: DspSaturator,
    last_out: Sample64,
}

impl DspInner {
    fn new(samplerate: f32) -> Self {
        Self {
            drive: SmoothedParam::exponential(1.0, samplerate, 10.0),
            model_switch: SaturatorType::Tanh,
            feedback: SmoothedParam::linear(0.0, samplerate, 10.0),
            adaa_level: 0,
            adaa_epsilon: SmoothedParam::linear(1e-4, samplerate, 100.0),
            cur_saturator: DspSaturator::Adaa2(Adaa::new(DspSaturatorAdaa2::HardClip)),
            last_out: Sample64::zero(),
        }
    }

    fn update_from_params(&mut self) {
        self.cur_saturator = match (self.model_switch, self.adaa_level) {
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
            (SaturatorType::Asinh, 1) => DspSaturator::Adaa1(Adaa::new(DspSaturatorAdaa1::Asinh)),
            (SaturatorType::Asinh, _) => DspSaturator::Adaa2(Adaa::new(DspSaturatorAdaa2::Asinh)),
            (SaturatorType::DiodeSymmetric, _) => DspSaturator::Direct(DspSaturatorDirect::Diode(
                DiodeClipperModel::new_silicon(1, 1),
            )),
            (SaturatorType::DiodeAssymetric, _) => DspSaturator::Direct(DspSaturatorDirect::Diode(
                DiodeClipperModel::new_germanium(1, 2),
            )),
        };
        let adaa_epsilon = Sample64::from_f64(self.adaa_epsilon.next_sample() as _);
        match &mut self.cur_saturator {
            DspSaturator::Adaa1(adaa) => adaa.epsilon = adaa_epsilon,
            DspSaturator::Adaa2(adaa) => adaa.epsilon = adaa_epsilon,
            _ => {}
        }
    }
}

impl HasParameters for DspInner {
    type Name = DspInnerParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            DspInnerParams::Drive => {
                self.drive.param = value;
            }
            DspInnerParams::Saturator => {
                self.model_switch = SaturatorType::from_index(value as _);
                self.update_from_params();
            }
            DspInnerParams::Feedback => {
                self.feedback.param = value;
            }
            DspInnerParams::AdaaLevel => {
                self.adaa_level = value.clamp(0.0, 2.0) as _;
                self.update_from_params();
            }
            DspInnerParams::AdaaEpsilon => {
                self.adaa_epsilon.param = value;
            }
        }
    }
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.adaa_epsilon.set_samplerate(samplerate);
        self.drive.set_samplerate(samplerate);
        self.feedback.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.cur_saturator.latency()
    }

    fn reset(&mut self) {
        self.adaa_epsilon.reset();
        self.drive.reset();
        self.feedback.reset();
        self.cur_saturator.reset();
    }
}

impl DSPProcess<1, 1> for DspInner {
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
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DspParams {
    InnerParam(DspInnerParams),
    DcBlocker,
    Oversampling,
}

impl ParamName for DspParams {
    fn count() -> usize {
        DspInnerParams::count() + 2
    }

    fn from_id(value: ParamId) -> Self {
        if value < DspInnerParams::count() as ParamId {
            DspParams::InnerParam(DspInnerParams::from_id(value))
        } else {
            match value - DspInnerParams::count() {
                0 => Self::DcBlocker,
                1 => Self::Oversampling,
                _ => unreachable!(),
            }
        }
    }

    fn into_id(self) -> ParamId {
        match self {
            Self::InnerParam(dsp_param) => dsp_param.into_id(),
            Self::DcBlocker => DspInnerParams::count(),
            Self::Oversampling => DspInnerParams::count() + 1,
        }
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("") // unused
    }
}

pub struct Dsp {
    use_dc_blocker: bool,
    oversample_amount: usize,
    inner: Oversampled<Sample, BlockAdapter<DspInner>>,
    dc_blocker: BlockAdapter<DcBlocker<Sample>>,
    dc_blocker_staging: Box<[Sample]>,
}

impl DSPMeta for Dsp {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner.set_samplerate(samplerate);
        self.dc_blocker.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        let inner_latency = self.inner.latency();
        if self.use_dc_blocker {
            inner_latency + self.dc_blocker.latency()
        } else {
            inner_latency
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.dc_blocker.reset();
    }
}

impl DSPProcessBlock<1, 1> for Dsp {
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Sample, 1>,
        mut outputs: AudioBufferMut<Sample, 1>,
    ) {
        let mut staging = AudioBufferMut::from(&mut self.dc_blocker_staging[..inputs.samples()]);

        self.inner.process_block(inputs, staging.as_mut());
        if self.use_dc_blocker {
            self.dc_blocker
                .process_block(staging.as_ref(), outputs.as_mut());
        } else {
            outputs.copy_from(staging.as_ref());
        }
    }
}

impl HasParameters for Dsp {
    type Name = DspParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            DspParams::InnerParam(p) => self.inner.set_parameter(p, value),
            DspParams::DcBlocker => {
                self.use_dc_blocker = value > 0.5;
            }
            DspParams::Oversampling => {
                self.oversample_amount = usize::pow(2, value as _);
                self.inner.set_oversampling_amount(self.oversample_amount);
            }
        }
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
        oversample_amount: 2,
        use_dc_blocker: true,
        dc_blocker: BlockAdapter(DcBlocker::new(samplerate)),
        dc_blocker_staging: vec![Sample::zero(); max_block_size].into_boxed_slice(),
    };
    RemoteControlled::new(samplerate, 1e3, dsp)
}
