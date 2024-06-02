use nalgebra::Complex;
use nih_plug::nih_log;
use nih_plug::util::db_to_gain_fast;
use numeric_literals::replace_float_literals;
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, DSPProcessBlock};
use valib::dsp::blocks::Series;
use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::{HasParameters, ParamId, ParamName, RemoteControlled, SmoothedParam};
use valib::filters::statespace::StateSpace;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::{Clipper, Saturator, Slew};
use valib::Scalar;
use valib::simd::{AutoF32x2, AutoF64x2, AutoSimd, SimdBool, SimdComplexField, SimdPartialOrd};
use valib::util::lerp;

#[replace_float_literals(T::from_f64(literal))]
fn smooth_min<T: Scalar>(t: T, a: T, b: T) -> T {
    // Polynomial
    //let h = (0.5 + 0.5 * (a - b) / t).simd_clamp(0.0, 1.0);
    //lerp(h, a, b) - t * h * (1.0 - h)

    // Exponential
    let r = (-a / t).simd_exp2() + (-b / t).simd_exp2();
    -t * r.simd_log2()
}

fn smooth_max<T: Scalar>(t: T, a: T, b: T) -> T {
    -smooth_min(t, -a, -b)
}

fn smooth_clamp<T: Scalar>(t: T, x: T, min: T, max: T) -> T {
    smooth_max(t, min, smooth_min(t, x, max))
}

#[derive(Debug, Copy, Clone)]
struct Bjt<T> {
    pub vcc: T,
    pub vee: T,
}

impl<T: Scalar> Default for Bjt<T> {
    fn default() -> Self {
        Self {
            vcc: T::from_f64(4.5),
            vee: T::from_f64(-4.5),
        }
    }
}

impl<T: Scalar> Saturator<T> for Bjt<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn saturate(&self, x: T) -> T {
        smooth_clamp(0.1, x - 0.770, self.vee, self.vcc) + 0.770
    }
}

impl<T: Scalar> DSPMeta for Bjt<T> {
    type Sample = T;
}

impl<T: Scalar> DSPProcess<1, 1> for Bjt<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.saturate(x)]
    }
}

#[derive(Debug, Clone)]
pub struct Bypass<P> {
    pub inner: P,
    pub active: SmoothedParam,
}

impl<P: DSPMeta> DSPMeta for Bypass<P> {
    type Sample = P::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.active.set_samplerate(samplerate);
        self.inner.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        let active = self.active.param > 0.5;
        if active {
            self.inner.latency()
        } else {
            0
        }
    }

    fn reset(&mut self) {
        self.active.reset();
        self.inner.reset();
    }
}

impl<P, const N: usize> DSPProcess<N, N> for Bypass<P>
where
    P: DSPProcess<N, N>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        let active = P::Sample::from_f64(self.active.next_sample() as _);
        let processed = self.inner.process(x);
        std::array::from_fn(|i| {
            let processed = active
                .simd_gt(P::Sample::from_f64(1e-6))
                .if_else(|| processed[i], || x[i]);
            lerp(active, x[i], processed)
        })
    }
}

impl<T> Bypass<T> {
    pub fn new(samplerate: f32, inner: T) -> Self {
        Self {
            inner,
            active: SmoothedParam::linear(1., samplerate, 15.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputStage<T: Scalar> {
    pub gain: SmoothedParam,
    state_space: StateSpace<T, 1, 1, 1>,
    clip: Bjt<T>,
}

impl<T: Scalar> DSPMeta for InputStage<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.gain.set_samplerate(samplerate);
        self.state_space.set_samplerate(samplerate);
        self.clip.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.state_space.latency() + self.clip.latency()
    }

    fn reset(&mut self) {
        self.gain.reset();
        self.state_space.reset();
        self.clip.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for InputStage<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let gain = T::from_f64(self.gain.next_sample() as _);
        let y = self.state_space.process([x * gain]);
        self.clip.process(y)
    }
}

impl<T: Scalar> InputStage<T> {
    pub fn new(samplerate: f32, gain: f32) -> Self {
        Self {
            gain: SmoothedParam::exponential(gain, samplerate, 10.0),
            state_space: crate::gen::input(T::from_f64(samplerate as _).simd_recip()),
            clip: Bjt::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClipperStage<T: Scalar> {
    dt: T,
    dist: SmoothedParam,
    state_space: StateSpace<T, 1, 3, 1>,
    feedback_gain: SmoothedParam,
    feedback_sample: T,
    slew: Slew<T>,
}

impl<T: Scalar> DSPMeta for ClipperStage<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.dt = T::from_f64(samplerate.recip() as _);
        self.feedback_gain.set_samplerate(samplerate);
        self.state_space.set_samplerate(samplerate);
        self.slew.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.state_space.latency() + self.slew.latency()
    }

    fn reset(&mut self) {
        self.state_space.reset();
        self.slew.reset();
        self.feedback_sample = T::zero();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for ClipperStage<T> {
    #[replace_float_literals(Self::Sample::from_f64(literal))]
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let dist = self.dist.next_sample_as();
        self.update_state_matrices(self.dt, dist);
        let [y] = self.state_space.process([
            x - T::from_f64(self.feedback_gain.next_sample() as _) * self.feedback_sample
        ]);
        let y = y.simd_asinh().simd_clamp(-4.5, 4.5);
        let [y] = self.slew.process([y]);
        self.feedback_sample = y;
        [y]
    }
}

impl<T: Scalar> ClipperStage<T> {
    pub fn new(samplerate: f32, dist: T) -> Self {
        let dt = T::from_f64(samplerate as _).simd_recip();
        Self {
            dt,
            dist: SmoothedParam::exponential(1.0, samplerate, 50.0),
            state_space: crate::gen::clipper(dt, dist),
            feedback_gain: SmoothedParam::exponential(0.0, samplerate, 50.0),
            feedback_sample: T::zero(),
            slew: Slew::new(T::from_f64(samplerate as _), T::from_f64(1e4) * dt),
        }
    }

    fn update_state_matrices(&mut self, dt: T, dist: T) {
        self.state_space
            .update_matrices(&crate::gen::clipper(dt, dist));
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ToneStage<T: Scalar> {
    tone: SmoothedParam,
    state_space: StateSpace<T, 1, 4, 1>,
    out_gain: SmoothedParam,
    dt: T,
}

impl<T: Scalar> DSPMeta for ToneStage<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.dt = T::from_f64(samplerate.recip() as _);
        self.state_space.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.state_space.latency()
    }

    fn reset(&mut self) {
        self.state_space.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for ToneStage<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let tone = self.tone.next_sample_as();
        self.update_state_matrices(self.dt, tone);
        let y = self.state_space.process(x);
        let vcc = T::from_f64(4.5);
        std::array::from_fn(|i| {
            y[i].simd_clamp(-vcc, vcc) * self.out_gain.next_sample_as::<T>() * T::from_f64(0.25)
        })
    }
}

impl<T: Scalar> ToneStage<T> {
    pub fn new(samplerate: f32, tone: T) -> Self {
        let dt = T::from_f64(samplerate.recip() as _);
        Self {
            dt,
            tone: SmoothedParam::linear(0.0, samplerate, 15.0),
            state_space: crate::gen::tone(dt, tone),
            out_gain: SmoothedParam::exponential(1., samplerate, 15.0),
        }
    }

    pub fn update_state_matrices(&mut self, dt: T, tone: T) {
        self.state_space
            .update_matrices(&crate::gen::tone(dt, tone));
    }
}

#[derive(Debug, Clone)]
pub struct OutputStage<T: Scalar> {
    inner: Bypass<StateSpace<T, 1, 2, 1>>,
    clip: Bjt<T>,
}

impl<T: Scalar> OutputStage<T> {
    pub fn new(samplerate: f32) -> Self {
        let dt = T::from_f64(samplerate as _).simd_recip();
        Self {
            inner: Bypass::new(samplerate, crate::gen::output(dt)),
            clip: Bjt::default(),
        }
    }
}

impl<T: Scalar> DSPMeta for OutputStage<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner.set_samplerate(samplerate);
        self.clip.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.inner.latency() + self.clip.latency()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.clip.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for OutputStage<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = self.inner.process(x);
        self.clip.process(y)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, ParamName)]
pub enum DspParams {
    Bypass,
    InputGain,
    Distortion,
    Tone,
    OutputGain,
    ComponentMismatch,
    BufferBypass,
}

type DspInner<T> = Bypass<
    Series<(
        Bypass<InputStage<T>>,
        ClipperStage<T>,
        ToneStage<T>,
        Bypass<OutputStage<T>>,
    )>,
>;

pub struct Dsp<T: Scalar> {
    inner: Oversampled<T, BlockAdapter<DspInner<T>>>,
    dt: T,
}

impl<T: Scalar> DSPMeta for Dsp<T> {
    type Sample = T;
}

impl<T: Scalar> DSPProcessBlock<1, 1> for Dsp<T> {
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, 1>,
        outputs: AudioBufferMut<Self::Sample, 1>,
    ) {
        self.inner.process_block(inputs, outputs);
    }
}

impl<T: Scalar> HasParameters for Dsp<T> {
    type Name = DspParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        nih_log!("Set parameter {param:?} {value}");
        let Bypass {
            active,
            inner:
                Series((
                    Bypass {
                        active: input_active,
                        inner: input,
                    },
                    clipper,
                    tone,
                    Bypass {
                        active: output_active,
                        ..
                    },
                )),
        } = &mut self.inner.inner.0;
        match param {
            DspParams::Bypass => {
                active.param = 1. - value;
            }
            DspParams::InputGain => {
                input.gain.param = value;
            }
            DspParams::Distortion => {
                clipper.dist.param = value;
            }
            DspParams::Tone => {
                tone.tone.param = value;
            }
            DspParams::OutputGain => {
                tone.out_gain.param = value;
            }
            DspParams::ComponentMismatch => {
                clipper.slew.max_diff =
                    component_matching_slew_rate(self.dt.simd_recip(), T::from_f64(value as _));
                clipper.feedback_gain.param = lerp(value, -0.5, 0.0);
            }
            DspParams::BufferBypass => {
                input_active.param = 1. - value;
                output_active.param = 1. - value;
            }
        }
    }
}

impl<T: Scalar> Dsp<T>
where
    Complex<T>: SimdComplexField,
{
    pub fn new(base_samplerate: f32, target_samplerate: f32) -> RemoteControlled<Self> {
        let oversample = target_samplerate / base_samplerate;
        let oversample = oversample.ceil() as usize;
        let samplerate = base_samplerate * oversample as f32;

        let inner = Series((
            Bypass::new(samplerate, InputStage::new(samplerate, 1.0)),
            ClipperStage::new(samplerate, T::zero()),
            ToneStage::new(samplerate, T::zero()),
            Bypass::new(samplerate, OutputStage::new(samplerate)),
        ));
        let inner = DspInner::new(samplerate, inner);
        let inner = Oversample::new(oversample, MAX_BLOCK_SIZE)
            .with_dsp(base_samplerate, BlockAdapter(inner));
        RemoteControlled::new(
            base_samplerate,
            1e3,
            Dsp {
                inner,
                dt: T::from_f64(samplerate as _).simd_recip(),
            },
        )
    }
}

pub const MAX_BLOCK_SIZE: usize = 512;

fn component_matching_slew_rate<T: Scalar>(samplerate: T, normalized: T) -> T {
    let min = T::from_f64(db_to_gain_fast(30.0) as _);
    let max = T::from_f64(db_to_gain_fast(100.0) as _);
    lerp(normalized, min, max) / samplerate
}
