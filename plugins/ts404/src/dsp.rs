use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use nalgebra::Complex;
use nih_plug::nih_log;
use nih_plug::prelude::{AtomicF32, Enum};
use nih_plug_vizia::vizia::prelude::Data;
use num_traits::{Float, ToPrimitive};
use numeric_literals::replace_float_literals;
use valib::{dsp::blocks::{Series, SwitchAB}, math::{smooth_max, smooth_min}};
use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::parameter::{HasParameters, ParamId, ParamName, RemoteControlled, SmoothedParam};
use valib::dsp::{blocks::Bypass, BlockAdapter, DSPMeta, DSPProcess, DSPProcessBlock};
use valib::filters::statespace::StateSpace;
use valib::math::smooth_clamp;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::{bjt, Saturator};
use valib::simd::{
    AutoF64x2, AutoSimd, SimdBool, SimdComplexField, SimdPartialOrd, SimdValue,
};
use valib::Scalar;

use clipping::ClippingStage;

mod clipping;

fn emitter_follower_input<T: Scalar>() -> bjt::CommonCollector<T> {
    bjt::CommonCollector {
        vee: T::zero(),
        vcc: T::from_f64(9.),
        xbias: T::from_f64(3.75041272),
        ybias: T::zero(),
    }
}

#[derive(Debug, Copy, Clone)]
struct OutputEmitterFollower<T> {
    pub t: T,
    pub xbias: T,
    pub kbias: T,
    pub ksat: T,
}

impl<T: Scalar> Default for OutputEmitterFollower<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn default() -> Self {
        Self {
            t: 0.04416026,
            xbias: -0.74491909,
            kbias: 8.43027281,
            ksat: 0.48964627,
        }
    }
}

impl<T: Scalar> Saturator<T> for OutputEmitterFollower<T> {
    fn saturate(&self, x: T) -> T {
        smooth_max(self.t, T::zero(), smooth_min(self.t, x + self.xbias, (x + self.kbias) * self.ksat))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Enum, Data)]
pub enum InputLevelMatching {
    Instrument,
    #[name="Instrument (Hot)"]
    InstrumentHot,
    Line,
    Eurorack,
}

impl fmt::Display for InputLevelMatching {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Self::variants()[self.to_index()])
    }
}

impl InputLevelMatching {
    #[replace_float_literals(T::from_f64(literal))]
    fn input_scale<T: Scalar>(&self) -> T {
        match self {
            Self::Instrument => 0.1833,
            Self::InstrumentHot => 0.55,
            Self::Line => 1.1,
            Self::Eurorack => 10.,
        }
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn output_scale<T: Scalar>(&self) -> T {
        match self {
            Self::Instrument => 3.,
            Self::InstrumentHot => 1.,
            Self::Line => 0.5,
            Self::Eurorack => 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputStage<T: Scalar> {
    pub gain: SmoothedParam,
    state_space: StateSpace<T, 1, 1, 1>,
    clip: bjt::CommonCollector<T>,
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

#[profiling::all_functions]
impl<T: fmt::Debug + Scalar> DSPProcess<1, 1> for InputStage<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let gain = self.gain.next_sample_as();
        let [y] = self.state_space.process([x * gain]);
        self.clip.process([y + T::from_f64(4.5)])
    }
}

impl<T: Scalar> InputStage<T> {
    pub fn new(samplerate: f32, gain: f32) -> Self {
        Self {
            gain: SmoothedParam::exponential(gain, samplerate, 100.0),
            state_space: crate::gen::input(T::from_f64(samplerate as _).simd_recip()),
            clip: emitter_follower_input(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ToneStage<T: Scalar> {
    tone: SmoothedParam,
    state_space: StateSpace<T, 1, 4, 1>,
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

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<1, 1> for ToneStage<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let tone = self.tone.next_sample_as();
        self.update_state_matrices(self.dt, tone);
        let [y] = self.state_space.process(x);
        [smooth_clamp(T::from_f64(0.1), y, T::zero(), T::from_f64(9.))]
    }
}

impl<T: Scalar> ToneStage<T> {
    pub fn new(samplerate: f32, tone: T) -> Self {
        let dt = T::from_f64(samplerate.recip() as _);
        Self {
            dt,
            tone: SmoothedParam::linear(0.0, samplerate, 15.0),
            state_space: crate::gen::tone(dt, tone),
        }
    }

    pub fn update_state_matrices(&mut self, dt: T, tone: T) {
        self.state_space
            .update_matrices(&crate::gen::tone(dt, tone));
    }
}

#[derive(Debug, Clone)]
pub struct OutputStage<T: Scalar> {
    inner: StateSpace<T, 1, 2, 1>,
    clip: OutputEmitterFollower<T>,
    out_gain: SmoothedParam,
}

impl<T: Scalar> OutputStage<T> {
    pub fn new(samplerate: f32) -> Self {
        let dt = T::from_f64(samplerate as _).simd_recip();
        Self {
            inner: crate::gen::output(dt),
            clip: OutputEmitterFollower::default(),
            out_gain: SmoothedParam::exponential(1., samplerate, 100.0),
        }
    }
}

impl<T: Scalar> DSPMeta for OutputStage<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner.update_matrices(&crate::gen::output(T::from_f64(
            (samplerate as f64).recip(),
        )));
    }

    fn latency(&self) -> usize {
        self.inner.latency()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[profiling::all_functions]
impl<T: fmt::Debug + Scalar> DSPProcess<1, 1> for OutputStage<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let [y] = self.inner.process([self.clip.saturate(x)]);
        let out_gain = self.out_gain.next_sample_as::<T>();
        let out = y * out_gain;
        [out]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, ParamName)]
pub enum DspParams {
    Bypass,
    InputMode,
    Distortion,
    Tone,
    ComponentMismatch,
}

type DspInner<T> = SwitchAB<
    Bypass<T>,
    Series<(
        InputStage<T>,
        ClippingStage<T>,
        ToneStage<T>,
        OutputStage<T>,
    )>,
>;

pub struct Dsp<T: Scalar<Element: Float>> {
    inner: Oversampled<T, BlockAdapter<DspInner<T>>>,
}

impl<T: Scalar<Element: Float>> DSPMeta for Dsp<T> {
    type Sample = T;
}

#[profiling::all_functions]
impl<T: Scalar<Element: Float>> DSPProcessBlock<1, 1> for Dsp<T>
where
    <T as SimdValue>::Element: ToPrimitive,
{
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, 1>,
        outputs: AudioBufferMut<Self::Sample, 1>,
    ) {
        self.inner.process_block(inputs, outputs);
    }
}

impl<T: Scalar<Element: Float>> HasParameters for Dsp<T> {
    type Name = DspParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        nih_log!("Set parameter {param:?} {value}");
        let SwitchAB {
            b: Series((input, clipping, tone, output)),
            ..
        } = &mut self.inner.inner.0;
        let mut bypass = None;
        match param {
            DspParams::Bypass => {
                let b_active = value < 0.5;
                bypass = Some(b_active);
            }
            DspParams::InputMode => {
                let level_matching = InputLevelMatching::from_index(value as _);
                input.gain.param = level_matching.input_scale();
                output.out_gain.param = level_matching.output_scale();
            }
            DspParams::Distortion => {
                clipping.set_dist(T::from_f64(value as _));
            }
            DspParams::Tone => {
                tone.tone.param = value;
            }
            DspParams::ComponentMismatch => {
                clipping.set_age(T::from_f64(value as _));
            }
        }
        if let Some(b_active) = bypass {
            self.inner.inner.0.should_switch_to_b(b_active);
        }
    }
}

impl<T: Scalar<Element: Float>> Dsp<T>
where
    Complex<T>: SimdComplexField,
    <T as SimdValue>::Element: ToPrimitive,
{
    pub fn new(base_samplerate: f32, target_samplerate: f32) -> RemoteControlled<Self> {
        let oversample = target_samplerate / base_samplerate;
        let oversample = oversample.ceil() as usize;
        nih_log!("Requested oversample: {oversample}x");
        let oversample = Oversample::new(oversample, MAX_BLOCK_SIZE);
        let samplerate = base_samplerate * oversample.oversampling_amount() as f32;
        nih_log!(
            "Inner samplerate: {samplerate} Hz\n\t\tEffective oversample: {}",
            oversample.oversampling_amount()
        );

        let inner = Series((
            InputStage::new(samplerate, 1.0),
            ClippingStage::new(T::from_f64(samplerate as _)),
            ToneStage::new(samplerate, T::zero()),
            OutputStage::new(samplerate),
        ));
        let inner = DspInner::new(samplerate, Bypass::default(), inner, true);
        let inner = oversample.with_dsp(base_samplerate, BlockAdapter(inner));
        RemoteControlled::new(
            base_samplerate,
            1e3,
            Dsp {
                inner,
            },
        )
    }

    pub fn get_led_display(&self) -> Arc<AtomicF32> {
        let SwitchAB {
            b: Series((_, clipping, ..)),
            ..
        } = &self.inner.inner.0;
        clipping.led_display.clone()
    }

    pub fn set_led_display(&mut self, value: &Arc<AtomicF32>) {
        let SwitchAB {
            b: Series((_, clipping, ..)),
            ..
        } = &mut self.inner.inner.0;
        clipping.led_display = value.clone();
    }
}

pub const MAX_BLOCK_SIZE: usize = 64;
