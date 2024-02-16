use core::fmt;
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use enum_map::{Enum, EnumArray, EnumMap};
use nalgebra::SMatrix;
use portable_atomic::{AtomicBool, AtomicF32};

use crate::dsp::blocks::{ModMatrix, Series2, P1};
use crate::dsp::DSP;
use crate::saturators::Slew;

struct ParamImpl {
    value: AtomicF32,
    name: Option<String>,
    changed: AtomicBool,
}

/// Shared atomic float value for providing parameters to DSP algorithms without having direct reference
/// to them.
#[derive(Clone)]
pub struct Parameter(Arc<ParamImpl>);

impl fmt::Debug for Parameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Parameter")
            .field(&self.0.name)
            .field(&self.0.value.load(Ordering::Relaxed))
            .finish()
    }
}

impl Parameter {
    pub fn new(value: f32) -> Self {
        Self(Arc::new(ParamImpl {
            value: AtomicF32::new(value),
            name: None,
            changed: AtomicBool::new(true),
        }))
    }

    pub fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    pub fn named(self, name: impl ToString) -> Self {
        Self(Arc::new(ParamImpl {
            value: AtomicF32::new(self.0.value.load(Ordering::SeqCst)),
            name: Some(name.to_string()),
            changed: AtomicBool::new(self.0.changed.load(Ordering::SeqCst)),
        }))
    }

    pub fn get_value(&self) -> f32 {
        self.0.value.load(Ordering::Relaxed)
    }

    pub fn set_value(&self, value: f32) {
        self.0.changed.store(true, Ordering::Release);
        self.0.value.store(value, Ordering::Relaxed);
    }

    pub fn has_changed(&self) -> bool {
        self.0.changed.swap(false, Ordering::AcqRel)
    }

    pub fn get_bool(&self) -> bool {
        self.get_value() > 0.5
    }

    pub fn set_bool(&self, value: bool) {
        self.set_value(if value { 1.0 } else { 0.0 })
    }

    pub fn get_enum<E: Enum>(&self) -> E {
        let index = self.get_value().min(E::LENGTH as f32 - 1.0);
        eprintln!("get_enum index {index}");
        return E::from_usize(index.floor() as _);
    }

    pub fn set_enum<E: Enum>(&self, value: E) {
        let step = f32::recip(E::LENGTH as f32);
        self.set_value(value.into_usize() as f32 * step);
    }

    pub fn filtered<P>(&self, dsp: P) -> FilteredParam<P> {
        FilteredParam {
            param: self.clone(),
            dsp,
        }
    }

    pub fn smoothed_linear(&self, samplerate: f32, max_dur_ms: f32) -> SmoothedParam {
        SmoothedParam::linear(self.clone(), samplerate, max_dur_ms)
    }

    pub fn smoothed_exponential(&self, samplerate: f32, t60_ms: f32) -> SmoothedParam {
        SmoothedParam::exponential(self.clone(), samplerate, t60_ms)
    }
}

/// Filtered parameter value, useful with any DSP<1, 1, Sample=f32> algorithm.
pub struct FilteredParam<P> {
    pub param: Parameter,
    pub dsp: P,
}

impl<P: DSP<1, 1, Sample = f32>> FilteredParam<P> {
    /// Process the next sample generated from the parameter value.
    pub fn next_sample(&mut self) -> f32 {
        self.process([])[0]
    }
}

impl<P: DSP<1, 1, Sample = f32>> DSP<0, 1> for FilteredParam<P> {
    type Sample = f32;

    fn process(&mut self, _x: [Self::Sample; 0]) -> [Self::Sample; 1] {
        self.dsp.process([self.param.get_value()])
    }
}

#[derive(Debug, Copy, Clone)]
enum Smoothing {
    Exponential(Series2<P1<f32>, ModMatrix<f32, 3, 1>, 3>),
    Linear { slew: Slew<f32>, max_diff: f32 },
}

impl Smoothing {
    fn set_samplerate(&mut self, new_sr: f32) {
        match self {
            Self::Exponential(s) => s.left_mut().set_samplerate(new_sr),
            Self::Linear { slew, max_diff } => {
                slew.set_max_diff(*max_diff, new_sr);
            }
        }
    }
}

impl DSP<1, 1> for Smoothing {
    type Sample = f32;

    #[inline]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        match self {
            Self::Exponential(s) => s.process(x),
            Self::Linear { slew: s, .. } => s.process(x),
        }
    }
}

/// Smoothed parameter. Smoothing can be applied exponentially or linearly.
#[derive(Debug, Clone)]
pub struct SmoothedParam {
    pub param: Parameter,
    smoothing: Smoothing,
}

impl DSP<0, 1> for SmoothedParam {
    type Sample = f32;

    #[inline]
    fn process(&mut self, _x: [Self::Sample; 0]) -> [Self::Sample; 1] {
        self.smoothing.process([self.param.get_value()])
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        self.smoothing.set_samplerate(samplerate);
    }
}

impl SmoothedParam {
    /// Create a new linear smoothed parameter, with the given parameter and the maximum duration in milliseconds.
    ///
    /// # Arguments
    ///
    /// * `param`: Inner parameter to tap values from.
    /// * `samplerate`: Samplerate at which the smoother will run.
    /// * `duration_ms`: Maximum duration of a sweep, that is the duration it would take to go from one extreme to the other.
    pub fn linear(param: Parameter, samplerate: f32, duration_ms: f32) -> Self {
        let max_diff = 1000.0 / duration_ms;
        Self {
            param,
            smoothing: Smoothing::Linear {
                slew: Slew::new(samplerate, max_diff),
                max_diff,
            },
        }
    }

    /// Create a new exponential smoothed parameter, with the given parameter and the T60 time constant in milliseconds.
    ///
    /// # Arguments
    ///
    /// * `param`: Inner parameter to tap values from
    /// * `samplerate`: Samplerate parameter
    /// * `t60_ms`: "Time to decay by 60 dB" -- the time it takes for the output to be within 0.1% of the target value.
    pub fn exponential(param: Parameter, samplerate: f32, t60_ms: f32) -> Self {
        let tau = 6.91 * t60_ms / 1e3;
        Self {
            param,
            smoothing: Smoothing::Exponential(Series2::new(
                P1::new(samplerate, tau.recip()),
                ModMatrix {
                    weights: SMatrix::<_, 1, 3>::new(1.0, 0.0, 0.0),
                    ..ModMatrix::default()
                },
            )),
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        self.process([])[0]
    }
}

pub trait HasParameters {
    type Enum: Copy + Enum;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter;

    fn param_name(&self, param: Self::Enum) -> Option<&str> {
        self.get_parameter(param).name()
    }

    fn iter_parameters(&self) -> impl Iterator<Item = (Self::Enum, &Parameter)> {
        (0..Self::Enum::LENGTH)
            .map(Self::Enum::from_usize)
            .map(|p| (p, self.get_parameter(p)))
    }

    fn full_name(&self, param: Self::Enum) -> String {
        self.get_parameter(param)
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Param {}", param.into_usize() + 1))
    }
}

/// Separate controller for controlling parameters of a [`DSP`] instance from another place.
pub struct ParamController<P: HasParameters>(EnumMap<P::Enum, Parameter>)
where
    P::Enum: EnumArray<Parameter>;

impl<P: HasParameters> From<&P> for ParamController<P>
where
    P::Enum: EnumArray<Parameter>,
{
    fn from(value: &P) -> Self {
        Self(EnumMap::from_fn(|p| value.get_parameter(p).clone()))
    }
}

impl<P: HasParameters> Deref for ParamController<P>
where
    P::Enum: EnumArray<Parameter>,
{
    type Target = EnumMap<P::Enum, Parameter>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
