//! Shared values for passing parameters into DSP code.
//!
//! By design, those parameters are not used by the implementations in this
//! library. This is because it would be too limiting and cumbersome to go from
//! single-valued parameter values to potentially multi-valied values (eg. a
//! filter cutoff of a stereo filter, where the stereo signal is represented as
//! a single f32x2 SIMD type). They are provided as is for bigger modules, and
//! the traits implemented by "container" modules (like [`Series`] or
//! [`Parallel`]) to ease their use in propagating parameters.
//!
//! [`Series`]: crate::dsp::blocks::Series
//! [`Parallel`]: crate::dsp::blocks::Parallel
use core::fmt;
use std::borrow::Cow;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::{self, Deref};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use nalgebra::SMatrix;
use portable_atomic::{AtomicBool, AtomicF32};

use crate::dsp::blocks::{ModMatrix, Series2, P1};
use crate::dsp::{DSPMeta, DSPProcess};
use crate::saturators::Slew;

pub use valib_derive::ParamName;

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

    pub fn get_discrete(&self, num_values: usize) -> usize {
        (self.get_value() * num_values as f32).round() as _
    }

    pub fn set_discrete(&self, num_values: usize, value: usize) {
        self.set_value(value as f32 / num_values as f32);
    }

    /* TODO: Figure out how to make this work

    pub fn get_enum<E: Enum>(&self) -> E {
        let index = self.get_value().min(E::LENGTH as f32 - 1.0);
        E::from_usize(index.floor() as _)
    }

    pub fn set_enum<E: Enum>(&self, value: E) {
        let step = f32::recip(E::LENGTH as f32);
        self.set_value(value.into_usize() as f32 * step);
    }
    */

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

impl<P: DSPProcess<1, 1, Sample = f32>> DSPMeta for FilteredParam<P> {
    type Sample = f32;
}

impl<P: DSPProcess<1, 1, Sample = f32>> FilteredParam<P> {
    /// Process the next sample generated from the parameter value.
    pub fn next_sample(&mut self) -> f32 {
        self.process([])[0]
    }
}

impl<P: DSPProcess<1, 1, Sample = f32>> DSPProcess<0, 1> for FilteredParam<P> {
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

impl DSPMeta for Smoothing {
    type Sample = f32;
}

impl DSPProcess<1, 1> for Smoothing {
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

impl DSPMeta for SmoothedParam {
    type Sample = f32;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.smoothing.set_samplerate(samplerate);
    }
}

impl DSPProcess<0, 1> for SmoothedParam {
    #[inline]
    fn process(&mut self, _x: [Self::Sample; 0]) -> [Self::Sample; 1] {
        self.smoothing.process([self.param.get_value()])
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
        let initial_state = param.get_value();
        Self {
            param,
            smoothing: Smoothing::Linear {
                slew: Slew::new(samplerate, max_diff).with_state(initial_state),
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
        let initial_value = param.get_value();
        Self {
            param,
            smoothing: Smoothing::Exponential(Series2::new(
                P1::new(samplerate, tau.recip()).with_state(initial_value),
                ModMatrix {
                    weights: SMatrix::<_, 1, 3>::new(1.0, 0.0, 0.0),
                },
            )),
        }
    }

    pub fn next_sample(&mut self) -> f32 {
        self.process([])[0]
    }
}

/// Parameter ID alias. Useful for type-erasing parameter names and make communication easier, but
/// this risks unwanted transmutations if not handled properly.
///
/// Note that transmuting between param ids is safe because both sides allow converting to and from
/// a [`ParamId`].
pub type ParamId = u64;

/// Trait for types that are parameter names.
///
/// This trait is most easily implemented as an enum of all possible parameters, but it allows
/// constructs such as `[P; N]` where `P: ParamName` and `const N: usize` to be defined for
/// automatic duplication of parameters, or 1->N communication of parameter values.
pub trait ParamName: Sized {
    /// Total number of elements in this type
    fn count() -> usize;

    /// Construct a [`Self`] from a [`ParamId`] value. The caller is expected to verify `value <
    /// Self::count()`, and so this method is declared as infallible.
    fn from_id(value: ParamId) -> Self;

    /// Construct a [`ParamId`] from this [`Self`].
    ///
    /// It is expected that round-trip conversion returns the same enum as the one we started with,
    /// that is:
    ///
    /// ```rust,compile_fail
    /// assert_eq!(self, Self::from_id(self.into_id()));
    /// ```
    ///
    /// where `Self: PartialEq`.
    fn into_id(self) -> ParamId;

    /// Return a user-friendly name for this parameter name.
    fn name(&self) -> Cow<'static, str>;

    /// Create an iterator returning all values for this type, that is, all values converted from
    /// IDs in sequence in the range `0..Self::count()`.
    fn iter() -> impl Iterator<Item = Self> {
        (0..Self::count()).map(|i| Self::from_id(i as _))
    }
}

/// Trait of types which have modulatable parameters.
pub trait HasParameters {
    /// Parameter name type
    type Enum: Copy + ParamName;

    /// Gets the matching [`Parameter`] for the parameter name
    fn get_parameter(&self, param: Self::Enum) -> &Parameter;

    /// Overridable shortcut for `self.get_parameter(param).set_value(value)`, which can be used to
    /// react to changes in an event-driven way.
    ///
    /// Consumers of this trait should use this method directly to allow behavior customization,
    /// instead of getting the parameter and setting its value directly.
    fn set_parameter(&self, param: Self::Enum, value: f32) {
        self.get_parameter(param).set_value(value);
    }

    /// Get the name of this parameter, that is either the name set on the [`Parameter`], or the
    /// default name of the [`Self::Enum`] type.
    fn param_name(&self, param: Self::Enum) -> Cow<'static, str> {
        self.get_parameter(param)
            .name()
            .map(|s| Cow::Owned(s.to_string()))
            .unwrap_or_else(|| param.name())
    }

    /// Iterate through all parameters in this type.
    fn iter_parameters(&self) -> impl Iterator<Item = (Self::Enum, &Parameter)> {
        Self::Enum::iter().map(|p| (p, self.get_parameter(p)))
    }
}

/// Specialized map type for storing values associated to parameters.
#[derive(Debug, Clone)]
pub struct ParamMap<P, T> {
    data: Vec<T>,
    __param: PhantomData<P>,
}

impl<P: ParamName, T: Default> Default for ParamMap<P, T> {
    fn default() -> Self {
        Self::new(|_| T::default())
    }
}

pub struct ParamMapIntoIter<P, T> {
    data: std::vec::IntoIter<T>,
    item: u64,
    __param: PhantomData<P>,
}

impl<P: ParamName, T> Iterator for ParamMapIntoIter<P, T> {
    type Item = (P, T);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.data.next()?;
        let i = self.item;
        self.item += 1;
        Some((P::from_id(i as _), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = P::count() as usize;
        (len, Some(len))
    }
}

impl<P: ParamName, T> ExactSizeIterator for ParamMapIntoIter<P, T> {
    fn len(&self) -> usize {
        P::count() as _
    }
}

impl<P: ParamName, T> IntoIterator for ParamMap<P, T> {
    type IntoIter = ParamMapIntoIter<P, T>;
    type Item = (P, T);

    fn into_iter(self) -> Self::IntoIter {
        ParamMapIntoIter {
            data: self.data.into_iter(),
            item: 0,
            __param: PhantomData,
        }
    }
}

impl<P: ParamName, T> ops::Index<P> for ParamMap<P, T> {
    type Output = T;

    fn index(&self, index: P) -> &Self::Output {
        &self.data[index.into_id() as usize]
    }
}

impl<P: ParamName + Clone, T> ops::Index<&P> for ParamMap<P, T> {
    type Output = T;

    fn index(&self, index: &P) -> &Self::Output {
        &self.data[index.clone().into_id() as usize]
    }
}

impl<P: ParamName, T> ops::IndexMut<P> for ParamMap<P, T> {
    fn index_mut(&mut self, index: P) -> &mut Self::Output {
        &mut self.data[index.into_id() as usize]
    }
}

impl<P: ParamName + Clone, T> ops::IndexMut<&P> for ParamMap<P, T> {
    fn index_mut(&mut self, index: &P) -> &mut Self::Output {
        &mut self.data[index.clone().into_id() as usize]
    }
}

impl<P: ParamName, T> ParamMap<P, T> {
    pub fn new(fill_fn: impl FnMut(P) -> T) -> Self {
        Self {
            data: Vec::from_iter(P::iter().map(fill_fn)),
            __param: PhantomData,
        }
    }

    pub fn iter(&self) -> impl '_ + Iterator<Item = (P, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, x)| (P::from_id(i as _), x))
    }

    pub fn iter_mut(&mut self) -> impl '_ + Iterator<Item = (P, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, x)| (P::from_id(i as _), x))
    }
}

/// Separate controller for controlling parameters of a [`DSPProcess`] instance from another place.
#[derive(Debug, Clone)]
pub struct ParamController<P: HasParameters>(ParamMap<P::Enum, Parameter>);

impl<P: HasParameters> From<&P> for ParamController<P> {
    fn from(value: &P) -> Self {
        Self(ParamMap::new(|p| value.get_parameter(p).clone()))
    }
}

impl<P: HasParameters> Deref for ParamController<P> {
    type Target = ParamMap<P::Enum, Parameter>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
