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
use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use portable_atomic::{AtomicBool, AtomicF32};

pub use valib_derive::ParamName;

use crate::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use crate::dsp::{DSPMeta, DSPProcess, DSPProcessBlock};
use crate::Scalar;

/// Filtered parameter value, useful with any DSP<1, 1, Sample=f32> algorithm.
pub struct FilteredParam<P> {
    /// Raw parameter value. This can be set directly to update the parameter.
    pub param: f32,
    /// Process which will take in the raw value and filter it.
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
        self.dsp.process([self.param])
    }
}

#[derive(Debug, Copy, Clone)]
enum Smoothing {
    Exponential {
        state: f32,
        fc: f32,
        lambda: f32,
    },
    Linear {
        samplerate: f32,
        last_out: f32,
        max_per_sec: f32,
    },
}

impl Smoothing {
    fn set_samplerate(&mut self, new_sr: f32) {
        match self {
            Self::Exponential { fc, lambda, .. } => {
                *lambda = *fc / new_sr;
            }
            Self::Linear { samplerate, .. } => {
                *samplerate = new_sr;
            }
        }
    }

    fn is_changing(&self, value: f32) -> bool {
        match self {
            Self::Exponential { state, .. } => (value - state).abs() < 1e-6,
            Self::Linear { last_out, .. } => (value - last_out).abs() < 1e-6,
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
            Self::Exponential { state, lambda, .. } => {
                *state += (x[0] - *state) * *lambda;
                [*state]
            }
            Self::Linear {
                samplerate,
                last_out,
                max_per_sec,
            } => {
                let max_diff = *max_per_sec / *samplerate;
                let diff = x[0] - *last_out;
                *last_out += diff.clamp(-max_diff, max_diff);
                [*last_out]
            }
        }
    }
}

/// Smoothed parameter. Smoothing can be applied exponentially or linearly.
#[derive(Debug, Copy, Clone)]
pub struct SmoothedParam {
    /// Raw parameter value; can be set directly to change the target of the smoothed parameter.
    pub param: f32,
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
        self.smoothing.process([self.param])
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
    pub fn linear(initial_value: f32, samplerate: f32, duration_ms: f32) -> Self {
        Self {
            param: initial_value,
            smoothing: Smoothing::Linear {
                samplerate,
                max_per_sec: duration_ms.recip(),
                last_out: initial_value,
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
    pub fn exponential(initial_value: f32, samplerate: f32, t60_ms: f32) -> Self {
        let tau = 6.91 / t60_ms * 1e3;
        Self {
            param: initial_value,
            smoothing: Smoothing::Exponential {
                state: initial_value,
                fc: tau,
                lambda: tau / samplerate,
            },
        }
    }

    /// Returns the current smoothed value of the parameter.
    pub fn current_value(&self) -> f32 {
        match self.smoothing {
            Smoothing::Linear { last_out, .. } => last_out,
            Smoothing::Exponential { state, .. } => state,
        }
    }

    /// Computes the next sample of the smoother.
    pub fn next_sample(&mut self) -> f32 {
        self.process([])[0]
    }

    /// Computes the next sample of the smoother, casting it into a `T`.
    pub fn next_sample_as<T: Scalar>(&mut self) -> T {
        T::from_f64(self.next_sample() as _)
    }

    /// Returns true when the smoother is still in the process of smoothing the change to the raw value.
    pub fn is_changing(&self) -> bool {
        self.smoothing.is_changing(self.param)
    }
}

/// Parameter ID alias. Useful for type-erasing parameter names and make communication easier, but
/// this risks unwanted transmutations if not handled properly.
///
/// Note that transmuting between param ids is safe because both sides allow converting to and from
/// a [`ParamId`].
pub type ParamId = usize;

/// Trait for types that are parameter names.
///
/// This trait is most easily implemented as an enum of all possible parameters, but it allows
/// constructs such as `[P; N]` where `P: ParamName` and `const N: usize` to be defined for
/// automatic duplication of parameters, or 1->N communication of parameter values.
pub trait ParamName: Copy {
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
    type Name: Copy + ParamName;

    /// Set a new value for the parameter at the given parameter name.
    fn set_parameter(&mut self, param: Self::Name, value: f32);
}

/// Extension trait for types which have parameters.
pub trait HasParametersExt: HasParameters {
    /// Set the parameter as a boolean value. It will be encoded such that `value > 0.5` decodes
    /// back to the input boolean value.
    ///
    /// # Arguments
    ///
    /// * `param`: Name of the parameter to change
    /// * `value`: Boolean value to set
    ///
    /// returns: ()
    fn set_parameter_bool(&mut self, param: Self::Name, value: bool) {
        self.set_parameter(param, if value { 1.0 } else { 0.0 });
    }
}

impl<'a, P: HasParameters> HasParameters for &'a mut P {
    type Name = P::Name;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        HasParameters::set_parameter(*self, param, value);
    }
}

impl<P: HasParameters> HasParameters for Box<P> {
    type Name = P::Name;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        P::set_parameter(&mut *self, param, value);
    }
}

/// Dynamic parameter type which advertises as having `N` possible names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dynamic<const N: ParamId>(ParamId);

impl<const N: ParamId> Dynamic<N> {
    /// Create a new `Dynamic<N>` parameter name, checking the passed in ID for validity.
    ///
    /// # Arguments
    ///
    /// * `value`: Raw ID to use for this parameter type. The parameter is valid when `value < N`.
    ///
    /// returns: Option<Dynamic<{ N }>>
    pub fn new(value: ParamId) -> Option<Self> {
        (value < N).then_some(Self(value))
    }
}

impl<const N: ParamId> ParamName for Dynamic<N> {
    fn count() -> usize {
        N
    }

    fn from_id(value: ParamId) -> Self {
        Self::new(value).unwrap()
    }

    fn into_id(self) -> ParamId {
        self.0
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.0.to_string())
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

/// Type which implements [`Iterator`] listing the parameters and their associated value.
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
        let len = P::count();
        (len, Some(len))
    }
}

impl<P: ParamName, T> ExactSizeIterator for ParamMapIntoIter<P, T> {
    fn len(&self) -> usize {
        P::count() as _
    }
}

impl<P: ParamName, T> IntoIterator for ParamMap<P, T> {
    type Item = (P, T);
    type IntoIter = ParamMapIntoIter<P, T>;

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
        &self.data[index.into_id()]
    }
}

impl<P: ParamName + Clone, T> ops::Index<&P> for ParamMap<P, T> {
    type Output = T;

    fn index(&self, index: &P) -> &Self::Output {
        &self.data[(*index).into_id()]
    }
}

impl<P: ParamName, T> ops::IndexMut<P> for ParamMap<P, T> {
    fn index_mut(&mut self, index: P) -> &mut Self::Output {
        &mut self.data[index.into_id()]
    }
}

impl<P: ParamName + Clone, T> ops::IndexMut<&P> for ParamMap<P, T> {
    fn index_mut(&mut self, index: &P) -> &mut Self::Output {
        &mut self.data[(*index).into_id()]
    }
}

impl<P: ParamName, T> ParamMap<P, T> {
    /// Create a new parameter map, filled in by the provided closure.
    ///
    /// # Arguments
    ///
    /// * `fill_fn`: Closure which is called for each parameter, and returns the associated value.
    ///
    /// returns: ParamMap<P, T>
    pub fn new(fill_fn: impl FnMut(P) -> T) -> Self {
        Self {
            data: Vec::from_iter(P::iter().map(fill_fn)),
            __param: PhantomData,
        }
    }

    /// Iterate over parameters and references to their values.
    pub fn iter(&self) -> impl '_ + Iterator<Item = (P, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, x)| (P::from_id(i as _), x))
    }

    /// Iterate over parameters and mutable references to their values.
    pub fn iter_mut(&mut self) -> impl '_ + Iterator<Item = (P, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, x)| (P::from_id(i as _), x))
    }
}

/// Object-safe trait for types with parameters.
pub trait HasParametersErased {
    /// Set a parameter value by its raw param ID.
    ///
    /// # Arguments
    ///
    /// * `param_id`: Raw param ID to set.
    /// * `value`: Value to set the param ID with.
    ///
    /// returns: ()
    fn set_parameter_raw(&mut self, param_id: ParamId, value: f32);
}

/// Proxy parameter updates to another type. This allows thread-safe control of processors via their
/// parameters.
pub struct ParamsProxy<P: ParamName> {
    params: ParamMap<P, Arc<AtomicF32>>,
    param_changed: ParamMap<P, Arc<AtomicBool>>,
}

/// Type alias for the type that allows remote control of processors via their parameters.
pub type RemoteControl<P> = Arc<ParamsProxy<P>>;

impl<P: ParamName> ParamsProxy<P> {
    /// Create a new param proxy.
    pub fn new() -> Arc<Self> {
        let params = ParamMap::new(|_| Arc::new(AtomicF32::new(0.0)));
        let param_changed = ParamMap::new(|_| Arc::new(AtomicBool::new(false)));
        Arc::new(Self {
            params,
            param_changed,
        })
    }

    /// Set a parameter for a remote type.
    ///
    /// # Arguments
    ///
    /// * `param`: Parameter to set
    /// * `value`: Value to set
    ///
    /// returns: ()
    pub fn set_parameter(&self, param: P, value: f32) {
        self.param_changed[param].store(true, Ordering::SeqCst);
        self.params[param].store(value, Ordering::SeqCst);
    }

    fn get_update(&self, param: P) -> Option<f32> {
        let has_changed = self.param_changed[param]
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or(false);
        if has_changed {
            return Some(self.params[param].load(Ordering::SeqCst));
        }
        None
    }
}

/// Type which remote controls the type `P` through its [`RemoteControlled::proxy`].
pub struct RemoteControlled<P: HasParameters> {
    /// Remote-controlled type
    pub inner: P,
    /// Remote control proxy, which you can clone and send to another thread.
    pub proxy: RemoteControl<P::Name>,
    update_params_phase: f32,
    update_params_step: f32,
}

impl<P: HasParameters + DSPMeta> DSPMeta for RemoteControlled<P> {
    type Sample = P::Sample;
}

impl<P: HasParameters + DSPProcess<I, O>, const I: usize, const O: usize> DSPProcess<I, O>
    for RemoteControlled<P>
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        self.update_params_phase += self.update_params_step;
        if self.update_params_phase > 1.0 {
            self.update_params_phase -= 1.0;
            self.update_parameters();
        }

        self.inner.process(x)
    }
}

#[profiling::all_functions]
impl<P: HasParameters + DSPProcessBlock<I, O>, const I: usize, const O: usize> DSPProcessBlock<I, O>
    for RemoteControlled<P>
{
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, I>,
        outputs: AudioBufferMut<Self::Sample, O>,
    ) {
        self.update_params_phase += self.update_params_step * inputs.samples() as f32;
        if self.update_params_phase > 1.0 {
            self.update_parameters();
            self.update_params_phase = self.update_params_phase.fract();
        }
        self.inner.process_block(inputs, outputs);
    }

    fn max_block_size(&self) -> Option<usize> {
        self.inner.max_block_size()
    }
}

impl<P: HasParameters> RemoteControlled<P> {
    /// Create a new remote, controlling the passed in processor.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Sample rate at which the processor and remote control will run
    /// * `update_frequency`: Frequency (in Hz) at which the remote control will check for updated
    ///     parameters, and transfer them to the inner processor.
    /// * `inner`: Inner processor, that is going to be controlled by this.
    ///
    /// returns: RemoteControlled<P>
    pub fn new(samplerate: f32, update_frequency: f32, inner: P) -> Self {
        Self {
            inner,
            proxy: ParamsProxy::new(),
            update_params_phase: 0.0,
            update_params_step: update_frequency * samplerate.recip(),
        }
    }
}

#[profiling::all_functions]
impl<P: HasParameters> RemoteControlled<P> {
    /// Check for update on all parameters, and transmit them to the inner processor if they have
    /// changed.
    pub fn update_parameters(&mut self) {
        for param in P::Name::iter() {
            if let Some(value) = self.proxy.get_update(param) {
                self.inner.set_parameter(param, value);
            }
        }
    }
}
