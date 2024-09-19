#![warn(missing_docs)]
//! # Saturators
//!
//! This crate provides abstractions over saturators, as well as several standard saturator
//! functions.
//!
//! Saturators are set up so that their processing and their updating are separate; this allows
//! setting up iterative methods to improve accuracy. They should also provide a differentiation
//! function for more complex iteration schemes and feedback processing.
use num_traits::One;
use numeric_literals::replace_float_literals;
use std::ops;

use clippers::DiodeClipperModel;

use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::math::fast;
use valib_core::Scalar;

pub mod adaa;
pub mod bjt;
pub mod clippers;

/// Trait for types which are saturators.
///
/// Saturators are single-sample processors can have state, however the state must be updated after
/// the fact. This allows evaluating saturators with multistep methods or iterative schemes, i.e. to
/// resolve instantaneous feedback.
#[allow(unused_variables)]
pub trait Saturator<T: Scalar> {
    /// Saturate an input with a frozen state.
    fn saturate(&self, x: T) -> T;

    /// Update the state given an input and the output of [`Self::saturate`] for that input.
    #[inline(always)]
    fn update_state(&mut self, x: T, y: T) {}

    /// Differentiate the saturator at the given input.
    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn sat_diff(&self, x: T) -> T {
        (self.saturate(x + 1e-4) - self.saturate(x)) / 1e-4
    }
}

/// Trait for types which are multi-saturators.
///
/// Multi-saturators are a generalization of saturators which are N-dimensional.
pub trait MultiSaturator<T: Scalar, const N: usize> {
    /// Saturate the inputs with a frozen state.
    fn multi_saturate(&self, x: [T; N]) -> [T; N];

    /// Update the state given an input and the output of [`Self::saturate`] for that input.
    fn update_state_multi(&mut self, x: [T; N], y: [T; N]);

    /// Differentiate the saturator at the given input.
    fn sat_jacobian(&self, x: [T; N]) -> [T; N];
}

impl<'a, T: Scalar, S: Saturator<T>> MultiSaturator<T, 1> for &'a mut S {
    fn multi_saturate(&self, x: [T; 1]) -> [T; 1] {
        [self.saturate(x[0])]
    }

    fn update_state_multi(&mut self, x: [T; 1], y: [T; 1]) {
        self.update_state(x[0], y[0]);
    }

    fn sat_jacobian(&self, x: [T; 1]) -> [T; 1] {
        [self.sat_diff(x[0])]
    }
}

macro_rules! impl_multisat_tuples {
    ($count:literal; $($t:ident),*) => { ::paste::paste! {
        #[allow(non_snake_case)]
        impl<T: $crate::Scalar, $($t: $crate::Saturator<T>),*> MultiSaturator<T, $count> for ($($t,)*) {
            fn multi_saturate(&self, [$([<x $t>]),*]: [T; $count]) -> [T; $count] {
                let ($($t),*) = self;
                [$($t.saturate([<x $t>])),*]
            }

            fn update_state_multi(&mut self, [$([<x $t>]),*]: [T; $count], [$([<y $t>]),*]: [T; $count]) {
                let ($($t),*) = self;
                $(
                $t.update_state([<x $t>], [<y $t>]);
                )*
            }

            fn sat_jacobian(&self, [$([<x $t>]),*]: [T; $count]) -> [T; $count] {
                let ($($t),*) = self;
                [$($t.sat_diff([<x $t>])),*]
            }
        }
    } };
}

impl_multisat_tuples!(2; A, B);
impl_multisat_tuples!(3; A, B, C);
impl_multisat_tuples!(4; A, B, C, D);
impl_multisat_tuples!(5; A, B, C, D, E);
impl_multisat_tuples!(6; A, B, C, D, E, F);
impl_multisat_tuples!(7; A, B, C, D, E, F, G);
impl_multisat_tuples!(8; A, B, C, D, E, F, G, H);
impl_multisat_tuples!(9; A, B, C, D, E, F, G, H, I);
impl_multisat_tuples!(10; A, B, C, D, E, F, G, H, I, J);
impl_multisat_tuples!(11; A, B, C, D, E, F, G, H, I, J, K);
impl_multisat_tuples!(12; A, B, C, D, E, F, G, H, I, J, K, L);

/// Linear "saturator", a noop saturator which can be used when wanting no saturation.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Linear;

impl<S: Scalar> Saturator<S> for Linear {
    #[inline(always)]
    fn saturate(&self, x: S) -> S {
        x
    }

    #[inline(always)]
    fn sat_diff(&self, _: S) -> S {
        S::one()
    }
}

#[profiling::all_functions]
impl<S: Scalar, const N: usize> MultiSaturator<S, N> for Linear {
    #[inline(always)]
    fn multi_saturate(&self, x: [S; N]) -> [S; N] {
        x
    }

    #[inline(always)]
    fn update_state_multi(&mut self, _x: [S; N], _y: [S; N]) {}

    #[inline(always)]
    fn sat_jacobian(&self, _x: [S; N]) -> [S; N] {
        [S::one(); N]
    }
}

/// The `tanh` function as a saturator.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Tanh;

#[profiling::all_functions]
impl<S: Scalar> Saturator<S> for Tanh {
    #[inline(always)]
    fn saturate(&self, x: S) -> S {
        fast::tanh(x)
    }

    #[inline(always)]
    #[replace_float_literals(S::from_f64(literal))]
    fn sat_diff(&self, x: S) -> S {
        let tanh = fast::tanh(x);
        1. - tanh * tanh
    }
}

/// The `asinh` function as a saturator.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Asinh;

#[profiling::all_functions]
impl<T: Scalar> Saturator<T> for Asinh {
    fn saturate(&self, x: T) -> T {
        x.simd_asinh()
    }

    fn sat_diff(&self, x: T) -> T {
        let x0 = x * x + T::one();
        x0.simd_sqrt().simd_recip()
    }
}

/// Hard-clipper saturator, keeping the output within the provided bounds.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Clipper<T> {
    /// Minimum bound
    pub min: T,
    /// Maximum bound
    pub max: T,
}

impl<T: Copy + One + ops::Neg<Output = T>> Default for Clipper<T> {
    fn default() -> Self {
        let max = T::one();
        let min = -max;
        Self { min, max }
    }
}

#[profiling::all_functions]
impl<T: Scalar> Saturator<T> for Clipper<T> {
    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn saturate(&self, x: T) -> T {
        x.simd_min(self.max).simd_max(self.min)
    }

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn sat_diff(&self, x: T) -> T {
        let mask = x.simd_abs().simd_gt(1.0);
        (1.0).select(mask, 0.0)
    }
}

#[profiling::all_functions]
impl<T: Scalar, const N: usize> MultiSaturator<T, N> for Clipper<T> {
    fn multi_saturate(&self, x: [T; N]) -> [T; N] {
        x.map(|x| self.saturate(x))
    }

    fn update_state_multi(&mut self, _x: [T; N], _y: [T; N]) {}

    fn sat_jacobian(&self, x: [T; N]) -> [T; N] {
        x.map(|x| self.sat_diff(x))
    }
}

/// Blend the output of a saturator with its input by the given amount.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blend<T, S> {
    /// Amount of blending of the input to add to the output. The output will be scaled down to keep
    pub amt: T,
    inner: S,
}

#[profiling::all_functions]
impl<T: Scalar, S: Saturator<T>> Saturator<T> for Blend<T, S> {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        x + self.amt * (self.inner.saturate(x) - x)
    }

    #[inline(always)]
    fn update_state(&mut self, x: T, y: T) {
        self.inner.update_state(x, y)
    }

    #[inline(always)]
    fn sat_diff(&self, x: T) -> T {
        T::one() + self.amt * (self.inner.sat_diff(x) - T::one())
    }
}

impl<T: Scalar, S: Default> Default for Blend<T, S> {
    fn default() -> Self {
        Self {
            amt: T::from_f64(0.5),
            inner: S::default(),
        }
    }
}

/// Runtime-switchable dynamic saturator
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Dynamic<T> {
    /// Linear "saturator". No saturation.
    Linear,
    /// `tanh` function
    Tanh,
    /// `asinh` function
    Asinh,
    /// Hard clipping between -1 and 1
    HardClipper,
    /// Diode clipper model
    DiodeClipper(DiodeClipperModel<T>),
    /// "Overdrive" clipper model
    SoftClipper(Blend<T, DiodeClipperModel<T>>),
}

#[profiling::all_functions]
impl<T: Scalar> Saturator<T> for Dynamic<T> {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        match self {
            Self::Linear => Linear.saturate(x),
            Self::HardClipper => Clipper::default().saturate(x),
            Self::Tanh => Tanh.saturate(x),
            Self::Asinh => Asinh.saturate(x),
            Self::DiodeClipper(clip) => clip.saturate(x),
            Self::SoftClipper(clip) => clip.saturate(x),
        }
    }

    #[inline(always)]
    fn sat_diff(&self, x: T) -> T {
        match self {
            Self::Linear => Linear.sat_diff(x),
            Self::HardClipper => Clipper::default().sat_diff(x),
            Self::Asinh => Asinh.sat_diff(x),
            Self::Tanh => Tanh.sat_diff(x),
            Self::DiodeClipper(clip) => clip.sat_diff(x),
            Self::SoftClipper(clip) => clip.sat_diff(x),
        }
    }
}

impl<T> Default for Dynamic<T> {
    fn default() -> Self {
        Self::Linear
    }
}

/// Slew rate saturator. Only allows the signal to change up to a maximum speed.
#[derive(Debug, Clone, Copy)]
pub struct Slew<T> {
    /// Maximum difference between two consecutive samples.
    pub max_diff: T,
    last_out: T,
}

impl<T: Scalar> Slew<T> {
    /// Returns true if the signal is being rate-limited by the slew.
    ///
    /// # Arguments
    ///
    /// `target`: Target value
    pub fn is_changing(&self, target: T) -> T::SimdBool {
        (target - self.last_out)
            .simd_abs()
            .simd_gt(T::from_f64(1e-6))
    }

    /// Returns the last output.
    pub fn current_value(&self) -> T {
        self.last_out
    }
}

impl<T: Scalar> Default for Slew<T> {
    fn default() -> Self {
        Self {
            max_diff: T::from_f64(1.0),
            last_out: T::from_f64(0.0),
        }
    }
}

impl<T: Scalar> DSPMeta for Slew<T> {
    type Sample = T;

    fn latency(&self) -> usize {
        0
    }

    fn reset(&mut self) {
        self.last_out = T::from_f64(0.0);
    }
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<1, 1> for Slew<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = self.slew(x[0]);
        self.last_out = y;
        [y]
    }
}

impl<T: Scalar> Slew<T> {
    /// Create a new slew rate limiter.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Sample rate the limiter will be running at
    /// * `max_diff`: Maximum speed in units/s
    ///
    /// returns: Slew<T>
    pub fn new(samplerate: T, max_diff: T) -> Self {
        Self {
            max_diff: max_diff / samplerate,
            last_out: T::from_f64(0.0),
        }
    }

    /// Sets the current state of the slew limiter.
    ///
    /// # Arguments
    ///
    /// * `state`: New state ("last output") of the slew limiter.
    ///
    /// returns: Slew<T>
    pub fn with_state(mut self, state: T) -> Self {
        self.last_out = state;
        self
    }

    /// Set the maximum difference within a second that the signal will be able to change.
    ///
    /// # Arguments
    ///
    /// * `max`: Maximum difference (units/s)
    /// * `samplerate`:  Sample rate of the slew limiter
    ///
    /// returns: ()
    pub fn set_max_diff(&mut self, max: T, samplerate: T) {
        self.max_diff = max / samplerate;
    }

    fn slew_diff(&self, x: T) -> T {
        let diff = x - self.last_out;
        diff.simd_clamp(-self.max_diff, self.max_diff)
    }

    fn slew(&self, x: T) -> T {
        self.last_out + self.slew_diff(x)
    }
}

#[profiling::all_functions]
impl<T: Scalar> Saturator<T> for Slew<T> {
    fn saturate(&self, x: T) -> T {
        self.slew(x)
    }

    fn update_state(&mut self, _x: T, y: T) {
        self.last_out = y;
    }

    fn sat_diff(&self, x: T) -> T {
        self.slew_diff(x)
    }
}

/// Boost the input to the saturator, then reduce the saturator output by the same amount.
///
/// Also biases the inputs and corrects at the output.
#[derive(Debug, Clone, Copy)]
pub struct Driven<T, S> {
    /// Drive amount
    pub drive: T,
    /// Bias amount
    pub bias: T,
    /// Inner saturator
    pub saturator: S,
}

#[profiling::all_functions]
impl<T: Scalar, S: Saturator<T>> Saturator<T> for Driven<T, S> {
    fn saturate(&self, x: T) -> T {
        self.saturator.saturate(x * self.drive) / self.drive
    }

    #[inline(always)]
    fn update_state(&mut self, x: T, y: T) {
        let x = x * self.drive;
        let y = self.drive / y;
        self.saturator.update_state(x, y);
    }

    fn sat_diff(&self, x: T) -> T {
        self.saturator.sat_diff(x * self.drive)
    }
}
