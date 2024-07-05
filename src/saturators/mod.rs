use nalgebra::{SMatrix, SVector};
use num_traits::One;
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;
use std::ops;

use clippers::DiodeClipperModel;

use crate::dsp::{DSPMeta, DSPProcess};
use crate::math::{newton_rhapson_steps, smooth_clamp, RootEq};
use crate::Scalar;

pub mod adaa;
pub mod clippers;

#[allow(unused_variables)]
pub trait Saturator<T: Scalar> {
    /// Saturate an input with a frozen state.
    fn saturate(&self, x: T) -> T;

    /// Update the state given an input and the output of [`Self::saturate`].
    #[inline(always)]
    fn update_state(&mut self, x: T, y: T) {}

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn sat_diff(&self, x: T) -> T {
        (self.saturate(x + 1e-4) - self.saturate(x)) / 1e-4
    }
}

pub trait MultiSaturator<T: Scalar, const N: usize> {
    fn multi_saturate(&self, x: [T; N]) -> [T; N];

    fn update_state_multi(&mut self, x: [T; N], y: [T; N]);

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
        impl<T: $crate::Scalar, $($t: $crate::saturators::Saturator<T>),*> MultiSaturator<T, $count> for ($($t,)*) {
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
    fn multi_saturate(&self, x: [S; N]) -> [S; N] {
        x
    }

    fn update_state_multi(&mut self, x: [S; N], y: [S; N]) {}

    fn sat_jacobian(&self, x: [S; N]) -> [S; N] {
        [S::one(); N]
    }
}

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Tanh;

#[profiling::all_functions]
impl<S: Scalar> Saturator<S> for Tanh {
    #[inline(always)]
    fn saturate(&self, x: S) -> S {
        x.simd_tanh()
    }

    #[inline(always)]
    #[replace_float_literals(S::from_f64(literal))]
    fn sat_diff(&self, x: S) -> S {
        1. - x.simd_tanh().simd_powi(2)
    }
}

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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Clipper<T> {
    pub min: T,
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

    fn update_state_multi(&mut self, x: [T; N], y: [T; N]) {}

    fn sat_jacobian(&self, x: [T; N]) -> [T; N] {
        x.map(|x| self.sat_diff(x))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blend<T, S> {
    amt: T,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Dynamic<T> {
    Linear,
    Tanh,
    Asinh,
    HardClipper,
    DiodeClipper(DiodeClipperModel<T>),
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

#[derive(Debug, Clone, Copy)]
pub struct Slew<T> {
    pub max_diff: T,
    last_out: T,
}

impl<T: Scalar> Slew<T> {
    pub fn is_changing(&self, target: T) -> T::SimdBool {
        (target - self.last_out)
            .simd_abs()
            .simd_gt(T::from_f64(1e-6))
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
    pub fn new(samplerate: T, max_diff: T) -> Self {
        Self {
            max_diff: max_diff / samplerate,
            last_out: T::from_f64(0.0),
        }
    }

    pub fn with_state(mut self, state: T) -> Self {
        self.last_out = state;
        self
    }

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

#[derive(Debug, Copy, Clone)]
pub struct Bjt<T> {
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

#[profiling::all_functions]
impl<T: Scalar> Saturator<T> for Bjt<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn saturate(&self, x: T) -> T {
        smooth_clamp(0.1, x - 0.770, self.vee, self.vcc) + 0.770
    }
}

impl<T: Scalar> DSPMeta for Bjt<T> {
    type Sample = T;
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<1, 1> for Bjt<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.saturate(x)]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Driven<T, S> {
    pub drive: T,
    pub saturator: S,
}

#[profiling::all_functions]
impl<T: Scalar, S: Saturator<T>> Saturator<T> for Driven<T, S> {
    fn saturate(&self, x: T) -> T {
        self.saturator.saturate(x * self.drive) / self.drive
    }

    #[inline(always)]
    fn update_state(&mut self, x: T, y: T) {
        let x = x / self.drive;
        let y = self.drive * y;
        self.saturator.update_state(x, y);
    }

    fn sat_diff(&self, x: T) -> T {
        self.saturator.sat_diff(x * self.drive)
    }
}
