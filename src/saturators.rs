use crate::clippers::DiodeClipperModel;
use numeric_literals::replace_float_literals;
use simba::simd::SimdValue;

use crate::Scalar;

pub trait Saturator<T: Scalar>: Default {
    /// Saturate an input with a frozen state.
    fn saturate(&self, x: T) -> T;

    /// Update the state given an input.
    #[inline(always)]
    fn update_state(&mut self, _x: T) {}

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn sat_diff(&self, x: T) -> T {
        (self.saturate(x + 1e-4) - self.saturate(x)) / 1e-4
    }
}

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

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Tanh;

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

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Clipper;

impl<S: Scalar> Saturator<S> for Clipper {
    #[inline(always)]
    #[replace_float_literals(S::from_f64(literal))]
    fn saturate(&self, x: S) -> S {
        x.simd_min(1.0).simd_max(-1.0)
    }

    #[inline(always)]
    #[replace_float_literals(S::from_f64(literal))]
    fn sat_diff(&self, x: S) -> S {
        let mask = x.simd_abs().simd_gt(1.0);
        (1.0).select(mask, 0.0)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Blend<T, S> {
    amt: T,
    inner: S,
}

impl<T: Scalar, S: Saturator<T>> Saturator<T> for Blend<T, S> {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        x + self.amt * (self.inner.saturate(x) - x)
    }

    #[inline(always)]
    fn update_state(&mut self, x: T) {
        self.inner.update_state(x)
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
    HardClipper,
    DiodeClipper(DiodeClipperModel<T>),
    SoftClipper(Blend<T, DiodeClipperModel<T>>),
}

impl<T: Scalar> Saturator<T> for Dynamic<T> {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        match self {
            Self::Linear => Linear.saturate(x),
            Self::HardClipper => Clipper.saturate(x),
            Self::Tanh => Tanh.saturate(x),
            Self::DiodeClipper(clip) => clip.saturate(x),
            Self::SoftClipper(clip) => clip.saturate(x),
        }
    }

    #[inline(always)]
    fn sat_diff(&self, x: T) -> T {
        match self {
            Self::Linear => Linear.sat_diff(x),
            Self::HardClipper => Clipper.sat_diff(x),
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
