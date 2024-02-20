use crate::dsp::DSP;
use numeric_literals::replace_float_literals;

use crate::saturators::adaa::{Antiderivative, Antiderivative2};
use clippers::DiodeClipperModel;

use crate::Scalar;

pub mod adaa;
pub mod clippers;

#[allow(unused_variables)]
pub trait Saturator<T: Scalar>: Default {
    /// Saturate an input with a frozen state.
    fn saturate(&self, x: T) -> T;

    /// Update the state given an input and the output of `saturate`.
    #[inline(always)]
    fn update_state(&mut self, x: T, y: T) {}

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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Asinh;

impl<T: Scalar> Saturator<T> for Asinh {
    fn saturate(&self, x: T) -> T {
        x.simd_asinh()
    }

    fn sat_diff(&self, x: T) -> T {
        let x0 = x * x + T::one();
        x0.simd_sqrt().simd_recip()
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

#[derive(Debug, Clone, Copy)]
pub struct Slew<T> {
    pub max_diff: T,
    last_out: T,
}

impl<T: Scalar> Default for Slew<T> {
    fn default() -> Self {
        Self {
            max_diff: T::from_f64(1.0),
            last_out: T::from_f64(0.0),
        }
    }
}

impl<T: Scalar> DSP<1, 1> for Slew<T> {
    type Sample = T;

    fn latency(&self) -> usize {
        0
    }

    fn reset(&mut self) {
        self.last_out = T::from_f64(0.0);
    }

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

impl<T: Scalar> Saturator<T> for Slew<T> {
    fn saturate(&self, x: T) -> T {
        self.slew(x)
    }

    fn sat_diff(&self, x: T) -> T {
        self.slew_diff(x)
    }

    fn update_state(&mut self, _x: T, y: T) {
        self.last_out = y;
    }
}
