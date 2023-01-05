use num_traits::NumCast;
use numeric_literals::replace_float_literals;

use crate::Scalar;

pub trait Saturator<T: Scalar>: Default {
    /// Saturate an input with a frozen state.
    fn saturate(&self, x: T) -> T;

    /// Update the state given an input.
    #[inline(always)]
    fn update_state(&mut self, _x: T) {}

    #[inline(always)]
    #[replace_float_literals(T::from(literal).unwrap())]
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
        x.tanh()
    }

    #[inline(always)]
    #[replace_float_literals(S::from(literal).unwrap())]
    fn sat_diff(&self, x: S) -> S {
        1. - x.tanh().powi(2)
    }
}

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct Clipper;

impl<S: Scalar> Saturator<S> for Clipper {
    #[inline(always)]
    fn saturate(&self, x: S) -> S {
        x.min(S::one()).max(S::one().neg())
    }

    #[inline(always)]
    #[replace_float_literals(S::from(literal).unwrap())]
    fn sat_diff(&self, x: S) -> S {
        if x < -1. || x > 1. {
            0.
        } else {
            1.
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DiodeClipper<T> {
    param: T,
}

impl<T: Scalar> Default for DiodeClipper<T> {
    fn default() -> Self {
        // Parameter fit from a KiCad simulation of a buffer -> 2x3 diode clipper stage from -12V to 12V,
        // which yields a maximum difference of -20 dB.
        Self {
            param: T::from(0.75975449).unwrap(),
        }
    }
}

impl<T: Scalar> Saturator<T> for DiodeClipper<T> {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        self.param * x / (self.param + x.abs())
    }

    #[inline(always)]
    fn sat_diff(&self, x: T) -> T {
        let frac = self.param / (self.param + x.abs());
        frac.powi(2)
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

impl<T: NumCast, S: Default> Default for Blend<T, S> {
    fn default() -> Self {
        Self {
            amt: T::from(0.5).unwrap(),
            inner: S::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Dynamic<T> {
    Linear,
    Tanh,
    HardClipper,
    DiodeClipper(DiodeClipper<T>),
    SoftClipper(Blend<T, DiodeClipper<T>>),
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
