use crate::Scalar;
use num_traits::{Float, NumAssign};
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;

pub trait Saturator<T: Scalar>: Default {
    /// Saturate an input with a frozen state.
    fn saturate(&self, x: T) -> T;

    /// Update the state given an input.
    #[inline(always)]
    fn update_state(&mut self, x: T) {}

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

#[derive(Debug, Copy, Clone)]
pub struct Hysteresis<T: Scalar, S: Saturator<T>> {
    alpha: T,
    c: T,
    last: T,
    sat: S,
}

impl<T: Scalar, S: Saturator<T>> Default for Hysteresis<T, S> {
    fn default() -> Self {
        Self {
            alpha: T::one(),
            c: T::one(),
            last: T::EQUILIBRIUM,
            sat: S::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Dynamic {
    Linear,
    Tanh,
    Clipper,
}

impl<T: Scalar> Saturator<T> for Dynamic {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        match self {
            Self::Linear => Linear.saturate(x),
            Self::Clipper => Clipper.saturate(x),
            Self::Tanh => Tanh.saturate(x),
        }
    }

    #[inline(always)]
    fn sat_diff(&self, x: T) -> T {
        match self {
            Self::Linear => Linear.sat_diff(x),
            Self::Clipper => Clipper.sat_diff(x),
            Self::Tanh => Tanh.sat_diff(x),
        }
    }
}

impl Default for Dynamic {
    fn default() -> Self {
        Self::Linear
    }
}
