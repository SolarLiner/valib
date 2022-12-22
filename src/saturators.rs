use crate::Scalar;

pub trait Saturator<S: Scalar> {
    fn saturate(x: S) -> S;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Linear;

impl<S: Scalar> Saturator<S> for Linear {
    #[inline(always)]
    fn saturate(x: S) -> S {
        x
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Tanh;

impl<S: Scalar> Saturator<S> for Tanh {
    #[inline(always)]
    fn saturate(x: S) -> S {
        x.tanh()
    }
}

pub struct Clipper;

impl<S: Scalar> Saturator<S> for Clipper {
    #[inline(always)]
    fn saturate(x: S) -> S {
        x.min(S::one()).max(S::zero())
    }
}