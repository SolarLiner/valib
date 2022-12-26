//! Transposed Direct Form II Biquad implementation - nonlinearities based on https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-4-nonlinear-biquad-filters-ae6b3f23cb0e

use crate::saturators::{Dynamic, Saturator};
use crate::DSP;
use crate::{DspAnalysis, Scalar};
use nalgebra::Complex;
use num_traits::real::Real;
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;
use std::ops::Neg;

#[derive(Debug, Copy, Clone)]
pub struct Biquad<T, S> {
    na: [T; 2],
    b: [T; 3],
    s: [T; 2],
    sats: [S; 2],
}

impl<T> Biquad<T, Dynamic> {
    pub fn set_saturators(&mut self, a: Dynamic, b: Dynamic) {
        self.sats = [a,b];
    }
}

impl<T: Copy, S> Biquad<T, S> {
    pub fn update_coefficients(&mut self, other: &Self) {
        self.na = other.na;
        self.b = other.b;
    }
}

impl<T: Scalar, S: Default> Biquad<T, S> {
    pub fn new(b: [T; 3], a: [T; 2]) -> Self {
        Self {
            na: a.map(T::neg),
            b,
            s: [T::EQUILIBRIUM; 2],
            sats: Default::default(),
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn lowpass(fc: T, q: T) -> Self {
        let w0 = T::TAU() * fc;
        let (sw0, cw0) = w0.sin_cos();
        let b1 = 1. - cw0;
        let b0 = b1 / 2.;
        let b2 = b0;

        let alpha = sw0 / (2. * q);
        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2], [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn highpass(fc: T, q: T) -> Self {
        let w0 = T::TAU() * fc;
        let (sw0, cw0) = w0.sin_cos();
        let b0 = 1. + cw0;
        let b1 = -(1. + cw0);
        let b2 = b0;

        let alpha = sw0 / (2. * q);
        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2], [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn bandpass_peak0(fc: T, q: T) -> Self {
        let w0 = T::TAU() * fc;
        let (sw0, cw0) = w0.sin_cos();
        let alpha = sw0 / (2. * q);

        let b0 = alpha;
        let b1 = 0.;
        let b2 = -alpha;

        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    pub fn reset(&mut self) {
        self.s.fill(T::EQUILIBRIUM);
    }
}

impl<T: Scalar, S: Saturator<T>> DSP<1, 1> for Biquad<T, S> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let x = x[0];
        let in0 = x * self.b[0] + self.s[0];
        let in1 = x * self.b[1] + self.s[1] + self.sats[0].saturate(in0) * self.na[0];
        let in2 = x * self.b[2] + self.sats[1].saturate(in0) * self.na[1];
        self.s = [in1, in2];
        self.sats[0].update_state(in0);
        self.sats[1].update_state(in0);
        [in0]
    }
}

impl<T: Scalar, S> DspAnalysis<1, 1> for Biquad<T, S>
where
    Self: DSP<1, 1, Sample = T>,
{
    fn h_z(&self, z: [Complex<Self::Sample>; 1]) -> [Complex<Self::Sample>; 1] {
        let z: Complex<T> = z[0];
        let num = z.powi(-1).scale(self.b[1]) + z.powi(-2).scale(self.b[2]) + self.b[0];
        let den = z.powi(-1).scale(-self.na[0]) + z.powi(-2).scale(-self.na[1]) + T::one();
        [num / den]
    }
}
