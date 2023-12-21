//! Transposed Direct Form II Biquad implementation - nonlinearities based on https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-5-nonlinear-feedback-filters-115e65fc0402

use crate::{
    saturators::{Dynamic, Saturator}, Scalar,
};
use nalgebra::Complex;
use numeric_literals::replace_float_literals;
use crate::dsp::analysis::DspAnalysis;
use crate::dsp::DSP;

#[derive(Debug, Copy, Clone)]
pub struct Biquad<T, S> {
    na: [T; 2],
    b: [T; 3],
    s: [T; 2],
    sats: [S; 2],
}

impl<T> Biquad<T, Dynamic<T>> {
    pub fn with_saturators(mut self, a: Dynamic<T>, b: Dynamic<T>) -> Biquad<T, Dynamic<T>> {
        self.set_saturators(a, b);
        self
    }

    pub fn set_saturators(&mut self, a: Dynamic<T>, b: Dynamic<T>) {
        self.sats = [a, b];
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
            s: [T::zero(); 2],
            sats: Default::default(),
        }
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn lowpass(fc: T, q: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let b1 = 1. - cw0;
        let b0 = b1 / 2.;
        let b2 = b0;

        let alpha = sw0 / (2. * q);
        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn highpass(fc: T, q: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let b1 = -(1. + cw0);
        let b0 = -b1 / 2.;
        let b2 = b0;

        let alpha = sw0 / (2. * q);
        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn bandpass_peak0(fc: T, q: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let alpha = sw0 / (2. * q);

        let b0 = alpha;
        let b1 = 0.;
        let b2 = -alpha;

        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn notch(fc: T, q: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let alpha = sw0 / (2. * q);

        let b0 = 1.;
        let b1 = -2. * cw0;
        let b2 = 1.;

        let a0 = 1. + alpha;
        let a1 = -2. * cw0;
        let a2 = 1. - alpha;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn allpass(fc: T, q: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let alpha = sw0 / (2. * q);

        let b0 = 1. - alpha;
        let b1 = -2. * cw0;
        let b2 = 1. + alpha;

        let a0 = b2;
        let a1 = b1;
        let a2 = b0;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn peaking(fc: T, q: T, amp: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let alpha = sw0 / (2. * q);

        let b0 = 1. + alpha * amp;
        let b1 = -2. * cw0;
        let b2 = 1. - alpha * amp;

        let a0 = 1. + alpha / amp;
        let a1 = b1;
        let a2 = 1. - alpha / amp;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn lowshelf(fc: T, q: T, amp: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let alpha = sw0 / (2. * q);

        let t = (amp + 1.) - (amp - 1.) * cw0;
        let tp = (amp - 1.) - (amp + 1.) * cw0;
        let u = 2. * amp.simd_sqrt() * alpha;

        let b0 = amp * (t + u);
        let b1 = 2. * amp * (tp);
        let b2 = amp * (t - u);

        let t = (amp + 1.) + (amp - 1.) * cw0;
        let a0 = t + u;
        let a1 = -2. * ((amp - 1.) + (amp + 1.) * cw0);
        let a2 = t - u;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn highshelf(fc: T, q: T, amp: T) -> Self {
        let w0 = T::simd_two_pi() * fc;
        let (sw0, cw0) = w0.simd_sin_cos();
        let alpha = sw0 / (2. * q);

        let b0 = amp * ((amp + 1.) + (amp - 1.) * cw0 + 2. * amp.simd_sqrt() * alpha);
        let b1 = -2. * amp * ((amp + 1.) + (amp - 1.) * cw0);
        let b2 = amp * ((amp + 1.) + (amp - 1.) * cw0 - (2. * amp.simd_sqrt() * alpha));

        let a0 = (amp + 1.) - (amp - 1.) * cw0 + 2. * amp.simd_sqrt() * alpha;
        let a1 = 2. * ((amp - 1.) - (amp + 1.) * cw0);
        let a2 = ((amp + 1.) - (amp - 1.) * cw0) - 2. * amp.simd_sqrt() * alpha;

        Self::new([b0, b1, b2].map(|b| b / a0), [a1, a2].map(|a| a / a0))
    }

    pub fn reset(&mut self) {
        self.s.fill(T::zero());
    }
}

impl<T: Scalar, S: Saturator<T>> DSP<1, 1> for Biquad<T, S> {
    type Sample = T;

    #[inline]
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let x = x[0];
        let in0 = x * self.b[0] + self.s[0];
        let in1 = x * self.b[1] + self.s[1] + self.sats[0].saturate(in0 / 10.) * 10. * self.na[0];
        let in2 = x * self.b[2] + self.sats[1].saturate(in0 / 10.) * 10. * self.na[1];
        self.s = [in1, in2];
        self.sats[0].update_state(in0 / 10.);
        self.sats[1].update_state(in0 / 10.);
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
