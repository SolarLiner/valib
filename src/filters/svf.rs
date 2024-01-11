//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use nalgebra::Complex;
use num_traits::One;
use numeric_literals::replace_float_literals;

use crate::{
    dsp::analysis::DspAnalysis,
    saturators::{Linear, Saturator},
    Scalar,
};
use crate::dsp::DSP;

/// SVF topology filter, with optional non-linearities.
#[derive(Debug, Copy, Clone)]
pub struct Svf<T, Mode = Linear> {
    s: [T; 2],
    r: T,
    fc: T,
    g: T,
    g1: T,
    d: T,
    w_step: T,
    sats: [Mode; 2],
}

impl<T: Scalar, S: Saturator<T>> DSP<1, 3> for Svf<T, S> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 3] {
        let [s1, s2] = self.s;

        let hp = (x[0] - self.g1 * s1 - s2) * self.d;

        let v1 = self.g * hp;
        let bp = v1 + s1;
        let s1 = bp + v1;

        let v2 = self.g * bp;
        let lp = v2 + s2;
        let s2 = lp + v2;

        self.s = [
            self.sats[0].saturate(s1 / 10.) * 10.,
            self.sats[1].saturate(s2 / 10.) * 10.,
        ];
        self.sats[0].update_state(s1 / 10., self.s[0]);
        self.sats[1].update_state(s2 / 10., self.s[1]);
        [lp, bp, hp]
    }
}

impl<T: Scalar, S: Saturator<T>> DspAnalysis<1, 3> for Svf<T, S> {
    #[replace_float_literals(T::from_f64(literal))]
    fn h_z(
        &self,
        samplerate: Self::Sample,
        z: Complex<Self::Sample>,
    ) -> [[Complex<Self::Sample>; 3]; 1] {
        let omega_c = samplerate * self.fc;
        let x0 = z + Complex::one();
        let x1 = x0.powi(2) * omega_c.simd_powi(2);
        let x2 = z - Complex::one();
        let x3 = x2.powi(2) * 4.0 * samplerate.simd_powi(2);
        let x4 = x0 * x2 * samplerate * omega_c;
        let x5 = Complex::<T>::one() / (-x4 * 4.0 * self.r + x1 + x3);
        [[x1 * x5, -x4 * x5 * 2.0, x3 * x5]]
    }
}

impl<T: Scalar, C: Default> Svf<T, C> {
    /// Create a new SVF filter with the provided sample rate, frequency cutoff (in Hz) and resonance amount
    /// (in 0..1 for stable filters, otherwise use bounded nonlinearities).
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: T, fc: T, r: T) -> Self {
        let mut this = Self {
            s: [T::zero(); 2],
            r,
            fc,
            g: T::zero(),
            g1: T::zero(),
            d: T::zero(),
            w_step: T::zero(),
            sats: Default::default(),
        };
        this.set_samplerate(samplerate);
        this
    }

    pub fn reset(&mut self) {
        self.s.fill(T::zero());
    }

    /// Set the samplerate this filter will be running at.
    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_samplerate(&mut self, samplerate: T) {
        self.w_step = T::simd_pi() / samplerate;
        self.update_coefficients();
    }

    /// Set the new filter cutoff frequency (in Hz).
    pub fn set_cutoff(&mut self, freq: T) {
        self.fc = freq;
        self.update_coefficients();
    }

    /// Set the resonance amount (in 0..1 for stable filters, otherwise use bounded nonlinearities).
    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_r(&mut self, r: T) {
        self.r = 2. * r;
        self.update_coefficients();
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn update_coefficients(&mut self) {
        self.g = self.w_step * self.fc;
        self.g1 = 2. * self.r + self.g;
        self.d = (1. + 2. * self.r * self.g + self.g * self.g).simd_recip();
    }

    fn freq_cutoff(&self) -> T {
        self.g / self.w_step
    }
}

impl<T: Scalar, S: Saturator<T>> Svf<T, S> {
    /// Apply these new saturators to this SVF instance, returning a new instance of it.
    pub fn set_saturators(&mut self, s1: S, s2: S) {
        self.sats = [s1, s2];
    }

    /// Replace the saturators in this Biquad instance with the provided values.
    pub fn with_saturators(mut self, s1: S, s2: S) -> Self {
        self.set_saturators(s1, s2);
        self
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::ComplexField;

    use super::*;

    #[test]
    fn test_svf_hz() {
        let filter = Svf::<_, Linear>::new(1024.0, 10.0, 0.5);
        let hz: [_; 512] =
            std::array::from_fn(|i| i as f64).map(|f| filter.freq_response(1024.0, f)[0].map(|c| c.abs()));
        insta::assert_csv_snapshot!(&hz as &[_], { "[]" => insta::rounded_redaction(3)})
    }
}
