//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use nalgebra::{Complex};
use numeric_literals::replace_float_literals;

use crate::{
    saturators::{Linear, Saturator}, Scalar,
};
use crate::dsp::analog::DspAnalog;
use crate::dsp::DSP;

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
        self.sats[0].update_state(s1 / 10.);
        self.sats[1].update_state(s2 / 10.);
        [lp, bp, hp]
    }
}

impl<T: Scalar, S: Saturator<T>> DspAnalog<1, 3> for Svf<T, S> {
    #[replace_float_literals(Complex::new(T::from_f64(literal), T::zero()))]
    fn h_s(&self, s: [Complex<Self::Sample>; 1]) -> [Complex<Self::Sample>; 3] {
        let s = s[0];
        let wc = 2. * T::simd_pi() * self.freq_cutoff();
        let s2 = s.powi(2);
        let wc2 = wc.powi(2);
        let denom = s2 + 2. * self.r * wc * s + wc2;
        let hp = s2 / denom;
        let bp = 2. * self.r * wc * s / denom;
        let lp = wc2 / denom;
        [lp, bp, hp]
    }
}

impl<T: Scalar, C: Default> Svf<T, C> {
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

    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_samplerate(&mut self, samplerate: T) {
        self.w_step = T::simd_pi() / samplerate;
        self.update_coefficients();
    }

    pub fn set_cutoff(&mut self, freq: T) {
        self.fc = freq;
        self.update_coefficients();
    }

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
    pub fn set_saturators(&mut self, s1: S, s2: S) {
        self.sats = [s1, s2];
    }

    pub fn with_saturators(mut self, s1: S, s2: S) -> Self {
        self.set_saturators(s1, s2);
        self
    }
}
