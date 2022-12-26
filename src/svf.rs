//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use crate::{
    saturators::{Linear, Saturator},
    Scalar,
    DSP
};
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;

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
    #[replace_float_literals(T::from(literal).unwrap())]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 3] {
        let [s1, s2] = self.s;

        let hp = (x[0] - self.g1 * s1 - s2) * self.d;

        let v1 = self.g * hp;
        let bp = v1 + s1;
        let s1 = bp + v1;

        let v2 = self.g * bp;
        let lp = v2 + s2;
        let s2 = lp + v2;

        self.s = [self.sats[0].saturate(s1), self.sats[1].saturate(s2)];
        [lp, bp, hp]
    }
}

impl<T: Scalar, C: Default> Svf<T, C> {
    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn new(samplerate: T, fc: T, r: T) -> Self {
        let mut this = Self {
            s: [T::EQUILIBRIUM; 2],
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
        self.s.fill(T::EQUILIBRIUM);
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn set_samplerate(&mut self, samplerate: T) {
        self.w_step = 0.5 * T::PI() / samplerate;
        self.update_coefficients();
    }

    pub fn set_fc(&mut self, fc: T) {
        self.fc = fc;
        self.update_coefficients();
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn set_r(&mut self, r: T) {
        self.r = 2. * r;
        self.update_coefficients();
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn update_coefficients(&mut self) {
        self.g = self.w_step * self.fc;
        self.g1 = 2. * self.r + self.g;
        self.d = (1. + 2. * self.r * self.g + self.g * self.g).recip();
    }
}
