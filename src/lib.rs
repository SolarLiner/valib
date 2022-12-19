//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use std::marker::PhantomData;
use dasp_sample::FloatSample;
use num_traits::{Float, FloatConst};

pub trait Scalar: Float + FloatConst + FloatSample {}

impl<T: Float + FloatConst + FloatSample> Scalar for T {}

pub trait DSP<const I: usize, const O: usize> {
    type Sample: Scalar;

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O];
}

/// Freestanding integrator, discretized with TPT
#[derive(Debug, Copy, Clone)]
pub struct Integrator<T>(T);

impl<T: Scalar> Default for Integrator<T> {
    fn default() -> Self {
        Self(T::EQUILIBRIUM)
    }
}

impl<T: Scalar> DSP<1, 1> for Integrator<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let in0 = x[0] + self.0;
        self.0 += in0;
        self.0
    }
}

/// 6 dB/oct one-pole filter using the "one-sample trick" (fig. 3.31, eq. 3.32).
/// Outputs modes as follows: [LP, HP, AP].
#[derive(Debug, Copy, Clone)]
pub struct P1<T> {
    w_step: T,
    fc: T,
    s: T,
}

impl<T: Scalar, P> P1<T> {
    pub fn new(samplerate: T, fc: T) -> Self {
        Self {
            w_step: T::PI() / samplerate,
            fc,
            s: T::EQUILIBRIUM,
        }
    }

    pub fn set_fc(&mut self, fc: T) {
        self.fc = fc;
    }
}

impl<T: Scalar> DSP<1, 3> for P1<T> {
    type Sample = T;

    #[inline(always)]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 3] {
        // One-sample feedback trick over a transposed integrator, implementation following
        // eq (3.32), page 77
        let g = self.w_step * self.fc;
        let k = g / (1. + g);
        let v = k * (x[0] - self.s);
        let lp = v + self.s;
        self.s = lp + v;

        let hp = x[0] - lp;
        let ap = 2.*lp - x[0];
        [lp, hp, ap]
    }
}
