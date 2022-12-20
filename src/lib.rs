//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use dasp_sample::FloatSample;
use nalgebra::{clamp, Complex, ComplexField, RealField};
use num_traits::{Float, FloatConst};
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;

pub trait Scalar: Float + FloatConst + FloatSample {}

impl<T: Float + FloatConst + FloatSample> Scalar for T {}

pub trait DSP<const I: usize, const O: usize> {
    type Sample: Scalar;

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O];
}

pub trait DspAnalysis<const I: usize, const O: usize>: DSP<I, O> {
    fn h_z(&self, z: [Complex<Self::Sample>; I]) -> [Complex<Self::Sample>; O];
    fn freq_response(&self, jw: [Self::Sample; I]) -> [Complex<Self::Sample>; O]
    where
        Self::Sample: RealField,
    {
        let z = jw.map(|jw| Complex::exp(Complex::i() * jw));
        self.h_z(z)
    }
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
        self.0 = self.0 + in0;
        [self.0]
    }
}

impl<T: Scalar> DspAnalysis<1, 1> for Integrator<T> {
    #[replace_float_literals(Complex::from(T::from(literal).unwrap()))]
    fn h_z(&self, z: [Complex<Self::Sample>; 1]) -> [Complex<Self::Sample>; 1] {
        [1. / 2. * (z[0] + 1.) / (z[0] - 1.)]
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

impl<T: Scalar> P1<T> {
    pub fn new(samplerate: T, fc: T) -> Self {
        Self {
            w_step: T::PI() / samplerate,
            fc,
            s: T::EQUILIBRIUM,
        }
    }

    pub fn reset(&mut self) {
        self.s = T::EQUILIBRIUM;
    }

    pub fn set_samplerate(&mut self, samplerate: T) {
        self.w_step = T::PI() / samplerate
    }

    pub fn set_fc(&mut self, fc: T) {
        self.fc = fc;
    }
}

impl<T: Scalar> DSP<1, 3> for P1<T> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 3] {
        // One-sample feedback trick over a transposed integrator, implementation following
        // eq (3.32), page 77
        let g = self.w_step * self.fc;
        let k = g / (1. + g);
        let v = k * (x[0] - self.s);
        let lp = v + self.s;
        self.s = lp + v;

        let hp = x[0] - lp;
        let ap = 2. * lp - x[0];
        [lp, hp, ap]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Clean;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Driven;

#[derive(Debug, Copy, Clone)]
pub struct Svf<T, Mode = Clean> {
    s: [T; 2],
    r: T,
    fc: T,
    g: T,
    g1: T,
    d: T,
    w_step: T,
    __mode: PhantomData<Mode>,
}

impl<T: Scalar> DSP<1, 3> for Svf<T, Clean> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 3] {
        let [s1, s2] = self.s;

        let hp = (x[0] - self.g1 * s1 - s2) * self.d;

        let v1 = self.g * hp;
        let bp = v1 + s1;
        let s1 = bp + v1;

        let v2 = self.g * hp;
        let lp = v2 + s2;
        let s2 = lp + v2;

        self.s = [s1, s2];
        [lp, bp, hp]
    }
}

impl<T: Scalar> DSP<1, 3> for Svf<T, Driven> {
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

        self.s = [s1.tanh(), s2.tanh()];
        [lp, bp, hp]
    }
}

impl<T: Scalar, C> Svf<T, C> {
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
            __mode: PhantomData,
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
