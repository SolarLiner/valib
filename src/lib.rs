use dasp_sample::FloatSample;
use nalgebra::{Complex, ComplexField, RealField};
use num_traits::{Float, FloatConst};
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;

// pub mod ladder;
// pub mod sallenkey;
pub mod biquad;
pub mod saturators;
pub mod svf;

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

#[derive(Debug, Copy, Clone, Default)]
pub struct Bypass<S>(PhantomData<S>);

impl<S: Scalar, const N: usize> DSP<N, N> for Bypass<S> {
    type Sample = S;

    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        x
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

impl<P: DSP<N, N>, const A: usize, const N: usize> DSP<N, N> for [P; A] {
    type Sample = P::Sample;

    #[inline(always)]
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        self.iter_mut().fold(x, |x, f| f.process(x))
    }
}

macro_rules! series_tuple {
    ($($p:ident),*) => {
        impl<__Sample: Scalar, $($p: DSP<N, N, Sample = __Sample>),*, const N: usize> DSP<N, N> for ($($p),*) {
            type Sample = __Sample;

            #[allow(non_snake_case)]
            #[inline(always)]
            fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
                let ($($p),*) = self;
                $(
                let x = $p.process(x);
                )*
                x
            }
        }
    };
}

series_tuple!(A, B);
series_tuple!(A, B, C);
series_tuple!(A, B, C, D);
series_tuple!(A, B, C, D, E);
series_tuple!(A, B, C, D, E, F);
series_tuple!(A, B, C, D, E, F, G);
series_tuple!(A, B, C, D, E, F, G, H);
