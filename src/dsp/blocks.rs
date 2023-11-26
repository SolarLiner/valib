use numeric_literals::replace_float_literals;
use nalgebra::Complex;
use std::marker::PhantomData;
use crate::dsp::analysis::DspAnalysis;
use crate::dsp::DSP;
use crate::Scalar;

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
        Self(T::zero())
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
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
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
            w_step: T::simd_pi() / samplerate,
            fc,
            s: T::zero(),
        }
    }

    pub fn reset(&mut self) {
        self.s = T::zero();
    }

    pub fn set_samplerate(&mut self, samplerate: T) {
        self.w_step = T::simd_pi() / samplerate
    }

    pub fn set_fc(&mut self, fc: T) {
        self.fc = fc;
    }
}

impl<T: Scalar> DSP<1, 3> for P1<T> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
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

/// Process inner DSP blocks in series. `DSP` is implemented for tuples up to 8 elements all of the same I/O configuration.
#[derive(Debug, Copy, Clone)]
pub struct Series<T>(pub T);

macro_rules! series_tuple {
    ($($p:ident),*) => {
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSP<N, N, Sample = __Sample>),*, const N: usize> DSP<N, N> for $crate::dsp::blocks::Series<($($p),*)> {
            type Sample = __Sample;

            #[allow(non_snake_case)]
            #[inline(always)]
            fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
                let Self(($($p),*)) = self;
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

/// Process inner DSP blocks in parallel. Input is fanned out to all inner blocks, then summed back out.
#[derive(Debug, Copy, Clone)]
pub struct Parallel<T>(pub T);

macro_rules! parallel_tuple {
    ($($p:ident),*) => {
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSP<N, N, Sample = __Sample>),*, const N: usize> $crate::dsp::DSP<N, N> for $crate::dsp::blocks::Parallel<($($p),*)> {
            type Sample = __Sample;

            #[allow(non_snake_case)]
            #[inline(always)]
            fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
                let Self(($($p),*)) = self;
                let mut x = [Self::Sample::zero(); N];
                $(
                let y = $p.process(x);
                for i in 0..N {
                    x[i] += y[i];
                }
                )*
                x
            }
        }
    };
}

parallel_tuple!(A, B);
parallel_tuple!(A, B, C);
parallel_tuple!(A, B, C, D);
parallel_tuple!(A, B, C, D, E);
parallel_tuple!(A, B, C, D, E, F);
parallel_tuple!(A, B, C, D, E, F, G);
parallel_tuple!(A, B, C, D, E, F, G, H);
