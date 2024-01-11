use std::marker::PhantomData;

use nalgebra::{Complex, SMatrix, SVector};
use num_traits::{One, Zero};
use numeric_literals::replace_float_literals;

use crate::dsp::analysis::DspAnalysis;
use crate::dsp::DSP;
use crate::Scalar;

/// "Bypass" struct, which simply forwards the input to the output.
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
pub struct Integrator<T>(pub T);

impl<T: Scalar> Default for Integrator<T> {
    fn default() -> Self {
        Self(T::zero())
    }
}

impl<T: Scalar> DSP<1, 1> for Integrator<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let in0 = x[0] + self.0;
        self.0 += in0;
        [self.0]
    }
}

impl<T: Scalar> DspAnalysis<1, 1> for Integrator<T> {
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, z: [Complex<Self::Sample>; 1]) -> [Complex<Self::Sample>; 1] {
        [1. / 2. * (z[0] + 1.) / (z[0] - 1.)]
    }
}

/// Sum inputs into a single output
#[derive(Debug, Copy, Clone, Default)]
pub struct Sum<T, const N: usize>(PhantomData<[T; N]>);

impl<T, const N: usize> DSP<N, 1> for Sum<T, N>
where
    T: Scalar,
{
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; 1] {
        [x.into_iter().fold(T::zero(), |a, b| a + b)]
    }
}

impl<T, const N: usize> DspAnalysis<N, 1> for Sum<T, N>
where
    Self: DSP<N, 1>,
{
    fn h_z(&self, z: [Complex<Self::Sample>; N]) -> [Complex<Self::Sample>; 1] {
        [z.into_iter().fold(Complex::zero(), |a, b| a + b)]
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

/// Specialized `Series` struct that doesn't restrict the I/O count of either DSP struct
#[derive(Debug, Copy, Clone)]
pub struct Series2<A, B, const INNER: usize>(A, PhantomData<[(); INNER]>, B);

impl<A, B, const INNER: usize> Series2<A, B, INNER> {
    /// Construct a new `Series2` instance, with each inner DSP instance given.
    pub const fn new<const I: usize, const O: usize>(a: A, b: B) -> Self
    where
        A: DSP<I, INNER>,
        B: DSP<INNER, O>,
    {
        Self(a, PhantomData, b)
    }

    /// Returns a reference to the first DSP instance, which processes the incoming audio first.
    pub const fn left(&self) -> &A {
        &self.0
    }

    /// Returns a mutable reference to the first DSP instance, which processes the incoming audio first.
    pub fn left_mut(&mut self) -> &mut A {
        &mut self.0
    }

    /// Returns a reference to the second DSP instance, which processes the incoming audio last.
    pub const fn right(&self) -> &B {
        &self.2
    }

    /// Returns a mutable reference to the second DSP instance, which processes the incoming audio last.
    pub fn right_mut(&mut self) -> &mut B {
        &mut self.2
    }
}

impl<A, B, const I: usize, const J: usize, const O: usize> DSP<I, O> for Series2<A, B, J>
where
    A: DSP<I, J>,
    B: DSP<J, O, Sample = A::Sample>,
{
    type Sample = A::Sample;

    fn latency(&self) -> usize {
        let Self(a, _, b) = self;
        a.latency() + b.latency()
    }

    fn reset(&mut self) {
        let Self(a, _, b) = self;
        a.reset();
        b.reset();
    }

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        let Self(a, _, b) = self;
        let j = a.process(x);
        b.process(j)
    }
}

impl<P: DSP<N, N>, const N: usize, const C: usize> DSP<N, N> for Series<[P; C]> {
    type Sample = P::Sample;

    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        self.0.iter_mut().fold(x, |x, dsp| dsp.process(x))
    }

    fn latency(&self) -> usize {
        self.0.iter().map(|dsp| dsp.latency()).sum()
    }

    fn reset(&mut self) {
        for dsp in self.0.iter_mut() {
            dsp.reset();
        }
    }
}

impl<P, const N: usize, const C: usize> DspAnalysis<N, N> for Series<[P; C]>
where
    P: DspAnalysis<N, N>,
{
    fn h_z(&self, z: [Complex<Self::Sample>; N]) -> [Complex<Self::Sample>; N] {
        self.0.iter().fold([Complex::one(); N], |acc, f| {
            let ret = f.h_z(z);
            std::array::from_fn(|i| acc[i] * ret[i])
        })
    }
}

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
                let mut ret = [Self::Sample::zero(); N];
                $(
                let y = $p.process(x);
                for i in 0..N {
                    ret[i] += y[i];
                }
                )*
                ret
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

impl<P: DSP<I, O>, const I: usize, const O: usize, const N: usize> DSP<I, O> for Parallel<[P; N]> {
    type Sample = P::Sample;

    fn latency(&self) -> usize {
        self.0.iter().fold(0, |max, dsp| max.max(dsp.latency()))
    }

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        self.0
            .iter_mut()
            .map(|dsp| dsp.process(x))
            .fold([Self::Sample::from_f64(0.0); O], |out, dsp| {
                std::array::from_fn(|i| out[i] + dsp[i])
            })
    }

    fn reset(&mut self) {
        for dsp in self.0.iter_mut() {
            dsp.reset();
        }
    }
}

impl<P, const I: usize, const O: usize, const N: usize> DspAnalysis<I, O> for Parallel<[P; N]>
where
    P: DspAnalysis<I, O>,
{
    fn h_z(&self, z: [Complex<Self::Sample>; I]) -> [Complex<Self::Sample>; O] {
        self.0.iter().fold([Complex::zero(); O], |acc, f| {
            let ret = f.h_z(z);
            std::array::from_fn(|i| acc[i] + ret[i])
        })
    }
}

/// Mod matrix struct, with direct access to the summing matrix
#[derive(Debug, Copy, Clone)]
pub struct ModMatrix<T, const I: usize, const O: usize> {
    /// Mod matrix weights, setup in column-major form to produce outputs from inputs with a single matrix-vector
    /// multiplication.
    pub weights: SMatrix<T, O, I>,
}

impl<T, const I: usize, const O: usize> Default for ModMatrix<T, I, O>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            weights: SMatrix::from([[T::from_f64(0.0); O]; I]),
        }
    }
}

impl<T, const I: usize, const O: usize> DSP<I, O> for ModMatrix<T, I, O>
where
    T: Scalar,
{
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        let res = self.weights * SVector::from(x);
        std::array::from_fn(|i| res[i])
    }
}

/// Feedback adapter with a one-sample delay and integrated mixing and summing point.
pub struct Feedback<P, const N: usize> where P: DSP<N, N> {
    memory: [P::Sample; N],
    /// Inner DSP instance
    pub inner: P,
    /// Mixing vector, which is lanewise-multiplied from the output and summed back to the input at the next sample.
    pub mix: [P::Sample; N],
}

impl<P, const N: usize> DSP<N, N> for Feedback<P, N>
where P: DSP<N, N>
{
    type Sample = P::Sample;

    fn latency(&self) -> usize {
        self.inner.latency()
    }

    fn reset(&mut self) {
        self.memory.fill(P::Sample::from_f64(0.0));
        self.inner.reset();
    }

    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        let x = std::array::from_fn(|i| self.memory[i] * self.mix[i] + x[i]);
        let y = self.inner.process(x);
        self.memory = y;
        y
    }
}

impl<P: DSP<N, N>, const N: usize> Feedback<P, N> {
    /// Create a new Feedback adapter with the provider inner DSP instance. Sets the mix to 0 by default.
    pub fn new(dsp: P) -> Self {
        Self {
            memory: [P::Sample::from_f64(0.0); N],
            inner: dsp,
            mix: [P::Sample::from_f64(0.0); N],
        }
    }

    /// Unwrap this adapter and give back the inner DSP instance.
    pub fn into_inner(self) -> P {
        self.inner
    }
}
