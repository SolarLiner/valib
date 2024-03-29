//! Small [`DSPProcess`] building blocks for reusability.
use std::marker::PhantomData;

use nalgebra::{Complex, ComplexField, SMatrix, SVector};
use num_traits::{One, Zero};
use numeric_literals::replace_float_literals;

use crate::dsp::analysis::DspAnalysis;
use crate::dsp::{DSPMeta, DSPProcess};
use crate::Scalar;

/// "Bypass" struct, which simply forwards the input to the output.
#[derive(Debug, Copy, Clone, Default)]
pub struct Bypass<S>(PhantomData<S>);

impl<T: Scalar> DSPMeta for Bypass<T> {
    type Sample = T;
}

impl<T: Scalar, const N: usize> DSPProcess<N, N> for Bypass<T>
where
    Self: DSPMeta<Sample = T>,
{
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

impl<T: Scalar> DSPMeta for Integrator<T> {
    type Sample = T;

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.0 = T::zero();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for Integrator<T>
where
    Self: DSPMeta<Sample = T>,
{
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let x2 = x[0] / 2.0;
        let out = x2 + self.0;
        self.0 = x2 + out;
        [out]
    }
}

impl<T: Scalar> DspAnalysis<1, 1> for Integrator<T>
where
    Self: DSPProcess<1, 1, Sample = T>,
{
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 1] {
        [[1. / 2. * (z + 1.) / (z - 1.)]]
    }
}

/// Sum inputs into a single output
#[derive(Debug, Copy, Clone, Default)]
pub struct Sum<T, const N: usize>(PhantomData<[T; N]>);

impl<T: Scalar, const N: usize> DSPMeta for Sum<T, N> {
    type Sample = T;
}

impl<T, const N: usize> DSPProcess<N, 1> for Sum<T, N>
where
    Self: DSPMeta<Sample = T>,
    T: Scalar,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; 1] {
        [x.into_iter().fold(T::zero(), |a, b| a + b)]
    }
}

impl<T, const N: usize> DspAnalysis<N, 1> for Sum<T, N>
where
    Self: DSPProcess<N, 1>,
{
    fn h_z(&self, _z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; N] {
        [[Complex::one()]; N]
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

impl<T: Scalar> DSPMeta for P1<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.w_step = T::simd_pi() / T::from_f64(samplerate as _)
    }

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.s = T::zero();
    }
}

impl<T: Scalar> DspAnalysis<1, 3> for P1<T>
where
    Self: DSPProcess<1, 3, Sample = T>,
    Self::Sample: nalgebra::RealField,
{
    #[replace_float_literals(Complex::from_real(< T as Scalar >::from_f64(literal)))]
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 3]; 1] {
        let lp = (z - 1.0) / (z + 1.0) * self.fc / 2.0;
        let hp = 1.0 - lp;
        let ap = 2.0 * lp - 1.0;
        [[lp, hp, ap]]
    }
}

impl<T: Scalar> P1<T> {
    pub fn new(samplerate: T, fc: T) -> Self {
        Self {
            w_step: T::simd_pi() / samplerate,
            fc,
            s: T::zero(),
        }
    }

    pub fn with_state(mut self, state: T) -> Self {
        self.s = state;
        self
    }

    pub fn set_fc(&mut self, fc: T) {
        self.fc = fc;
    }
}

impl<T: Scalar> DSPProcess<1, 3> for P1<T>
where
    Self: DSPMeta<Sample = T>,
{
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

/// Process inner DSP blocks in series. `DSP` is implemented for tuples up to 8 elements all the same I/O configuration.
#[derive(Debug, Copy, Clone)]
pub struct Series<T>(pub T);

macro_rules! series_tuple {
    ($($p:ident),*) => {
        #[allow(non_snake_case)]
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSPMeta<Sample = __Sample>),*> DSPMeta for $crate::dsp::blocks::Series<($($p),*)> {
            type Sample = __Sample;

            fn set_samplerate(&mut self, samplerate: f32) {
                let Self(($($p),*)) = self;
                $(
                $p.set_samplerate(samplerate);
                )*
            }

            fn latency(&self) -> usize {
                let Self(($($p),*)) = self;
                0 $(
                + $p.latency()
                )*
            }

            fn reset(&mut self) {
                let Self(($($p),*)) = self;
                $(
                $p.reset();
                )*
            }
        }

        #[allow(non_snake_case)]
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSPProcess<N, N, Sample = __Sample>),*, const N: usize> DSPProcess<N, N> for $crate::dsp::blocks::Series<($($p),*)> {
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

impl<P: DSPMeta, const C: usize> DSPMeta for Series<[P; C]> {
    type Sample = P::Sample;

    fn latency(&self) -> usize {
        self.0.iter().map(|p| p.latency()).sum()
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        for p in &mut self.0 {
            p.set_samplerate(samplerate);
        }
    }

    fn reset(&mut self) {
        for p in &mut self.0 {
            p.reset();
        }
    }
}

impl<P: DSPProcess<N, N>, const N: usize, const C: usize> DSPProcess<N, N> for Series<[P; C]>
where
    Self: DSPMeta<Sample = P::Sample>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        self.0.iter_mut().fold(x, |x, dsp| dsp.process(x))
    }
}

impl<P, const N: usize, const C: usize> DspAnalysis<N, N> for Series<[P; C]>
where
    Self: DSPProcess<N, N, Sample = P::Sample>,
    P: DspAnalysis<N, N>,
{
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; N]; N] {
        self.0.iter().fold([[Complex::one(); N]; N], |acc, f| {
            let ret = f.h_z(z);
            std::array::from_fn(|i| std::array::from_fn(|j| acc[i][j] * ret[i][j]))
        })
    }
}

/// Specialized `Series` struct that doesn't restrict the I/O count of either DSP struct
#[derive(Debug, Copy, Clone)]
pub struct Series2<A, B, const INNER: usize>(A, PhantomData<[(); INNER]>, B);

impl<A, B, const INNER: usize> Series2<A, B, INNER> {
    /// Construct a new `Series2` instance, with each inner DSP instance given.
    pub const fn new<const I: usize, const O: usize>(a: A, b: B) -> Self
    where
        A: DSPProcess<I, INNER>,
        B: DSPProcess<INNER, O>,
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

impl<A, B, const J: usize> DSPMeta for Series2<A, B, J>
where
    A: DSPMeta,
    B: DSPMeta<Sample = A::Sample>,
{
    type Sample = A::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.0.set_samplerate(samplerate);
        self.2.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        let Self(a, _, b) = self;
        a.latency() + b.latency()
    }

    fn reset(&mut self) {
        let Self(a, _, b) = self;
        a.reset();
        b.reset();
    }
}

impl<A, B, const I: usize, const J: usize, const O: usize> DSPProcess<I, O> for Series2<A, B, J>
where
    Self: DSPMeta<Sample = A::Sample>,
    A: DSPProcess<I, J>,
    B: DSPProcess<J, O, Sample = A::Sample>,
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        let Self(a, _, b) = self;
        let j = a.process(x);
        b.process(j)
    }
}

impl<A, B, const I: usize, const J: usize, const O: usize> DspAnalysis<I, O> for Series2<A, B, J>
where
    Self: DSPProcess<I, O>,
    A: DspAnalysis<I, J, Sample = Self::Sample>,
    B: DspAnalysis<J, O, Sample = Self::Sample>,
{
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; O]; I] {
        let ha = SMatrix::<_, J, I>::from(self.0.h_z(z));
        let hb = SMatrix::<_, O, J>::from(self.2.h_z(z));
        let res = hb * ha;
        res.into()
    }
}

/// Process inner DSP blocks in parallel. Input is fanned out to all inner blocks, then summed back out.
#[derive(Debug, Copy, Clone)]
pub struct Parallel<T>(pub T);

macro_rules! parallel_tuple {
    ($($p:ident),*) => {
        #[allow(non_snake_case)]
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSPMeta<Sample = __Sample>),*> $crate::dsp::DSPMeta for $crate::dsp::blocks::Parallel<($($p),*)> {
            type Sample = __Sample;

            fn latency(&self) -> usize {
                let Self(($($p),*)) = self;
                let latency = 0;
                $(
                let latency = latency.max($p.latency());
                )*
                latency
            }

            fn set_samplerate(&mut self, samplerate: f32) {
                let Self(($($p),*)) = self;
                $(
                $p.set_samplerate(samplerate);
                )*
            }

            fn reset(&mut self) {
                let Self(($($p),*)) = self;
                $(
                $p.reset();
                )*
            }
        }

        #[allow(non_snake_case)]
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSPProcess<N, N, Sample = __Sample>),*, const N: usize> $crate::dsp::DSPProcess<N, N> for $crate::dsp::blocks::Parallel<($($p),*)> {
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

impl<P: DSPMeta, const C: usize> DSPMeta for Parallel<[P; C]> {
    type Sample = P::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        for s in &mut self.0 {
            s.set_samplerate(samplerate);
        }
    }

    fn latency(&self) -> usize {
        self.0.iter().fold(0, |max, dsp| max.max(dsp.latency()))
    }

    fn reset(&mut self) {
        for dsp in self.0.iter_mut() {
            dsp.reset();
        }
    }
}

impl<P: DSPProcess<I, O>, const I: usize, const O: usize, const N: usize> DSPProcess<I, O>
    for Parallel<[P; N]>
where
    Self: DSPMeta<Sample = P::Sample>,
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        self.0
            .iter_mut()
            .map(|dsp| dsp.process(x))
            .fold([Self::Sample::from_f64(0.0); O], |out, dsp| {
                std::array::from_fn(|i| out[i] + dsp[i])
            })
    }
}

impl<P, const I: usize, const O: usize, const N: usize> DspAnalysis<I, O> for Parallel<[P; N]>
where
    Self: DSPProcess<I, O, Sample = P::Sample>,
    P: DspAnalysis<I, O>,
{
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; O]; I] {
        self.0.iter().fold([[Complex::zero(); O]; I], |acc, f| {
            let ret = f.h_z(z);
            std::array::from_fn(|i| std::array::from_fn(|j| acc[i][j] + ret[i][j]))
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

impl<T, const I: usize, const O: usize> DSPMeta for ModMatrix<T, I, O>
where
    T: Scalar,
{
    type Sample = T;
}

impl<T, const I: usize, const O: usize> DSPProcess<I, O> for ModMatrix<T, I, O>
where
    Self: DSPMeta<Sample = T>,
    T: Scalar,
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        let res = self.weights * SVector::from(x);
        std::array::from_fn(|i| res[i])
    }
}

/// Feedback adapter with a one-sample delay and integrated mixing and summing point.
pub struct Feedback<FF, FB, const N: usize>
where
    FF: DSPProcess<N, N>,
{
    memory: [FF::Sample; N],
    /// Inner DSP instance
    pub feedforward: FF,
    pub feedback: FB,
    /// Mixing vector, which is lanewise-multiplied from the output and summed back to the input at the next sample.
    pub mix: [FF::Sample; N],
}

impl<FF, FB, const N: usize> DSPMeta for Feedback<FF, FB, N>
where
    FF: DSPProcess<N, N>,
    FB: DSPProcess<N, N, Sample = FF::Sample>,
{
    type Sample = FF::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.feedforward.set_samplerate(samplerate);
        self.feedback.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.feedforward.latency()
    }

    fn reset(&mut self) {
        self.memory.fill(FB::Sample::from_f64(0.0));
        self.feedforward.reset();
        self.feedback.reset();
    }
}

impl<FF, const N: usize> DSPMeta for Feedback<FF, (), N>
where
    FF: DSPProcess<N, N>,
{
    type Sample = FF::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.feedforward.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.feedforward.latency()
    }

    fn reset(&mut self) {
        self.memory.fill(Self::Sample::zero());
        self.feedforward.reset();
    }
}

impl<FF: DSPProcess<N, N>, const N: usize> DSPProcess<N, N> for Feedback<FF, (), N>
where
    Self: DSPMeta<Sample = FF::Sample>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        let x = std::array::from_fn(|i| self.memory[i] * self.mix[i] + x[i]);
        let y = self.feedforward.process(x);
        self.memory = y;
        y
    }
}

impl<FF, FB, const N: usize> DSPProcess<N, N> for Feedback<FF, FB, N>
where
    Self: DSPMeta<Sample = FF::Sample>,
    FF: DSPProcess<N, N>,
    FB: DSPProcess<N, N, Sample = FF::Sample>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        let fb = self.feedback.process(self.memory);
        let x = std::array::from_fn(|i| fb[i] * self.mix[i] + x[i]);
        let y = self.feedforward.process(x);
        self.memory = y;
        y
    }
}

impl<FF: DSPProcess<N, N>, FB, const N: usize> Feedback<FF, FB, N> {
    /// Create a new Feedback adapter with the provider inner DSP instance. Sets the mix to 0 by default.
    pub fn new(feedforward: FF, feedback: FB) -> Self {
        Self {
            memory: [FF::Sample::from_f64(0.0); N],
            feedforward,
            feedback,
            mix: [FF::Sample::from_f64(0.0); N],
        }
    }

    /// Unwrap this adapter and give back the inner DSP instance.
    pub fn into_inner(self) -> (FF, FB) {
        (self.feedforward, self.feedback)
    }
}
