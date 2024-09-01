//! Small [`DSPProcess`] building blocks for reusability.
use core::fmt;
use std::borrow::Cow;
use std::marker::PhantomData;

use nalgebra::{allocator::SameShapeVectorAllocator, Complex, ComplexField, SMatrix, SVector};
use num_traits::{Euclid, One, Zero};
use numeric_literals::replace_float_literals;

use crate::dsp::{
    parameter::{ParamId, ParamName},
    DSPMeta, DSPProcess,
};
use crate::Scalar;
use crate::{dsp::analysis::DspAnalysis, util::lerp};

use super::parameter::{Dynamic, HasParameters, SmoothedParam};

/// "Bypass" struct, which simply forwards the input to the output.
#[derive(Debug, Copy, Clone)]
pub struct Bypass<S>(PhantomData<S>);

impl<T> Default for Bypass<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ParamName)]
pub enum P1Params {
    Cutoff,
}

/// 6 dB/oct one-pole filter (fig. 3.31, eq. 3.32).
/// Outputs modes as follows: [LP, HP, AP].
#[derive(Debug, Copy, Clone)]
pub struct P1<T> {
    pub fc: T,
    pub s: T,
    w_step: T,
}

impl<T: Scalar> HasParameters for P1<T> {
    type Name = P1Params;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            P1Params::Cutoff => self.fc = T::from_f64(value as _),
        }
    }
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

#[profiling::all_functions]
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
    ($params_name:ident: $count:literal; $($p:ident),*) => {
        #[derive(Debug, Copy, Clone)]
        pub enum $params_name<$($p),*> {
            $($p($p)),*
        }

        impl<$($p: $crate::dsp::parameter::ParamName),*> ParamName for $params_name<$($p),*> {
            fn count() -> usize {
                $count
            }

            #[allow(unused_variables)]
            fn from_id(value: ParamId) -> Self {
                $(
                    if value < $p::count() {
                        return Self::$p($p::from_id(value));
                    }
                    let value = value - $p::count();
                )*
                unreachable!();
            }

            #[allow(unused, non_snake_case)]
            fn into_id(self) -> ParamId {
                let mut acc = 0;
                let count = 0;
                $(
                    let $p = (count + acc) as ParamId;
                    let count = $p::count();
                    acc += count;
                )*
                match self {
                    $(
                    Self::$p(p) => $p + p.into_id(),
                    )*
                }
            }

            fn name(&self) -> Cow<'static, str> {
                match self {
                     $(
                     Self::$p(p) => Cow::Owned(format!("{} {}", stringify!($p), p.name())),
                     )*
                }
            }
        }

        #[allow(non_snake_case)]
        impl<$($p: $crate::dsp::parameter::HasParameters),*> HasParameters for $crate::dsp::blocks::Series<($($p),*)> {
            type Name = $params_name<$($p::Name),*>;

            fn set_parameter(&mut self, param: Self::Name, value: f32) {
                let Self(($($p),*)) = self;
                match param {
                    $($params_name::$p(p) => $p.set_parameter(p, value)),*
                }
            }
        }

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

        #[allow(non_snake_case, unused)]
        #[profiling::all_functions]
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSPProcess<N, N, Sample = __Sample>),*, const N: usize> DSPProcess<N, N> for $crate::dsp::blocks::Series<($($p),*)> {
            #[allow(non_snake_case)]
            #[inline(always)]
            fn process(&mut self, mut x: [Self::Sample; N]) -> [Self::Sample; N] {
                let Self(($($p),*)) = self;
                let mut i = 0;
                $(
                {
                    profiling::scope!("Series inner", &format!("{i}"));
                    x = $p.process(x);
                    i += 1;
                }
                )*
                x
            }
        }
    };
}

series_tuple!(Tuple2Params: 2; A, B);
series_tuple!(Tuple3Params: 3; A, B, C);
series_tuple!(Tuple4Params: 4; A, B, C, D);
series_tuple!(Tuple5Params: 5; A, B, C, D, E);
series_tuple!(Tuple6Params: 6; A, B, C, D, E, F);
series_tuple!(Tuple7Params: 7; A, B, C, D, E, F, G);
series_tuple!(Tuple8Params: 8; A, B, C, D, E, F, G, H);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TupleArrayParams<Name, const N: usize>(pub ParamId, pub Name);

impl<Name: ParamName, const N: usize> ParamName for TupleArrayParams<Name, N> {
    fn count() -> usize {
        N * Name::count()
    }

    fn from_id(value: ParamId) -> Self {
        let (div, rem) = value.div_rem_euclid(&(Name::count() as _));
        Self(div, Name::from_id(rem))
    }

    fn into_id(self) -> ParamId {
        Name::count() as ParamId * self.0 + self.1.into_id()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!("{} {}", self.1.name(), self.0))
    }

    fn iter() -> impl Iterator<Item = Self> {
        (0..N).flat_map(|i| Name::iter().map(move |e| Self(i as ParamId, e)))
    }
}

impl<P: HasParameters, const N: usize> HasParameters for Series<[P; N]> {
    type Name = TupleArrayParams<P::Name, N>;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            TupleArrayParams(i, p) => self.0[i].set_parameter(p, value),
        }
    }
}

impl<P: DSPMeta, const C: usize> DSPMeta for Series<[P; C]> {
    type Sample = P::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        for p in &mut self.0 {
            p.set_samplerate(samplerate);
        }
    }

    fn latency(&self) -> usize {
        self.0.iter().map(|p| p.latency()).sum()
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
    #[profiling::function]
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        self.0.iter_mut().enumerate().fold(x, |x, (_i, dsp)| {
            profiling::scope!("Series", &format!("{_i}"));
            dsp.process(x)
        })
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

impl<'a, P: DSPMeta> DSPMeta for Series<&'a mut [P]> {
    type Sample = P::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        for p in &mut *self.0 {
            p.set_samplerate(samplerate);
        }
    }

    fn latency(&self) -> usize {
        self.0.iter().map(|p| p.latency()).sum()
    }

    fn reset(&mut self) {
        for p in &mut *self.0 {
            p.reset();
        }
    }
}

impl<'a, P: DSPProcess<N, N>, const N: usize> DSPProcess<N, N> for Series<&'a mut [P]>
where
    Self: DSPMeta<Sample = P::Sample>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        self.0.iter_mut().enumerate().fold(x, |x, (_i, dsp)| {
            profiling::scope!("Series", &format!("{_i}"));
            dsp.process(x)
        })
    }
}

/// Specialized `Tuple` struct that doesn't restrict the I/O count of either DSP struct
#[derive(Debug, Copy, Clone)]
pub struct Tuple2<A, B, const INNER: usize>(A, PhantomData<[(); INNER]>, B);

impl<A: HasParameters, B: HasParameters, const INNER: usize> HasParameters for Tuple2<A, B, INNER> {
    type Name = Tuple2Params<A::Name, B::Name>;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            Tuple2Params::A(p) => self.0.set_parameter(p, value),
            Tuple2Params::B(p) => self.2.set_parameter(p, value),
        }
    }
}

impl<A, B, const INNER: usize> Tuple2<A, B, INNER> {
    /// Construct a new `Tuple2` instance, with each inner DSP instance given.
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

impl<A, B, const J: usize> DSPMeta for Tuple2<A, B, J>
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

#[profiling::all_functions]
impl<A, B, const I: usize, const J: usize, const O: usize> DSPProcess<I, O> for Tuple2<A, B, J>
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

impl<A, B, const I: usize, const J: usize, const O: usize> DspAnalysis<I, O> for Tuple2<A, B, J>
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
    ($params_name: ident; $($p:ident),*) => {
        #[allow(non_snake_case,unused)]
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

        #[allow(non_snake_case,unused)]
        impl<__Sample: $crate::Scalar, $($p: $crate::dsp::DSPProcess<N, N, Sample = __Sample>),*, const N: usize> $crate::dsp::DSPProcess<N, N> for $crate::dsp::blocks::Parallel<($($p),*)> {
            #[inline(always)]
            #[profiling::function]
            fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
                let Self(($($p),*)) = self;
                let mut ret = [Self::Sample::zero(); N];
                let mut n = 0;
                $(
                {
                    profiling::scope!("Parallel", &format!("{n}"));
                    let y = $p.process(x);
                    for i in 0..N {
                        ret[i] += y[i];
                    }
                    n += 1;
                }
                )*
                ret
            }
        }
    };
}

parallel_tuple!(Tuple2Params; A, B);
parallel_tuple!(Tuple3Params; A, B, C);
parallel_tuple!(Tuple4Params; A, B, C, D);
parallel_tuple!(Tuple5Params; A, B, C, D, E);
parallel_tuple!(Tuple6Params; A, B, C, D, E, F);
parallel_tuple!(Tuple7Params; A, B, C, D, E, F, G);
parallel_tuple!(Tuple8Params; A, B, C, D, E, F, G, H);

impl<P: HasParameters, const N: usize> HasParameters for Parallel<[P; N]> {
    type Name = TupleArrayParams<P::Name, N>;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            TupleArrayParams(i, p) => self.0[i].set_parameter(p, value),
        }
    }
}

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

#[profiling::all_functions]
impl<P: DSPProcess<I, O>, const I: usize, const O: usize, const N: usize> DSPProcess<I, O>
    for Parallel<[P; N]>
where
    Self: DSPMeta<Sample = P::Sample>,
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(_i, dsp)| {
                profiling::scope!("Parallel", &format!("{_i}"));
                dsp.process(x)
            })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModMatrixParams<const I: usize, const O: usize>(pub ParamId, pub ParamId);

impl<const I: usize, const O: usize> ParamName for ModMatrixParams<I, O> {
    fn count() -> usize {
        O * I
    }

    fn from_id(value: ParamId) -> Self {
        let (div, rem) = value.div_rem_euclid(&(I as _));
        Self(div, rem)
    }

    fn into_id(self) -> ParamId {
        self.0 * I as ParamId + self.1
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!("{} -> {}", self.0, self.1))
    }

    fn iter() -> impl Iterator<Item = Self> {
        (0..I).flat_map(|i| (0..O).map(move |o| Self(i as _, o as _)))
    }
}

/// Mod matrix struct, with direct access to the summing matrix
#[derive(Debug, Copy, Clone)]
pub struct ModMatrix<T, const I: usize, const O: usize> {
    /// Mod matrix weights, setup in column-major form to produce outputs from inputs with a single matrix-vector
    /// multiplication.
    pub weights: SMatrix<T, O, I>,
}

impl<T: Scalar, const I: usize, const O: usize> HasParameters for ModMatrix<T, I, O> {
    type Name = ModMatrixParams<I, O>;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            ModMatrixParams(inp, out) => self.weights[(out, inp)] = T::from_f64(value as _),
        }
    }
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

#[profiling::all_functions]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackParams<FF, FB, const N: ParamId> {
    Feedforward(FF),
    Feedback(FB),
    Mix(Dynamic<N>),
}

impl<FF: ParamName, FB: ParamName, const N: ParamId> ParamName for FeedbackParams<FF, FB, N> {
    fn count() -> usize {
        FF::count() + FB::count() + Dynamic::<N>::count()
    }

    fn from_id(value: ParamId) -> Self {
        if value < FF::count() as ParamId {
            return Self::Feedforward(FF::from_id(value));
        }
        let value = value - FF::count() as ParamId;
        if value < FB::count() as ParamId {
            return Self::Feedback(FB::from_id(value));
        }
        let value = value - FB::count() as ParamId;
        Self::Mix(Dynamic::from_id(value))
    }

    fn into_id(self) -> ParamId {
        match self {
            Self::Feedforward(p) => p.into_id(),
            Self::Feedback(p) => FF::count() as ParamId + p.into_id(),
            Self::Mix(p) => (FF::count() + FB::count()) as ParamId + p.into_id(),
        }
    }

    fn name(&self) -> Cow<'static, str> {
        match self {
            Self::Feedforward(p) => Cow::Owned(format!("FF: {}", p.name())),
            Self::Feedback(p) => Cow::Owned(format!("FB: {}", p.name())),
            Self::Mix(p) => Cow::Owned(format!("Mix Channel {}", p.into_id() + 1)),
        }
    }
}

/// Feedback adapter with a one-sample delay and integrated mixing and summing point.
pub struct Feedback<FF, FB, const N: usize>
where
    FF: DSPMeta,
{
    memory: [FF::Sample; N],
    /// Inner DSP instance
    pub feedforward: FF,
    pub feedback: FB,
    /// Mixing vector, which is lanewise-multiplied from the output and summed back to the input at the next sample.
    pub mix: [SmoothedParam; N],
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

#[profiling::all_functions]
impl<FF: DSPProcess<N, N>, const N: usize> DSPProcess<N, N> for Feedback<FF, (), N>
where
    Self: DSPMeta<Sample = FF::Sample>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        let mix = self
            .mix
            .each_mut()
            .map(|p| p.next_sample_as::<FF::Sample>());
        let x = std::array::from_fn(|i| self.memory[i] * mix[i] + x[i]);
        let y = self.feedforward.process(x);
        self.memory = y;
        y
    }
}

#[profiling::all_functions]
impl<FF, FB, const N: usize> DSPProcess<N, N> for Feedback<FF, FB, N>
where
    Self: DSPMeta<Sample = FF::Sample>,
    FF: DSPProcess<N, N>,
    FB: DSPProcess<N, N, Sample = FF::Sample>,
{
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        let mix = self
            .mix
            .each_mut()
            .map(|p| p.next_sample_as::<FF::Sample>());
        let fb = self.feedback.process(self.memory);
        let x = std::array::from_fn(|i| fb[i] * mix[i] + x[i]);
        let y = self.feedforward.process(x);
        self.memory = y;
        y
    }
}

impl<FF: DSPProcess<N, N>, FB, const N: usize> Feedback<FF, FB, N> {
    /// Create a new Feedback adapter with the provider inner DSP instance. Sets the mix to 0 by default.
    pub fn new(samplerate: f32, feedforward: FF, feedback: FB, mix_smoothing_ms: f32) -> Self {
        Self {
            memory: [FF::Sample::from_f64(0.0); N],
            feedforward,
            feedback,
            mix: [SmoothedParam::linear(0.0, samplerate, mix_smoothing_ms); N],
        }
    }

    /// Unwrap this adapter and give back the inner DSP instance.
    pub fn into_inner(self) -> (FF, FB) {
        (self.feedforward, self.feedback)
    }
}

impl<FF: DSPMeta + HasParameters, const N: usize> HasParameters for Feedback<FF, (), N> {
    type Name = FeedbackParams<FF::Name, Dynamic<0>, N>;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            FeedbackParams::Feedforward(p) => self.feedforward.set_parameter(p, value),
            FeedbackParams::Feedback(_) => unreachable!(),
            FeedbackParams::Mix(p) => self.mix[p.into_id() as usize].param = value,
        }
    }
}

pub struct SwitchAB<A, B> {
    pub a: A,
    pub b: B,
    switch: SmoothedParam,
}

impl<A, B> SwitchAB<A, B> {
    pub fn new(samplerate: f32, a: A, b: B, b_active: bool) -> Self {
        Self {
            a,
            b,
            switch: SmoothedParam::linear(if b_active { 1.0 } else { 0.0 }, samplerate, 50.),
        }
    }

    pub fn is_a_active(&self) -> bool {
        self.switch.current_value() < 0.995
    }

    pub fn is_b_active(&self) -> bool {
        self.switch.current_value() > 0.005
    }

    pub fn is_transitioning(&self) -> bool {
        self.switch.is_changing()
    }

    pub fn switch_to_a(&mut self) {
        self.switch.param = 0.;
    }

    pub fn switch_to_b(&mut self) {
        self.switch.param = 1.;
    }

    pub fn should_switch_to_b(&mut self, should_switch: bool) {
        if should_switch {
            self.switch_to_b()
        } else {
            self.switch_to_a()
        }
    }
}

impl<A: DSPMeta, B: DSPMeta<Sample = A::Sample>> DSPMeta for SwitchAB<A, B> {
    type Sample = A::Sample;

    fn latency(&self) -> usize {
        let la = if self.is_a_active() {
            self.a.latency()
        } else {
            0
        };
        let lb = if self.is_b_active() {
            self.b.latency()
        } else {
            0
        };
        la.max(lb)
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        self.switch.set_samplerate(samplerate);
        self.a.set_samplerate(samplerate);
        self.b.set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        self.switch.reset();
        self.a.reset();
        self.b.reset();
    }
}

impl<
        A: DSPProcess<I, O>,
        B: DSPProcess<I, O, Sample = A::Sample>,
        const I: usize,
        const O: usize,
    > DSPProcess<I, O> for SwitchAB<A, B>
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        let t = self.switch.next_sample_as();
        match (self.is_a_active(), self.is_b_active()) {
            (false, false) => unreachable!(),
            (true, false) => self.a.process(x),
            (false, true) => self.b.process(x),
            (true, true) => {
                let a = self.a.process(x);
                let b = self.b.process(x);
                std::array::from_fn(|i| lerp(t, a[i], b[i]))
            }
        }
    }
}
