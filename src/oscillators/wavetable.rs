use std::{marker::PhantomData, ops::Range};

use num_traits::Num;

use crate::{
    dsp::DSP,
    math::interpolation::{Interpolate, Linear},
    simd::SimdPartialOrd,
    Scalar, SimdCast,
};

/// Wavetable oscillator, reading samples from its internal array, with a customizable interpolation method
/// its DSP implementation expects a phasor signal as its first input
pub struct Wavetable<T, const N: usize, Interp = Linear, const I: usize = 2> {
    array: [T; N],
    __interp: PhantomData<Interp>,
}

impl<T: Scalar + SimdCast<isize>, const N: usize, const I: usize, Interp: Interpolate<T, I>>
    DSP<1, 1> for Wavetable<T, N, Interp, I>
where
    <T as SimdCast<isize>>::Output: Copy + Num + SimdPartialOrd,
{
    type Sample = T;
    fn process(&mut self, [phase]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = Interp::interpolate_on_slice(phase.simd_fract(), &self.array);
        [y]
    }
}

impl<T, const N: usize, Interp> Wavetable<T, N, Interp> {
    pub const fn new(array: [T; N]) -> Self {
        Self {
            array,
            __interp: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, Interp> Wavetable<T, N, Interp> {
    pub fn from_fn(range: Range<T>, f: impl Fn(T) -> T) -> Self {
        let r = range.end - range.start;
        let step = T::from_f64(N as f64) / r;
        Self::new(std::array::from_fn(|i| {
            let x = T::from_f64(i as f64) * step;
            f(x)
        }))
    }

    pub fn sin() -> Self {
        Self::from_fn(T::zero()..T::simd_two_pi(), |x| x.simd_sin())
    }

    pub fn cos() -> Self {
        Self::from_fn(T::zero()..T::simd_two_pi(), |x| x.simd_cos())
    }

    pub fn tan() -> Self {
        Self::from_fn(T::zero()..T::simd_two_pi(), |x| x.simd_tan())
    }
}
