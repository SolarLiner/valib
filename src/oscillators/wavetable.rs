use crate::dsp::DSPMeta;
use crate::math::interpolation::{SimdIndex, SimdInterpolatable};
use crate::{
    dsp::DSPProcess,
    math::interpolation::{Interpolate, Linear},
};
use std::ops::Range;
use valib_core::{Scalar, SimdCast};

/// Wavetable oscillator, reading samples from its internal array, with a customizable interpolation method
/// its DSP implementation expects a phasor signal as its first input
pub struct Wavetable<T, const N: usize, Interp = Linear, const I: usize = 2> {
    array: [T; N],
    interpolation: Interp,
}

impl<T: Scalar, Interp, const I: usize, const N: usize> DSPMeta for Wavetable<T, N, Interp, I> {
    type Sample = T;
}

#[profiling::all_functions]
impl<T: Scalar + SimdCast<isize>, const N: usize, const I: usize, Interp: Interpolate<T, I>>
    DSPProcess<1, 1> for Wavetable<T, N, Interp, I>
where
    T: Scalar + SimdInterpolatable,
    <T as SimdCast<usize>>::Output: SimdIndex,
{
    fn process(&mut self, [phase]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = self
            .interpolation
            .interpolate_on_slice(phase.simd_fract(), &self.array);
        [y]
    }
}

impl<T, const N: usize, Interp> Wavetable<T, N, Interp> {
    pub const fn new(interpolation: Interp, array: [T; N]) -> Self {
        Self {
            array,
            interpolation,
        }
    }
}

impl<T: Scalar, const N: usize, Interp> Wavetable<T, N, Interp> {
    pub fn from_fn(interpolation: Interp, range: Range<T>, f: impl Fn(T) -> T) -> Self {
        let r = range.end - range.start;
        let step = T::from_f64(N as f64) / r;
        Self::new(
            interpolation,
            std::array::from_fn(|i| {
                let x = T::from_f64(i as f64) * step;
                f(x)
            }),
        )
    }

    pub fn sin(interpolation: Interp) -> Self {
        Self::from_fn(interpolation, T::zero()..T::simd_two_pi(), |x| x.simd_sin())
    }

    pub fn cos(interpolation: Interp) -> Self {
        Self::from_fn(interpolation, T::zero()..T::simd_two_pi(), |x| x.simd_cos())
    }

    pub fn tan(interpolation: Interp) -> Self {
        Self::from_fn(interpolation, T::zero()..T::simd_two_pi(), |x| x.simd_tan())
    }
}
