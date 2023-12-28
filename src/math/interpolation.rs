use crate::{util::simd_index_simd, SimdCast, SimdValue};

use nalgebra::SimdPartialOrd;
use num_traits::{FromPrimitive, Num};
use numeric_literals::replace_float_literals;

use crate::Scalar;

use super::lut::Lut;

/// Interpolation trait. Interpolators implement function that connect discrete points into a continuous function.
/// Functions can have any number of taps, both in the forward and bacward directions. It's up to the called to provide
/// either actual existing points or extrapolated values when the indices would be out of bounds.
pub trait Interpolate<T, const N: usize> {
    /// Provide the relative indices needed to compute the interpolation
    fn rel_indices() -> [isize; N];

    /// Interpolate a single point from the given taps (in the same order as the indices defined in [`rel_indices`]).
    /// The t parameter is assumed to be in the 0..=1 range, and it's up to the caller to provide values in the valid range.
    fn interpolate(t: T, taps: [T; N]) -> T;

    /// Interpolate a value from an entire slice, where the t parameter is the "floating index" into the slice
    /// (meaning 3.5 is halfway between index 3 and 4 on the given slice).
    fn interpolate_on_slice(t: T, values: &[T]) -> T
    where
        T: Scalar + SimdCast<isize>,
        <T as SimdCast<isize>>::Output: Copy + Num + SimdPartialOrd,
    {
        let ix_max = <T as SimdCast<isize>>::Output::splat(values.len() as isize - 1);
        // let rate = input.len() as f64 / output.len() as f64;
        let taps_ix = Self::rel_indices();

        let zero = <T as SimdCast<isize>>::Output::splat(0);
        let input_frac = t.simd_fract();
        let input_index = t.simd_floor().cast();
        let taps = taps_ix
            .map(|tap| <T as SimdCast<isize>>::Output::splat(tap) + input_index)
            .map(|tap| simd_index_simd(values, tap.simd_clamp(zero, ix_max)));
        Self::interpolate(input_frac, taps)
    }

    /// Interpolate one slice into another, where the output slice ends up containing the same "range" of values as
    /// the input slice, but also automatically performs interpolation using this instance.
    fn interpolate_slice(output: &mut [T], input: &[T])
    where
        T: Scalar + SimdCast<isize>,
        <T as SimdCast<isize>>::Output: Copy + Num + SimdPartialOrd,
    {
        let rate = input.len() as f64 / output.len() as f64;

        for (i, o) in output.iter_mut().enumerate() {
            let t = T::from_f64(rate * i as f64);
            *o = Self::interpolate_on_slice(t, input);
        }
    }
}

pub struct ZeroHold;

impl<T> Interpolate<T, 1> for ZeroHold {
    fn rel_indices() -> [isize; 1] {
        [0]
    }

    fn interpolate(_: T, [x]: [T; 1]) -> T {
        x
    }
}

/// Nearest-neighbor interpolation, where the neighbor is chosen based on how close it is to the floating index.
pub struct Nearest;

impl<T: SimdPartialOrd + FromPrimitive> Interpolate<T, 2> for Nearest {
    fn rel_indices() -> [isize; 2] {
        [0, 1]
    }

    fn interpolate(t: T, [a, b]: [T; 2]) -> T {
        b.select(t.simd_gt(T::from_f64(0.5).unwrap()), a)
    }
}

/// Standard-issue linear interpolation algorithm.
pub struct Linear;

impl<T: Scalar> Interpolate<T, 2> for Linear {
    fn rel_indices() -> [isize; 2] {
        [0, 1]
    }

    fn interpolate(t: T, [a, b]: [T; 2]) -> T {
        a + (b - a) * t
    }
}

/// Cubic algorithm, smoother than [`Linear`], but needs 4 taps instead of 2.
pub struct Cubic;

impl<T: Scalar> Interpolate<T, 4> for Cubic {
    fn rel_indices() -> [isize; 4] {
        [-1, 0, 1, 2]
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn interpolate(t: T, taps: [T; 4]) -> T {
        taps[1]
            + 0.5
                * t
                * (taps[2] - taps[0]
                    + t * (2.0 * taps[0] - 5.0 * taps[1] + 4.0 * taps[2] - taps[3]
                        + t * (3.0 * (taps[1] - taps[2]) + taps[3] - taps[0])))
    }
}

/// 4-tap cubic Hermite spline interpolation
pub struct Hermite;

impl<T: Scalar> Interpolate<T, 4> for Hermite {
    fn rel_indices() -> [isize; 4] {
        [-1, 0, 1, 2]
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn interpolate(t: T, taps: [T; 4]) -> T {
        let c0 = taps[1];
        let c1 = 0.5 * (taps[2] - taps[0]);
        let c2 = taps[0] - 2.5 * taps[1] + 2.0 * taps[2] - 0.5 * taps[3];
        let c3 = 1.5 * (taps[1] - taps[2]) + 0.5 * (taps[3] - taps[0]);
        ((c3 * t + c2) * t + c1) * t + c0
    }
}

#[derive(Debug, Clone)]
pub struct Sine;

impl<T: Scalar> Interpolate<T, 2> for Sine {
    fn rel_indices() -> [isize; 2] {
        [0, 1]
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn interpolate(t: T, taps: [T; 2]) -> T {
        let fac = T::simd_cos(t * T::simd_pi()) * 0.5 + 0.5;
        Linear::interpolate(fac, taps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_nearest() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
        Nearest::interpolate_slice(&mut actual, &a);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_interpolate_linear() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1.];
        Linear::interpolate_slice(&mut actual, &a);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_interpolate_cubic() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [
            0.0, 0.203125, 0.5, 0.796875, 1.0, 1.0703125, 1.0625, 1.0234375, 1.0, 1.0, 1.0, 1.0,
        ];
        Cubic::interpolate_slice(&mut actual, &a);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_interpolate_hermite() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [
            0.0, 0.203125, 0.5, 0.796875, 1.0, 1.0703125, 1.0625, 1.0234375, 1.0, 1.0, 1.0, 1.0,
        ];
        Hermite::interpolate_slice(&mut actual, &a);
        assert_eq!(actual, expected);
    }
}
