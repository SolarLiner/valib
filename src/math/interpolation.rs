use crate::{util::simd_index_simd, SimdCast, SimdValue};

use nalgebra::SimdPartialOrd;
use num_traits::{FromPrimitive, Num};
use numeric_literals::replace_float_literals;

use crate::Scalar;

pub trait Interpolate<T, const N: usize> {
    fn rel_indices() -> [isize; N];
    fn interpolate(t: T, taps: [T; N]) -> T;

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

pub struct Nearest;

impl<T: SimdPartialOrd + FromPrimitive> Interpolate<T, 2> for Nearest {
    fn rel_indices() -> [isize; 2] {
        [0, 1]
    }

    fn interpolate(t: T, [a, b]: [T; 2]) -> T {
        b.select(t.simd_gt(T::from_f64(0.5).unwrap()), a)
    }
}

pub struct Linear;

impl<T: Scalar> Interpolate<T, 2> for Linear {
    fn rel_indices() -> [isize; 2] {
        [0, 1]
    }

    fn interpolate(t: T, [a, b]: [T; 2]) -> T {
        a + (b - a) * t
    }
}

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
        let expected = [0.0, 0.203125, 0.5, 0.796875, 1.0, 1.0703125, 1.0625, 1.0234375, 1.0, 1.0, 1.0, 1.0];
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