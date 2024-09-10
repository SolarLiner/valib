use nalgebra::SimdPartialOrd;
use num_traits::{FromPrimitive, NumAssignOps, NumOps};
use numeric_literals::replace_float_literals;
use simba::simd::SimdValue;

use crate::util::simd_index_simd;
use crate::{Scalar, SimdCast};

pub trait SimdIndex:
    Copy + NumAssignOps + NumOps + SimdPartialOrd + SimdValue<Element = usize>
{
}

impl<T: Copy + NumAssignOps + NumOps + SimdPartialOrd + SimdValue<Element = usize>> SimdIndex
    for T
{
}

pub trait SimdInterpolatable: SimdCast<usize>
where
    <Self as SimdCast<usize>>::Output: SimdIndex,
{
    fn index_from_usize(value: usize) -> Self::Output {
        Self::Output::splat(value)
    }
}

impl<T: SimdCast<usize>> SimdInterpolatable for T where <T as SimdCast<usize>>::Output: SimdIndex {}

/// Interpolation trait. Interpolators implement function that connect discrete points into a continuous function.
/// Functions can have any number of taps, both in the forward and bacward directions. It's up to the called to provide
/// either actual existing points or extrapolated values when the indices would be out of bounds.
pub trait Interpolate<T, const N: usize> {
    /// Provide the indices needed for interpolating around the input index, where the index is the element that corresponds to t = 0.
    fn indices(index: usize) -> [usize; N];

    /// Interpolate a single point from the given taps (in the same order as the indices defined in [`Interpolate::indices`]).
    /// The t parameter is assumed to be in the 0..=1 range, and it's up to the caller to provide values in the valid range.
    fn interpolate(&self, t: T, taps: [T; N]) -> T;

    /// Interpolate a value from an entire slice, where the t parameter is the "floating index" into the slice
    /// (meaning 3.5 is halfway between index 3 and 4 on the given slice).
    #[profiling::function]
    fn interpolate_on_slice(&self, t: T, values: &[T]) -> T
    where
        T: Scalar + SimdInterpolatable,
        <T as SimdCast<usize>>::Output: SimdIndex,
    {
        let input_frac = t.simd_fract();
        let input_index = t.simd_floor().cast();
        let taps_ix: [_; N] = std::array::from_fn(|i| {
            let mut output = input_index;
            for j in 0..<T as SimdCast<usize>>::Output::LANES {
                output.replace(j, Self::indices(output.extract(j))[i]);
            }
            output
        });

        let zero = T::index_from_usize(0);
        let ix_max = T::index_from_usize(values.len() - 1);
        let taps = taps_ix.map(|tap| simd_index_simd(values, tap.simd_clamp(zero, ix_max)));
        self.interpolate(input_frac, taps)
    }

    /// Interpolate one slice into another, where the output slice ends up containing the same "range" of values as
    /// the input slice, but also automatically performs interpolation using this instance.
    #[profiling::function]
    fn interpolate_slice(&self, output: &mut [T], input: &[T])
    where
        T: Scalar + SimdInterpolatable,
        <T as SimdCast<usize>>::Output: SimdIndex,
    {
        let rate = input.len() as f64 / output.len() as f64;

        for (i, o) in output.iter_mut().enumerate() {
            let t = T::from_f64(rate * i as f64);
            *o = self.interpolate_on_slice(t, input);
        }
    }
}

/// Zero-hold interpolation, where the output is the input without additional computation.
#[derive(Debug, Copy, Clone)]
pub struct ZeroHold;

impl<T> Interpolate<T, 1> for ZeroHold {
    fn indices(index: usize) -> [usize; 1] {
        [index]
    }

    fn interpolate(&self, _: T, [x]: [T; 1]) -> T {
        x
    }
}

/// Nearest-neighbor interpolation, where the neighbor is chosen based on how close it is to the floating index.
#[derive(Debug, Copy, Clone)]
pub struct Nearest;

impl<T: SimdPartialOrd + FromPrimitive> Interpolate<T, 2> for Nearest {
    fn indices(index: usize) -> [usize; 2] {
        [index, index + 1]
    }

    fn interpolate(&self, t: T, [a, b]: [T; 2]) -> T {
        b.select(t.simd_gt(T::from_f64(0.5).unwrap()), a)
    }
}

/// Standard-issue linear interpolation algorithm.
#[derive(Debug, Copy, Clone)]
pub struct Linear;

impl<T: Scalar> Interpolate<T, 2> for Linear {
    fn indices(index: usize) -> [usize; 2] {
        [index, index + 1]
    }

    fn interpolate(&self, t: T, [a, b]: [T; 2]) -> T {
        a + (b - a) * t
    }
}

/// Sine interpolation, which is linear interpolation where the control input is first modulated by
/// a cosine function. Produes smoother results than bare linear interpolation because the interpolation
/// is smooth at the limits, as all derivatives are 0 there.
#[derive(Debug, Copy, Clone)]
pub struct MappedLinear<F>(pub F);

pub fn sine_interpolation<T: Scalar>() -> MappedLinear<impl Fn(T) -> T> {
    MappedLinear(|t| T::simd_cos(t * T::simd_pi()))
}

impl<T, F: Fn(T) -> T> Interpolate<T, 2> for MappedLinear<F>
where
    Linear: Interpolate<T, 2>,
{
    fn indices(index: usize) -> [usize; 2] {
        Linear::indices(index)
    }

    fn interpolate(&self, t: T, taps: [T; 2]) -> T {
        Linear.interpolate(self.0(t), taps)
    }
}

/// Cubic algorithm, smoother than [`Linear`], but needs 4 taps instead of 2.
#[derive(Debug, Copy, Clone)]
pub struct Cubic;

impl<T: Scalar> Interpolate<T, 4> for Cubic {
    fn indices(index: usize) -> [usize; 4] {
        [index.saturating_sub(1), index, index + 1, index + 2]
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn interpolate(&self, t: T, taps: [T; 4]) -> T {
        taps[1]
            + 0.5
                * t
                * (taps[2] - taps[0]
                    + t * (2.0 * taps[0] - 5.0 * taps[1] + 4.0 * taps[2] - taps[3]
                        + t * (3.0 * (taps[1] - taps[2]) + taps[3] - taps[0])))
    }
}

/// 4-tap cubic Hermite spline interpolation
#[derive(Debug, Copy, Clone)]
pub struct Hermite;

impl<T: Scalar> Interpolate<T, 4> for Hermite {
    fn indices(index: usize) -> [usize; 4] {
        [index.saturating_sub(1), index, index + 1, index + 2]
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn interpolate(&self, t: T, taps: [T; 4]) -> T {
        let c0 = taps[1];
        let c1 = 0.5 * (taps[2] - taps[0]);
        let c2 = taps[0] - 2.5 * taps[1] + 2.0 * taps[2] - 0.5 * taps[3];
        let c3 = 1.5 * (taps[1] - taps[2]) + 0.5 * (taps[3] - taps[0]);
        ((c3 * t + c2) * t + c1) * t + c0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Lanczos;

impl<T: Scalar> Interpolate<T, 7> for Lanczos {
    fn indices(index: usize) -> [usize; 7] {
        std::array::from_fn(|i| (index + i).saturating_sub(3))
    }

    fn interpolate(&self, t: T, taps: [T; 7]) -> T {
        let a = T::from_f64(3.0);
        let x = T::from_f64(4.0) + t;
        taps.iter()
            .copied()
            .enumerate()
            .map(|(i, s)| {
                let i = x.simd_floor() - a + T::one() + T::from_f64(i as _);
                s * Self::window::<T>(x - i)
            })
            .reduce(|a, b| a + b)
            .unwrap_or_else(T::zero)
    }
}

impl Lanczos {
    fn window<T: Scalar>(t: T) -> T {
        fn sinc<T: Scalar>(x: T) -> T {
            let nx = x * T::simd_pi();
            let y = nx.simd_sin() / nx;
            T::one().select(x.simd_eq(T::zero()), y)
        }

        sinc(t) * sinc(t / T::from_f64(3.0))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn test_interpolate<Interp, const TAPS: usize>(
        #[values(
            ZeroHold,
            Nearest,
            Linear,
            Cubic,
            Hermite,
            sine_interpolation(),
            Lanczos
        )]
        interp: Interp,
    ) where
        Interp: Interpolate<f64, TAPS>,
    {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        interp.interpolate_slice(&mut actual, &a);
        let name = format!(
            "test_interpolate_{}",
            std::any::type_name::<Interp>()
                .replace(['<', '>'], "__")
                .replace("::", "__")
        );
        insta::assert_csv_snapshot!(name, &actual as &[_], { "[]" => insta::rounded_redaction(6) });
    }
}
