use num_traits::{AsPrimitive, FromPrimitive, Num, One, Zero};
use numeric_literals::replace_float_literals;
use simba::simd::{SimdPartialOrd, SimdValue};

use crate::{Scalar, SimdCast};

pub fn simd_index_scalar<Simd: Zero + SimdValue, Index: SimdValue<Element = usize>>(
    values: &[Simd::Element],
    index: Index,
) -> Simd
where
    Simd::Element: Copy,
{
    let mut ret = Simd::zero();
    for i in 0..Simd::lanes() {
        let ix = index.extract(i);
        ret.replace(i, values[ix]);
    }
    ret
}

pub fn simd_index_simd<Simd: Zero + SimdValue, Index: SimdValue>(
    values: &[Simd],
    index: Index,
) -> Simd
where
    <Index as SimdValue>::Element: AsPrimitive<usize>,
{
    let mut ret = Simd::zero();
    for i in 0..Index::lanes() {
        let ix = index.extract(i).as_();
        ret.replace(i, values[ix].extract(i));
    }
    ret
}

#[replace_float_literals(T::from_f64(literal))]
#[deprecated = "Use math::interpolators"]
pub fn lerp_block<T: Scalar + SimdCast<usize>>(out: &mut [T], inp: &[T])
where
    <T as SimdCast<usize>>::Output: Copy + Num + FromPrimitive + SimdPartialOrd,
{
    let ix_max = <T as SimdCast<usize>>::Output::from_usize(inp.len() - 1).unwrap();
    let rate = T::from_f64(inp.len() as f64) / T::from_f64(out.len() as f64);

    for (i, y) in out.iter_mut().enumerate() {
        let j = T::from_f64(i as f64) * rate;
        let f = j.simd_fract();
        let j = j.simd_floor().cast();
        let jp1 = (j + <T as SimdCast<usize>>::Output::one()).simd_min(ix_max);
        let a = simd_index_simd(inp, j);
        let b = simd_index_simd(inp, jp1);
        *y = lerp(f, a, b);
    }
}

pub fn lerp<T: Scalar>(t: T, a: T, b: T) -> T {
    use crate::math::interpolation::{Interpolate, Linear};
    Linear.interpolate(t, [a, b])
}

#[replace_float_literals(T::from_f64(literal))]
pub fn midi_to_freq<T: Scalar>(midi_note: u8) -> T {
    440.0 * semitone_to_ratio(T::from_f64(midi_note as _) - 69.0)
}

#[replace_float_literals(T::from_f64(literal))]
pub fn semitone_to_ratio<T: Scalar>(semi: T) -> T {
    2.0.simd_powf(semi / 12.0)
}

#[cfg(test)]
mod tests {
    use crate::math::interpolation::{Interpolate, Linear};

    #[test]
    fn interp_block() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1.];
        Linear.interpolate_slice(&mut actual, &a);
        assert_eq!(actual, expected);
    }
}
