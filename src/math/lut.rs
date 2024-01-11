use std::ops::Range;

use num_traits::{FromPrimitive, Num};
use numeric_literals::replace_float_literals;

use crate::{simd::SimdPartialOrd, Scalar, SimdCast};

use super::interpolation::Interpolate;

#[derive(Debug, Clone)]
pub struct Lut<T, const N: usize> {
    array: [T; N],
    range: Range<T>,
}

impl<T, const N: usize> Lut<T, N> {
    pub const fn new(array: [T; N], range: Range<T>) -> Self {
        Self { array, range }
    }

    pub fn get<Interp, const I: usize>(&self, index: T) -> T
    where
        T: Scalar + SimdCast<isize>,
        Interp: Interpolate<T, I>,
        <T as SimdCast<isize>>::Output: Copy + Num + SimdPartialOrd,
    {
        let normalized = (index - self.range.start) / (self.range.end - self.range.start);
        let array_index = normalized * T::from_f64(N as f64);
        Interp::interpolate_on_slice(array_index, &self.array)
    }
}

impl<T: Scalar, const N: usize> Lut<T, N> {
    pub fn from_fn(range: Range<T>, f: impl Fn(T) -> T) -> Self {
        let start = range.start;
        let r = range.end - range.start;
        let rsize = T::from_f64(N as f64).simd_recip();
        let array = std::array::from_fn(|i| {
            let n = T::from_f64(i as f64) * rsize;
            f(start + n * r)
        });
        Self::new(array, range)
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn tanh() -> Self {
        Self::from_fn(-19.0..19.0, |x| x.simd_tanh())
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn atanh() -> Self {
        Self::from_fn(-1.0..1.0, |x| x.simd_atanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_tanh() {
        let tanh = Lut::<f64, 512>::tanh();
        insta::assert_debug_snapshot!(tanh);
    }

    #[test]
    fn test_lut_atanh() {
        let atanh = Lut::<f64, 512>::atanh();
        insta::assert_debug_snapshot!(atanh);
    }
}
