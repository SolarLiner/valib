//! Types providing Lookup Tables

use std::ops::Range;

use crate::{Scalar, SimdCast};
use numeric_literals::replace_float_literals;

use super::interpolation::{Interpolate, SimdIndex, SimdInterpolatable};

/// Lookup table storing a static array of `N` points of type `T`.
#[derive(Debug, Clone)]
pub struct Lut<T, const N: usize> {
    array: [T; N],
    range: Range<T>,
}

impl<T, const N: usize> Lut<T, N> {
    /// Construct a new lookup table with the given data and range of values.
    ///
    /// # Arguments
    ///
    /// * `array`:
    /// * `range`:
    ///
    /// returns: Lut<T, { N }>
    pub const fn new(array: [T; N], range: Range<T>) -> Self {
        Self { array, range }
    }

    /// Get the value at the given index, performing the given interpolation.
    ///
    /// # Arguments
    ///
    /// * `interp`: Interpolation method to use
    /// * `index`: Input value for the LUT
    ///
    /// returns: T
    #[profiling::function]
    pub fn get<Interp, const I: usize>(&self, interp: &Interp, index: T) -> T
    where
        T: Scalar + SimdInterpolatable,
        <T as SimdCast<usize>>::Output: SimdIndex,
        Interp: Interpolate<T, I>,
    {
        let normalized = (index - self.range.start) / (self.range.end - self.range.start);
        let array_index = normalized * T::from_f64(N as f64);
        interp.interpolate_on_slice(array_index, &self.array)
    }
}

impl<T: Scalar, const N: usize> Lut<T, N> {
    /// Construct a new lookup table from the provided range and function. This computes the
    /// function at each point in the LUT once, and stores it.
    ///
    /// # Arguments
    ///
    /// * `range`: Input range of the LUT
    /// * `f`: Function to map
    ///
    /// returns: Lut<T, { N }>
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

    /// Generate a LUT for the tanh function.
    #[replace_float_literals(T::from_f64(literal))]
    pub fn tanh() -> Self {
        Self::from_fn(-19.0..19.0, |x| x.simd_tanh())
    }

    /// Generate a LUT for the atanh function.
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
        insta::assert_csv_snapshot!(tanh.array.as_ref(), { "[]" => insta::rounded_redaction(3) });
    }

    #[test]
    fn test_lut_atanh() {
        let atanh = Lut::<f64, 512>::atanh();
        insta::assert_csv_snapshot!(atanh.array.as_ref(), { "[]" => insta::rounded_redaction(3) });
    }
}
