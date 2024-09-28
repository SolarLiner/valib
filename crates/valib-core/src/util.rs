//! Utilities for all of `valib`.

use crate::Scalar;
use nalgebra::{
    Dim, Matrix, MatrixView, MatrixViewMut, Storage, StorageMut, Vector, VectorView, VectorViewMut,
    ViewStorage, ViewStorageMut,
};
use num_traits::{AsPrimitive, Float, Zero};
use numeric_literals::replace_float_literals;
use simba::simd::SimdValue;

/// Transmutes a slice into a slice of static arrays, putting the remainder of the slice not fitting
/// as a separate slice.
///
/// # Arguments
///
/// * `data`: Slice to transmute
///
/// returns: (&[[T; N]], &[T])
///
/// # Examples
///
/// ```
/// use valib_core::util::as_nested_arrays;
/// let array = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// let (arrays, remainder) = as_nested_arrays(&array);
/// assert_eq!(&[[0,1,2],[3,4,5]], arrays);
/// assert_eq!(&[6,7], remainder);
/// ```
pub fn as_nested_arrays<T, const N: usize>(data: &[T]) -> (&[[T; N]], &[T]) {
    let rem = data.len() % N;
    let last = data.len() - rem;
    let (complete, remaining) = data.split_at(last);
    let outer_len = complete.len() / N;

    // Safety: Static arrays of N elements have the same representation as N contiguous elements
    let result = unsafe { std::slice::from_raw_parts(complete.as_ptr() as *const _, outer_len) };
    (result, remaining)
}

/// Transmutes a mutable slice into a slice of static arrays, putting the remainder of the slice not
/// fitting as a separate slice.
///
/// # Arguments
///
/// * `data`: Slice to transmute
///
/// returns: (&mut [[T; N]], &mut [T])
///
/// # Examples
///
/// ```
/// use valib_core::util::as_nested_arrays_mut;
/// let mut array = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// let (arrays, remainder) = as_nested_arrays_mut(&mut array);
/// assert_eq!(&[[0,1,2],[3,4,5]], arrays);
/// assert_eq!(&[6,7], remainder);
///
/// arrays[1][0] = 10;
/// assert_eq!(vec![0, 1, 2, 10, 4, 5, 6, 7], array);
/// ```
pub fn as_nested_arrays_mut<T, const N: usize>(data: &mut [T]) -> (&mut [[T; N]], &mut [T]) {
    let rem = data.len() % N;
    let last = data.len() - rem;
    let (complete, remaining) = data.split_at_mut(last);
    let outer_len = complete.len() / N;

    // Safety: Static arrays of N elements have the same representation as N contiguous elements
    let result =
        unsafe { std::slice::from_raw_parts_mut(complete.as_mut_ptr() as *mut _, outer_len) };
    (result, remaining)
}

/// Index a slice of scalars with a SIMD index, returning a SIMD with the corresponding scalar values
/// on each lane.
///
/// # Arguments
///
/// * `values`: Slice to index
/// * `index`: Index SIMD value
///
/// returns: Simd
///
/// # Examples
///
/// ```
/// use simba::simd::{AutoF32x4, AutoUsizex4};
/// use valib_core::util::simd_index_scalar;
/// let data = [0.0, 0.1, 0.2, 0.3];
/// let index = AutoUsizex4::new(0, 1, 2, 3);
/// let  ret = simd_index_scalar::<AutoF32x4, _>(&data, index);
/// assert_eq!(AutoF32x4::new(0.0, 0.1, 0.2, 0.3), ret);
/// ```
pub fn simd_index_scalar<Simd: Zero + SimdValue, Index: SimdValue<Element = usize>>(
    values: &[Simd::Element],
    index: Index,
) -> Simd
where
    Simd::Element: Copy,
{
    let mut ret = Simd::zero();
    for i in 0..Simd::LANES {
        let ix = index.extract(i);
        ret.replace(i, values[ix]);
    }
    ret
}

/// Index a slice of SIMD values with a SIMD index, returning a new SIMD values corresponding to
/// each lane of the values for each index.
///
/// # Arguments
///
/// * `values`: Slice to index
/// * `index`: Index SIMD value
///
/// returns: Simd
///
/// # Examples
///
/// ```
/// use simba::simd::{AutoF32x2, AutoUsizex2};
/// use valib_core::util::simd_index_simd;
/// let data = [AutoF32x2::new(0.0, 0.1), AutoF32x2::new(1.0, 1.1)];
/// let index = AutoUsizex2::new(1, 0);
/// let ret = simd_index_simd(&data, index);
/// assert_eq!(AutoF32x2::new(1.0, 0.1), ret);
/// ```
pub fn simd_index_simd<Simd: Zero + SimdValue, Index: SimdValue>(
    values: &[Simd],
    index: Index,
) -> Simd
where
    <Index as SimdValue>::Element: AsPrimitive<usize>,
{
    let mut ret = Simd::zero();
    for i in 0..Index::LANES {
        let ix = index.extract(i).as_();
        ret.replace(i, values[ix].extract(i));
    }
    ret
}

/// Generic method for checking if a SIMD float is finie or not. Returns a SIMD mask where the lanes
/// are finite.
///
/// # Arguments
///
/// * `value`: Value to check
///
/// returns: <Simd as SimdValue>::SimdBool
///
/// # Examples
///
/// ```
/// use simba::simd::{AutoF32x4, AutoBoolx4};
/// use valib_core::util::simd_is_finite;
/// let value = AutoF32x4::new(0.0, f32::INFINITY, f32::NAN, f32::EPSILON);
/// let ret = simd_is_finite(value);
/// assert_eq!(AutoBoolx4::new(true, false, false, true), ret);
/// ```
pub fn simd_is_finite<
    Simd: SimdValue<Element: Float, SimdBool: Default + SimdValue<Element = bool>>,
>(
    value: Simd,
) -> Simd::SimdBool {
    let mut mask = Simd::SimdBool::default();
    for i in 0..Simd::LANES {
        mask.replace(i, value.extract(i).is_finite())
    }
    mask
}

/// Shortcut function to perform linear interpolation. Uses the interpolation module.
pub fn lerp<T: Scalar>(t: T, a: T, b: T) -> T {
    use crate::math::interpolation::{Interpolate, Linear};
    Linear.interpolate(t, [a, b])
}

/// Computes the frequency of a MIDI note number, assuming 12TET and A4 = 440 Hz
///
/// # Arguments
///
/// * `midi_note`: MIDI note number
///
/// returns: T
#[replace_float_literals(T::from_f64(literal))]
pub fn midi_to_freq<T: Scalar>(midi_note: u8) -> T {
    440.0 * semitone_to_ratio(T::from_f64(midi_note as _) - 69.0)
}

/// Compute the ratio corresponding to the given semitone change, such that multiplying a frequency
/// by this value changes it by the given semitones.
///
/// # Arguments
///
/// * `semi`: Semitone change
///
/// returns: T
#[replace_float_literals(T::from_f64(literal))]
pub fn semitone_to_ratio<T: Scalar>(semi: T) -> T {
    2.0.simd_powf(semi / 12.0)
}

pub fn matrix_view<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>>(
    m: &Matrix<T, R, C, S>,
) -> MatrixView<T, R, C, S::RStride, S::CStride> {
    MatrixView::from_data(unsafe {
        let shape = m.shape_generic();
        ViewStorage::new_unchecked(&m.data, (0, 0), shape)
    })
}

pub fn matrix_view_mut<T: Scalar, R: Dim, C: Dim, S: StorageMut<T, R, C>>(
    m: &mut Matrix<T, R, C, S>,
) -> MatrixViewMut<T, R, C, S::RStride, S::CStride> {
    MatrixViewMut::from_data(unsafe {
        let shape = m.shape_generic();
        ViewStorageMut::new_unchecked(&mut m.data, (0, 0), shape)
    })
}

pub fn vector_view<T: Scalar, D: Dim, S: Storage<T, D>>(
    v: &Vector<T, D, S>,
) -> VectorView<T, D, S::RStride, S::CStride> {
    VectorView::from_data(unsafe {
        let shape = v.shape_generic();
        ViewStorage::new_unchecked(&v.data, (0, 0), shape)
    })
}

pub fn vector_view_mut<T: Scalar, D: Dim, S: StorageMut<T, D>>(
    v: &mut Vector<T, D, S>,
) -> VectorViewMut<T, D, S::RStride, S::CStride> {
    VectorViewMut::from_data(unsafe {
        let shape = v.shape_generic();
        ViewStorageMut::new_unchecked(&mut v.data, (0, 0), shape)
    })
}

#[cfg(feature = "test-utils")]
pub mod tests;
