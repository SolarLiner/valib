//! # `valib_core`
//!
//! Provides the basic definitions for all of `valib`. Contains basic DSP definitions and useful math
//! constructs.
#![warn(missing_docs)]
#![feature(generic_const_exprs)]

use az::CastFrom;
use num_traits::Zero;
use simba::simd::{AutoSimd, Simd, SimdRealField, SimdValue};

use crate::util::{as_nested_arrays, as_nested_arrays_mut};
pub use simba::simd;

pub mod benchmarking;
pub mod dsp;
pub mod math;
pub mod util;

/// Scalar trait. All of `valib` uses this trait as bound for scalar values.
///
/// A scalar is defined here to mean the value which is used as an audio sample. It very often is `f32`,
/// but can also be any SIMD type, where the values within the SIMD (also called scalars but for a
/// different reason)
pub trait Scalar: Copy + SimdRealField {
    /// Create a new [`Scalar`] from a single `f64` value. The resulting type, if it is a SIMD with
    /// multiple lanes, should have all lanes being this value.
    fn from_f64(value: f64) -> Self;

    /// Create a new [`Scalar`] containing the values passed in the array.
    fn from_values(values: [Self::Element; <Self as SimdValue>::LANES]) -> Self
    where
        [Self::Element; <Self as SimdValue>::LANES]:;

    /// Return the array of elements contained in this [`Scalar`].
    fn values(self) -> [Self::Element; <Self as SimdValue>::LANES]
    where
        [Self::Element; <Self as SimdValue>::LANES]:,
    {
        std::array::from_fn(|i| self.extract(i))
    }

    /// Return an iterator of the elements contained in this scalar.
    fn into_iter(self) -> impl ExactSizeIterator<Item = Self::Element> {
        (0..Self::LANES).map(move |i| self.extract(i))
    }
}

impl<T: Copy + SimdRealField> Scalar for T
where
    T::Element: Copy,
{
    fn from_f64(value: f64) -> Self {
        Self::from_subset(&value)
    }

    #[allow(clippy::needless_range_loop)]
    fn from_values(values: [Self::Element; <Self as SimdValue>::LANES]) -> Self
    where
        [Self::Element; <Self as SimdValue>::LANES]:,
    {
        let mut ret = Self::splat(values[0]);
        for i in 1..Self::LANES {
            unsafe {
                ret.replace_unchecked(i, values[i]);
            }
        }
        ret
    }
}

/// Trait for SIMD values which can be cast.
pub trait SimdCast<E>: SimdValue {
    /// Output type. This should be an SIMD value containing the same number of lanes as the input.
    type Output: SimdValue<Element = E>;

    /// Perform the cast.
    fn cast(self) -> Self::Output;
}

/// Shortcut method for casing a SIMD value into another one.
pub fn simd_cast<E, In: SimdCast<E>>(value: In) -> In::Output {
    value.cast()
}

impl<E1, E2, const N: usize> SimdCast<E2> for AutoSimd<[E1; N]>
where
    Self: SimdValue<Element = E1>,
    AutoSimd<[E2; N]>: SimdValue<Element = E2>,
    E2: CastFrom<E1>,
{
    type Output = AutoSimd<[E2; N]>;

    fn cast(self) -> Self::Output {
        let mut ret: AutoSimd<[E2; N]> = unsafe { std::mem::zeroed() };
        for i in 0..N {
            ret.replace(i, E2::cast_from(self.extract(i)));
        }
        ret
    }
}

macro_rules! impl_simdcast_primitives {
    ($ty:ty) => {
        impl<E2: SimdValue<Element = E2>> SimdCast<E2> for $ty
        where
            E2: CastFrom<$ty>,
        {
            type Output = E2;
            fn cast(self) -> Self::Output {
                E2::cast_from(self)
            }
        }
    };
}

impl_simdcast_primitives!(f32);
impl_simdcast_primitives!(f64);
impl_simdcast_primitives!(u8);
impl_simdcast_primitives!(u16);
impl_simdcast_primitives!(u32);
impl_simdcast_primitives!(u64);
impl_simdcast_primitives!(u128);
impl_simdcast_primitives!(i8);
impl_simdcast_primitives!(i16);
impl_simdcast_primitives!(i32);
impl_simdcast_primitives!(i64);
impl_simdcast_primitives!(i128);

macro_rules! impl_simdcast_wide {
    ($name:ty : [$prim:ty; $lanes:literal]) => {
        impl<E2> SimdCast<E2> for $name
        where
            E2: CastFrom<$prim>,
            simba::simd::AutoSimd<[E2; $lanes]>: Zero + SimdValue<Element = E2>,
        {
            type Output = simba::simd::AutoSimd<[E2; $lanes]>;

            fn cast(self) -> Self::Output {
                let mut ret = <Self::Output as Zero>::zero();
                for i in 0..$lanes {
                    ret.replace(i, E2::cast_from(self.extract(i)));
                }
                ret
            }
        }
    };
}

impl_simdcast_wide!(simd::WideF32x4 : [f32; 4]);
impl_simdcast_wide!(simd::WideF32x8 : [f32; 8]);
impl_simdcast_wide!(simd::WideF64x4 : [f64; 4]);

/// Trait for SIMD values which have a transparent repr with arrays, and as such can be directly
/// transmuted from them.
///
/// # Safety
///
/// This trait should **only** be implemented on types which are `#[repr(transparent)]` to a
/// `[Self::Element; Self::LANES]` array.
pub unsafe trait SimdFromSlice: Scalar {
    /// Transmutes a slice into a slice of [`Self`].
    fn from_slice(data: &[Self::Element]) -> (&[Self], &[Self::Element]);

    /// Transmutes a slice into a mut slice of [`Self`].
    fn from_slice_mut(data: &mut [Self::Element]) -> (&mut [Self], &mut [Self::Element]);
}

unsafe impl<T, const N: usize> SimdFromSlice for Simd<[T; N]>
where
    Self: Scalar<Element = T>,
{
    fn from_slice(data: &[Self::Element]) -> (&[Self], &[Self::Element]) {
        let (inner, remaining) = as_nested_arrays::<_, N>(data);
        // Satefy: Simd<N> is repr(transparent)
        let ret = unsafe { std::mem::transmute::<&[[T; N]], &[Simd<[T; N]>]>(inner) };
        (ret, remaining)
    }
    fn from_slice_mut(data: &mut [Self::Element]) -> (&mut [Self], &mut [Self::Element]) {
        let (inner, remaining) = as_nested_arrays_mut::<_, N>(data);
        // Satefy: Simd<N> is repr(transparent)
        let ret = unsafe { std::mem::transmute::<&mut [[T; N]], &mut [Simd<[T; N]>]>(inner) };
        (ret, remaining)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn is_compatible<T: Scalar>() {}
    const fn is_cast_compatible<From, To>()
    where
        From: SimdCast<To>,
    {
    }

    #[test]
    fn test_type_compatibility() {
        is_compatible::<f32>();
        is_compatible::<f64>();
        is_compatible::<simd::AutoF32x2>();
        is_compatible::<simd::AutoF32x4>();
        is_compatible::<simd::AutoF64x2>();
        is_compatible::<simd::AutoF64x4>();

        is_compatible::<simd::WideF32x4>();
        is_compatible::<simd::WideF64x4>();

        is_compatible::<simd::f32x2>();
        is_compatible::<simd::f32x4>();
        is_compatible::<simd::f64x2>();
        is_compatible::<simd::f64x4>();

        is_cast_compatible::<f32, usize>();
        is_cast_compatible::<f64, usize>();
        is_cast_compatible::<simd::AutoF32x4, usize>();
        is_cast_compatible::<simd::AutoF64x4, usize>();
    }
}
