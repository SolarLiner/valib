#![warn(missing_docs)]
#![feature(generic_const_exprs)]

use az::CastFrom;
use num_traits::Zero;
use simba::simd::{AutoSimd, Simd, SimdRealField, SimdValue};

use crate::util::{as_nested_arrays, as_nested_arrays_mut};
pub use simba::simd;

pub mod dsp;
pub mod math;
pub mod util;

pub trait Scalar: Copy + SimdRealField {
    fn from_f64(value: f64) -> Self;

    fn from_values(values: [Self::Element; <Self as SimdValue>::LANES]) -> Self
    where
        [Self::Element; <Self as SimdValue>::LANES]:;

    fn values(self) -> [Self::Element; <Self as SimdValue>::LANES]
    where
        [Self::Element; <Self as SimdValue>::LANES]:,
    {
        std::array::from_fn(|i| self.extract(i))
    }

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

pub trait SimdCast<E>: SimdValue {
    type Output: SimdValue<Element = E>;
    fn cast(self) -> Self::Output;
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

pub trait SimdFromSlice: Scalar {
    fn from_slice(data: &[Self::Element]) -> (&[Self], &[Self::Element]);
    fn from_slice_mut(data: &mut [Self::Element]) -> (&mut [Self], &mut [Self::Element]);
}

impl<T, const N: usize> SimdFromSlice for Simd<[T; N]>
where
    Self: Scalar,
{
    fn from_slice(data: &[Self::Element]) -> (&[Self], &[Self::Element]) {
        let (inner, remaining) = as_nested_arrays::<_, N>(data);
        // Satefy: Simd<N> is repr(transparent)
        let ret = unsafe { std::mem::transmute(inner) };
        (ret, remaining)
    }
    fn from_slice_mut(data: &mut [Self::Element]) -> (&mut [Self], &mut [Self::Element]) {
        let (inner, remaining) = as_nested_arrays_mut::<_, N>(data);
        // Satefy: Simd<N> is repr(transparent)
        let ret = unsafe { std::mem::transmute(inner) };
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
        is_cast_compatible::<simd::AutoF32x4, simd::AutoUsizex4>();
        is_cast_compatible::<simd::AutoF64x4, simd::AutoUsizex4>();
    }
}
