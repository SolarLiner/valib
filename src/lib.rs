use az::CastFrom;
use num_traits::Zero;
use simba::simd::{AutoSimd, SimdRealField, SimdValue};

pub use simba::simd;

pub mod filters;
pub mod dsp;
pub mod math;
pub mod oscillators;
pub mod oversample;
pub mod saturators;
pub mod util;
#[cfg(feature = "unstable-wdf")]
pub mod wdf;
pub mod fir;
pub mod voice;

pub trait Scalar: Copy + SimdRealField {
    fn from_f64(value: f64) -> Self;

    fn values<const N: usize>(self) -> [Self::Element; N] {
        assert_eq!(N, Self::lanes());
        std::array::from_fn(|i| self.extract(i))
    }

    fn into_iter(self) -> impl ExactSizeIterator<Item=Self::Element> {
        (0..Self::lanes()).map(move |i| self.extract(i))
    }
}

impl<T: Copy + SimdRealField> Scalar for T {
    fn from_f64(value: f64) -> Self {
        Self::from_subset(&value)
    }
}

pub trait SimdCast<E>: SimdValue {
    type Output: SimdValue<Element = E>;
    fn cast(self) -> Self::Output;
}

impl<E1, E2, const N: usize> SimdCast<E2> for simba::simd::AutoSimd<[E1; N]>
where
    Self: SimdValue<Element=E1>,
    simba::simd::AutoSimd<[E2; N]>: SimdValue<Element = E2>,
    E2: CastFrom<E1>,
{
    type Output = simba::simd::AutoSimd<[E2; N]>;

    fn cast(self) -> Self::Output {
        assert_eq!(Self::Output::lanes(), N);
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

impl_simdcast_wide!(simba::simd::WideF32x4 : [f32; 4]);
impl_simdcast_wide!(simba::simd::WideF32x8 : [f32; 8]);
impl_simdcast_wide!(simba::simd::WideF64x4 : [f64; 4]);

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
        is_compatible::<simba::simd::AutoF32x2>();
        is_compatible::<simba::simd::AutoF32x4>();
        is_compatible::<simba::simd::AutoF64x2>();
        is_compatible::<simba::simd::AutoF64x4>();
        is_compatible::<simba::simd::WideF32x4>();
        is_compatible::<simba::simd::WideF64x4>();

        is_cast_compatible::<f32, usize>();
        is_cast_compatible::<f64, usize>();
        is_cast_compatible::<simba::simd::AutoF32x4, usize>();
        is_cast_compatible::<simba::simd::AutoF64x4, usize>();
    }
}
