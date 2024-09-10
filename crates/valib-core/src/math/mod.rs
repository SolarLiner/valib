use nalgebra::{Complex, Dim, OVector};
use numeric_literals::replace_float_literals;
use simba::simd::{SimdBool, SimdComplexField};

use crate::Scalar;

pub mod interpolation;
pub mod lut;
pub mod nr;
#[cfg(feature = "math-polynom")]
pub mod polynom;

#[replace_float_literals(Complex::from(T::from_f64(literal)))]
pub fn freq_to_z<T: Scalar>(samplerate: T, f: T) -> Complex<T>
where
    Complex<T>: SimdComplexField,
{
    let jw = Complex::new(T::zero(), T::simd_two_pi() * f / samplerate);
    jw.simd_exp()
}

#[inline]
fn rms<T: SimdComplexField, D: Dim>(value: &OVector<T, D>) -> T
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, D>,
{
    value.map(|v| v.simd_powi(2)).sum().simd_sqrt()
}

#[replace_float_literals(T::from_f64(literal))]
pub fn bilinear_prewarming<T: Scalar>(samplerate: T, wc: T) -> T {
    2.0 * samplerate * T::simd_tan(wc / (2.0 * samplerate))
}

pub fn bilinear_prewarming_bounded<T: Scalar>(samplerate: T, wc: T) -> T {
    let wmax = samplerate * T::simd_frac_pi_2();
    wc.simd_lt(wmax).if_else(
        || bilinear_prewarming(samplerate, wc),
        || (wc * bilinear_prewarming(samplerate, wmax) / wmax),
    )
}

/// Exponential smooth minimum
#[replace_float_literals(T::from_f64(literal))]
#[inline]
pub fn smooth_min<T: Scalar>(t: T, a: T, b: T) -> T {
    let r = (-a / t).simd_exp2() + (-b / t).simd_exp2();
    -t * r.simd_log2()
}

/// Exponential smooth maximum
#[inline]
pub fn smooth_max<T: Scalar>(t: T, a: T, b: T) -> T {
    -smooth_min(t, -a, -b)
}

/// Exponential smooth clamping
#[inline]
pub fn smooth_clamp<T: Scalar>(t: T, x: T, min: T, max: T) -> T {
    smooth_max(t, min, smooth_min(t, x, max))
}
