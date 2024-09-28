//! # Math
//!
//! Functions implementing mathematical constructs, as used by the rest of `valib`.

use nalgebra::{Complex, Dim, VectorView};
use numeric_literals::replace_float_literals;
use simba::simd::{SimdBool, SimdComplexField};

use crate::Scalar;

pub mod interpolation;
pub mod lut;
pub mod nr;
#[cfg(feature = "math-polynom")]
pub mod polynom;

/// Return the complex number in the z-plane corresponding to the frequency `f` at sample rate
/// `samplerate`.
///
/// # Arguments
///
/// * `samplerate`: Sample rate of the z-plane
/// * `f`: Frequency in Hz
///
/// returns: Complex<T>
#[replace_float_literals(Complex::from(T::from_f64(literal)))]
pub fn freq_to_z<T: Scalar>(samplerate: T, f: T) -> Complex<T>
where
    Complex<T>: SimdComplexField,
{
    let jw = Complex::new(T::zero(), T::simd_two_pi() * f / samplerate);
    jw.simd_exp()
}

#[inline]
fn rms<T: SimdComplexField, D: Dim>(value: VectorView<T, D, impl Dim, impl Dim>) -> T
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    value.map(|v| v.simd_powi(2)).sum().simd_sqrt()
}

/// Returns the pre-warped pulsation (radian frequency) such that the frequency of a bilinear
/// discrete process matches the frequency of its analog counterpart.
///
/// # Arguments
///
/// `samplerate`: Sample rate of the discrete process
/// `wc`: Radian frequency (in Hz/rad) input
///
///
/// returns: `T`
#[replace_float_literals(T::from_f64(literal))]
pub fn bilinear_prewarming<T: Scalar>(samplerate: T, wc: T) -> T {
    2.0 * samplerate * T::simd_tan(wc / (2.0 * samplerate))
}

/// Returns the pre-warped pulsation (radian frequency) such that the frequency of a bilinear
/// discrete process matches the frequency of its analog counterpart. Bound the pre-warping to the
/// point at which the pre-warped and raw frequencies match, and only pre-warp frequencies below.
///
/// # Arguments
///
/// `samplerate`: Sample rate of the discrete process
/// `wc`: Radian frequency (in Hz/rad) input
///
/// returns `T`
pub fn bilinear_prewarming_bounded<T: Scalar>(samplerate: T, wc: T) -> T {
    let wmax = samplerate * T::simd_frac_pi_2();
    wc.simd_lt(wmax).if_else(
        || bilinear_prewarming(samplerate, wc),
        || wc * bilinear_prewarming(samplerate, wmax) / wmax,
    )
}

/// Exponential smooth minimum
///
/// # Arguments
///
/// `t`: Smoothing factor, where 0 is equivalent to a regular min.
/// `a`: Value 1
/// `b`: Value 2
///
/// returns: `T`
#[replace_float_literals(T::from_f64(literal))]
#[inline]
pub fn smooth_min<T: Scalar>(t: T, a: T, b: T) -> T {
    let r = (-a / t).simd_exp2() + (-b / t).simd_exp2();
    -t * r.simd_log2()
}

/// Exponential smooth maximum
///
/// # Arguments
///
/// `t`: Smoothing factor, where 0 is equivalent to a regular max.
/// `a`: Value 1
/// `b`: Value 2
///
/// returns: `T`
#[inline]
pub fn smooth_max<T: Scalar>(t: T, a: T, b: T) -> T {
    -smooth_min(t, -a, -b)
}

/// Exponential smooth clamping
///
/// # Arguments
///
/// `t`: Smoothing factor, where 0 is equivalent to a regular clamp.
/// `x`: Input value
/// `min`: Minimum value on the output
/// `max`: Maximum value on the output
///
/// returns: `T`
#[inline]
pub fn smooth_clamp<T: Scalar>(t: T, x: T, min: T, max: T) -> T {
    smooth_max(t, min, smooth_min(t, x, max))
}
