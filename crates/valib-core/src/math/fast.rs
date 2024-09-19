use crate::Scalar;
use numeric_literals::replace_float_literals;
use simba::simd::SimdBool;

/// Rational approximation of tanh(x) which is valid in the range -3..3
///
/// This approximation only includes the rational approximation part, and will diverge outside the
/// bounds. In order to apply the tanh function over a bigger interval, consider clamping either the
/// input or the output.
///
/// You should consider using [`tanh`] for a general-purpose faster tanh function, which uses
/// branching.
///
/// Source: <https://www.musicdsp.org/en/latest/Other/238-rational-tanh-approximation.html>
///
/// # Arguments
///
/// * `x`: Input value (low-error range: -3..3)
///
/// returns: T
#[replace_float_literals(T::from_f64(literal))]
pub fn rational_tanh<T: Scalar>(x: T) -> T {
    x * (27. + x * x) / (27. + 9. * x * x)
}

/// Fast approximation of tanh(x).
///
/// This approximation uses branching to clamp the output to -1..1 in order to be useful as a
/// general-purpose approximation of tanh.
///
/// Source: <https://www.musicdsp.org/en/latest/Other/238-rational-tanh-approximation.html>
///
/// # Arguments
///
/// * `x`: Input value
///
/// returns: T
pub fn tanh<T: Scalar>(x: T) -> T {
    rational_tanh(x).simd_clamp(-T::one(), T::one())
}

/// Fast approximation of exp, with maximum error in -1..1 of 0.59%, and in -3.14..3.14 of 9.8%.
///
/// You should consider using [`exp`] for a better approximation which uses this function, but
/// allows a greater range at the cost of branching.
///
/// Source: <https://www.musicdsp.org/en/latest/Other/222-fast-exp-approximations.html>
///
/// # Arguments
///
/// * `x`: Input value
///
/// returns: T
#[replace_float_literals(T::from_f64(literal))]
pub fn fast_exp5<T: Scalar>(x: T) -> T {
    (120. + x * (120. + x * (60. + x * (20. + x * (5. + x))))) * 0.0083333333
}

/// Fast approximation of exp, using [`fast_exp5`]. Uses branching to get a bigger range.
///
/// Maximum error in the 0..10.58 range is 0.45%.
///
/// Source: <https://www.musicdsp.org/en/latest/Other/222-fast-exp-approximations.html>
///
/// # Arguments
///
/// * `x`:
///
/// returns: T
#[replace_float_literals(T::from_f64(literal))]
pub fn exp<T: Scalar>(x: T) -> T {
    x.simd_lt(2.5).if_else2(
        || T::simd_e() * fast_exp5(x - 1.),
        (|| x.simd_lt(5.), || 33.115452 * fast_exp5(x - 3.5)),
        || 403.42879 * fast_exp5(x - 6.),
    )
}

/// Fast 2^x approximation, using [`exp`].
///
/// Maximum error in the 0..15.26 range is 0.45%.
///
/// Source: <https://www.musicdsp.org/en/latest/Other/222-fast-exp-approximations.html>
///
/// # Arguments
///
/// * `x`:
///
/// returns: T
///
/// # Examples
///
/// ```
///
/// ```
pub fn pow2<T: Scalar>(x: T) -> T {
    let log_two = T::simd_ln_2();
    exp(log_two * x)
}
