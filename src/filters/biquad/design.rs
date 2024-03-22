use std::f64::consts::PI;
use std::{fmt, ops};

use nalgebra::Complex;
use num_traits::{NumOps, One, Zero};
use simba::simd::{SimdComplexField, SimdValue};

use crate::dsp::blocks::Series;
use crate::math::polynom::Polynom;
use crate::saturators::Linear;
use crate::Scalar;

use super::Biquad;

/// Structure holding the numerator and denominator of a fraction as separate numbers.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Rational<T>(pub T, pub T);

impl<T> Rational<T> {
    /// Map the numerator and denominator of this rational
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> Rational<U> {
        Rational(map(self.0), map(self.1))
    }
}

impl<T: ops::Div<T>> Rational<T> {
    /// Evaluate the fraction and return a single output.
    pub fn eval(self) -> T::Output {
        self.0 / self.1
    }
}

/// Factored transfer function rational, stored as poles and zeros.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TransferFunction<T> {
    zeros: Vec<T>,
    poles: Vec<T>,
}

impl<T> TransferFunction<T> {
    /// Map the zeros and poles of the transfer function into another value (possibly of another type)
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> TransferFunction<U> {
        TransferFunction {
            zeros: self.zeros.into_iter().map(&mut map).collect(),
            poles: self.poles.into_iter().map(map).collect(),
        }
    }
}

impl<T: Copy + One + NumOps> TransferFunction<T> {
    pub fn eval(&self, x: T) -> T {
        let num = self
            .zeros
            .iter()
            .copied()
            .map(|z| x - z)
            .fold(T::one(), ops::Mul::mul);
        let den = self
            .poles
            .iter()
            .copied()
            .map(|p| x - p)
            .fold(T::one(), ops::Mul::mul);
        num / den
    }
}

impl<T: Copy + Zero + One + PartialEq + ops::AddAssign<T> + ops::Neg<Output = T>>
    TransferFunction<T>
{
    /// Expand the transfer function into a fraction of two polynomials
    pub fn as_polynom_rational(&self) -> Rational<Polynom<T>> {
        Rational(
            Polynom::from_roots(self.zeros.iter().copied()),
            Polynom::from_roots(self.poles.iter().copied()),
        )
    }
}

impl<T: Scalar> TransferFunction<Complex<T>>
where
    T::SimdBool: SimdValue<Element = bool>,
{
    /// Perform a bilinear transform from analog to digital, using Tustin's method.
    pub fn bilinear_transform(self, samplerate: T) -> Self {
        let mut res = self.map(|x| bilinear_transform(samplerate, x));
        let final_degree = res.poles.len().max(res.zeros.len());
        let to_add = final_degree - res.zeros.len();
        res.zeros
            .extend(std::iter::repeat(-Complex::one()).take(to_add));
        res
    }

    /// Checks whether the transfer function is BIBO-stable, assuming this instance is an analog transfer function
    pub fn is_analog_stable(&self) -> T::SimdBool {
        self.poles
            .iter()
            .map(|p| p.im.is_simd_negative())
            .fold(T::SimdBool::splat(true), ops::BitAnd::bitand)
    }

    /// Checks whether the transfer function is BIBO-stable, assuming this instance is a digital transfer function.
    pub fn is_digital_stable(&self) -> T::SimdBool
    where
        T::SimdBool: fmt::Debug,
    {
        self.poles
            .iter()
            .map(|p| dbg!(p.norm_sqr()).simd_le(T::one()))
            .fold(T::SimdBool::splat(true), ops::BitAnd::bitand)
    }
}

/// Creates a [`Biquad`] instance matching the given expanded digital transfer function.
/// Only biquadratic transfer functions (degree <= 2) can be passed into this function. Use the
/// [`cascaded_biquad_sections`] function to create a series of cascaded biquads instead.
pub fn biquad<T: Scalar>(transfer_function: Rational<Polynom<T>>) -> Biquad<T, Linear> {
    assert!(transfer_function.0.degree() <= 2);
    assert!(transfer_function.1.degree() <= 2);

    let a0 = transfer_function.1.get(2);
    Biquad::new(
        std::array::from_fn(|i| transfer_function.0.get(2 - i)),
        std::array::from_fn(|i| transfer_function.1.get(1 - i) / a0),
    )
}

/// Perform the bilinear transform over a single complex number, using Tustin's method.
pub fn bilinear_transform<T: Scalar>(samplerate: T, s: Complex<T>) -> Complex<T> {
    let samplerate = Complex::from(samplerate);
    let num = Complex::<T>::one() + s / samplerate;
    let den = Complex::<T>::one() - s / samplerate;
    num / den
}

/// Compute the transfer function of Nth order Butterworth filter.
pub fn butterworth<T: Scalar>(order: usize, fc: T) -> TransferFunction<Complex<T>>
where
    Complex<T>: SimdComplexField,
{
    let from_theta = |theta: T| Complex::simd_exp(-Complex::i() * theta.into());
    let poles = Vec::from_iter(
        (0..order)
            .map(|k| (2.0 * k as f64 + 1.0) * PI / (2.0 * order as f64))
            .map(|theta| from_theta(T::from_f64(theta)))
            .map(|pak| pak * T::simd_two_pi().into() * fc.into()),
    );

    TransferFunction {
        zeros: vec![],
        poles,
    }
}

/// Transform a factored transfer function into a set of biquadratic transfer functions recreating
/// the original one when applied in series.
/// The order of the input transfer function can be either even or odd; if it is odd, the last biquad
/// will only be a 1-st order filter.
pub fn into_biquadratic<T>(transfer_function: TransferFunction<T>) -> Vec<TransferFunction<T>> {
    let mut zeros2 = transfer_function.zeros.into_iter().array_chunks::<2>();
    let mut poles2 = transfer_function.poles.into_iter().array_chunks::<2>();
    let mut res =
        Vec::from_iter(
            zeros2
                .by_ref()
                .zip(poles2.by_ref())
                .map(|(z, p)| TransferFunction {
                    zeros: z.into(),
                    poles: p.into(),
                }),
        );

    match (zeros2.into_remainder(), poles2.into_remainder()) {
        (None, None) => {}
        (Some(z), None) => res.push(TransferFunction {
            zeros: z.collect(),
            poles: vec![],
        }),
        (None, Some(p)) => res.push(TransferFunction {
            zeros: vec![],
            poles: p.collect(),
        }),
        (Some(z), Some(p)) => res.push(TransferFunction {
            zeros: z.collect(),
            poles: p.collect(),
        }),
    }
    res
}

/// Instanciates a set of biquads in series which implement the Nth order transfer function given as argument.
pub fn cascaded_biquad_sections<T: Scalar>(
    transfer_function: TransferFunction<Complex<T>>,
) -> Series<Vec<Biquad<T, Linear>>> {
    let v = into_biquadratic(transfer_function)
        .into_iter()
        .map(|tf| tf.as_polynom_rational().map(|p| p.map(|x| x.re)))
        .map(biquad)
        .collect();
    Series(v)
}

/// Computes a filter that implements the given Nth order Butterworth filter as a series of cascaded
/// Biquad filters.
pub fn biquad_butterworth<T: Scalar>(order: usize, fc: T) -> Series<Vec<Biquad<T, Linear>>>
where
    Complex<T>: SimdComplexField,
{
    cascaded_biquad_sections(butterworth(order, fc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_butterworth_analog() {
        let butter = butterworth(2, 0.25f64);
        assert!(butter.is_analog_stable());
        insta::assert_debug_snapshot!(butter);
    }

    #[test]
    fn test_butterworth_digital() {
        let butter = dbg!(dbg!(butterworth(2, 0.25f64)).bilinear_transform(TAU));
        assert!(butter.is_digital_stable());
        insta::assert_debug_snapshot!(butter);
    }
}
