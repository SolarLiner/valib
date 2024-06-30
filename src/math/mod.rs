use std::hint;

use nalgebra::{Complex, Dim, OVector, SMatrix, SVector};
use num_traits::Float;
use numeric_literals::replace_float_literals;
use simba::simd::{SimdBool, SimdComplexField};

use crate::Scalar;

pub mod interpolation;
pub mod lut;
#[cfg(feature = "math-polynom")]
pub mod polynom;

/// Trait desciring a multivariate root equation. Root equations are solved with numerical methods such as
/// Newton-Rhapson, when linear algebra cannot be used (e.g. in the case of nonlinear systems).
pub trait RootEq<T, const N: usize> {
    /// Evaluate the equation at the given input vector.
    fn eval(&self, input: &SVector<T, N>) -> SVector<T, N>;

    /// Evaluate the **inverse** jacobian of the equation. When the jacobian cannot be computed, [`None`] should be returned instead.
    fn j_inv(&self, input: &SVector<T, N>) -> Option<SMatrix<T, N, N>>;
}

/// Perform a single step of the Newton-Rhapson algorithm. This takes the inverse jabobian and computes the differential to the next step.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
pub fn nr_step<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    input: &SVector<T, N>,
) -> Option<SVector<T, N>>
where
    T::Element: Float,
{
    let ret = eq.j_inv(input).map(|jinv| jinv * eq.eval(input))?;
    let all_finite = ret
        .iter()
        .copied()
        .flat_map(|v| v.into_iter())
        .all(|v| v.is_finite());

    debug_assert!(all_finite);

    all_finite.then_some(ret)
}

/// Solve the given root equation using Newton-Rhapson for a specified number of setps.
#[inline]
pub fn newton_rhapson_steps<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    iter: usize,
) -> T
where
    T::Element: Float,
{
    let Some(mut step) = nr_step(eq, value) else {
        return T::zero();
    };
    for _ in 1..iter {
        *value -= step;
        if let Some(newstep) = nr_step(eq, value) {
            step = newstep;
        } else {
            break;
        }
    }
    rms(&step)
}

/// Solve the given root equation using Newton-Rhapson until the RMS of the differential is lesser than the given tolerance.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline(always))]
pub fn newton_rhapson_tolerance<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
) -> usize
where
    T::Element: Float,
{
    let mut i = 0;

    while let Some(step) = nr_step(eq, value) {
        debug_assert!(i < 500, "Newton-Rhapson got stuck");
        i += 1;
        let tgt = hint::black_box(rms(&step)).simd_lt(tol);
        if tgt.all() {
            break;
        }
        let changed = *value - step;
        *value = value.zip_map(&changed, |v, c| v.select(tgt, c));
    }

    i
}

/// Solve the given root equation using Newton-Rhapson, until either the RMS of the differential is less than the
/// given tolerances, or the specified max number of steps has been taken.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline(always))]
pub fn newton_rhapson_tol_max_iter<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
    max_iter: usize,
) where
    T::Element: Float,
{
    for _ in 0..max_iter {
        let Some(step) = nr_step(eq, value) else {
            break;
        };
        let tgt = rms(&step).simd_lt(tol);
        if tgt.all() {
            break;
        }
        let changed = *value - step;
        *value = value.zip_map(&changed, |v, c| v.select(tgt, c));
    }
}

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
pub fn smooth_min<T: Scalar>(t: T, a: T, b: T) -> T {
    // Polynomial
    //let h = (0.5 + 0.5 * (a - b) / t).simd_clamp(0.0, 1.0);
    //lerp(h, a, b) - t * h * (1.0 - h)

    // Exponential
    let r = (-a / t).simd_exp2() + (-b / t).simd_exp2();
    -t * r.simd_log2()
}

/// Exponential smooth maximum
pub fn smooth_max<T: Scalar>(t: T, a: T, b: T) -> T {
    -smooth_min(t, -a, -b)
}

/// Exponential smooth clamping
pub fn smooth_clamp<T: Scalar>(t: T, x: T, min: T, max: T) -> T {
    smooth_max(t, min, smooth_min(t, x, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SqrtNumerical {
        squared: f64,
    }

    impl RootEq<f64, 1> for SqrtNumerical {
        fn eval(&self, input: &SVector<f64, 1>) -> SVector<f64, 1> {
            SVector::<_, 1>::new(self.squared - input[0].powi(2))
        }

        fn j_inv(&self, input: &SVector<f64, 1>) -> Option<SMatrix<f64, 1, 1>> {
            let diff = -2.0 * input[0];
            Some(SVector::<_, 1>::new((diff + 1e-6).recip()))
        }
    }

    #[test]
    fn test_solve() {
        let equ = SqrtNumerical { squared: 4.0 };
        let mut actual = SVector::<_, 1>::new(0.0);
        let iters = newton_rhapson_tolerance(&equ, &mut actual, 1e-4);
        let expected = 2.0;

        println!("Iterations: {iters} | {actual}");
        assert!((expected - actual[0].abs()).abs() <= 1e-4);
    }

    #[test]
    #[should_panic]
    fn test_detect_nan() {
        struct Equ;

        impl RootEq<f64, 1> for Equ {
            fn eval(&self, _: &SVector<f64, 1>) -> SVector<f64, 1> {
                SVector::zeros()
            }

            fn j_inv(&self, _: &SVector<f64, 1>) -> Option<SMatrix<f64, 1, 1>> {
                Some(SVector::<_, 1>::new(f64::NAN))
            }
        }

        assert_eq!(
            0,
            newton_rhapson_tolerance(&Equ, &mut SVector::zeros(), 1e-4)
        );
    }

    #[test]
    fn test_freq_to_z() {}
}
