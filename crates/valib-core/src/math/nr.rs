//! Module for working with numerical root finding, using the Newton-Rhapson method.
use crate::math;
use crate::Scalar;
use nalgebra::{SMatrix, SVector};
use num_traits::Float;
use simba::simd::SimdBool;
use std::hint;

/// Trait desciring a multivariate root equation. Root equations are solved with numerical methods such as
/// Newton-Rhapson, when linear algebra cannot be used (e.g. in the case of nonlinear systems).
pub trait RootEq<T, const N: usize> {
    /// Evaluate the equation at the given input vector.
    fn eval(&self, input: &SVector<T, N>) -> SVector<T, N>;

    /// Evaluate the **inverse** jacobian of the equation. When the jacobian cannot be computed, [`None`] should be returned instead.
    fn j_inv(&self, input: &SVector<T, N>) -> Option<SMatrix<T, N, N>>;
}

/// Perform a single step of the Newton-Rhapson algorithm. This takes the inverse jabobian and
/// computes the differential to the next step.
///
/// # Arguments
///
/// * `eq`: Equation to solve
/// * `input`: Pre-iteration value.
///
/// returns: Option<Matrix<T, Const<{ N }>, Const<1>, ArrayStorage<T, { N }, 1>>>
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

    all_finite.then_some(ret)
}

/// Solve the given root equation using Newton-Rhapson for a specified number of setps.
///
/// Returns the root-mean-square error after the last iteration.
///
/// # Arguments
///
/// * `eq`: Equation to solve
/// * `value`: Initial guess and output value
/// * `iter`: Number of iterations to perform
///
/// returns: T
#[inline]
#[profiling::function]
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
    math::rms(&step)
}

/// Solve the given root equation using Newton-Rhapson until the RMS of the differential is lesser
/// than the given tolerance.
///
/// # Arguments
///
/// * `eq`: Equation to solve
/// * `value`: Initial guess and output value
/// * `tol`: Maximum tolerance for the output.
///
/// returns: usize
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[profiling::function]
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
        let tgt = hint::black_box(math::rms(&step)).simd_lt(tol);
        if tgt.all() {
            break;
        }
        let changed = *value - step;
        *value = value.zip_map(&changed, |v, c| v.select(tgt, c));
    }

    i
}

/// Solve the given root equation using Newton-Rhapson, until either the RMS of the differential is
/// less than the given tolerances, or the specified max number of steps has been taken.
///
/// Returns the number of iterations performed.
///
/// # Arguments
///
/// * `eq`: Equation to solve
/// * `value`: Initial guess and output value
/// * `tol`: Maximum tolerance for the output.
/// * `max_iter`: Maximum number of iterations to perform.
///
/// returns: usize
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[profiling::function]
pub fn newton_rhapson_tol_max_iter<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
    max_iter: usize,
) -> usize
where
    T::Element: Float,
{
    for i in 0..max_iter {
        let Some(step) = nr_step(eq, value) else {
            return i;
        };
        let changed = *value - step;
        let tgt = math::rms(&step).simd_lt(tol);
        *value = value.zip_map(&changed, |v, c| v.select(tgt, c));
        if tgt.all() {
            return i;
        }
    }
    max_iter
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
}
