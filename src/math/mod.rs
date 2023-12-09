use std::hint;

use nalgebra::{Dim, OVector, SMatrix, SVector};
use num_traits::Float;
use simba::simd::{SimdComplexField,SimdBool};

use crate::Scalar;

pub trait RootEq<T, const N: usize> {
    fn eval(&self, input: &SVector<T, N>) -> SVector<T, N>;

    fn j_inv(&self, input: &SVector<T, N>) -> Option<SMatrix<T, N, N>>;
}

#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
pub fn nr_step<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    input: &SVector<T, N>,
) -> Option<SVector<T, N>> where T::Element: Float {
    let ret = eq.j_inv(input).map(|jinv| jinv * eq.eval(input))?;
    let all_finite = ret.iter().copied().flat_map(|v| v.into_iter()).all(|v| v.is_finite());

    debug_assert!(all_finite);

    all_finite.then_some(ret)
}

#[inline]
pub fn newton_rhapson_steps<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    iter: usize,
) -> T where T::Element: Float {
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

#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline(always))]
pub fn newton_rhapson_tolerance<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
) -> usize where T::Element: Float {
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

#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline(always))]
pub fn newton_rhapson_tol_max_iter<T: Scalar, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
    max_iter: usize,
) where T::Element: Float {
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

#[inline]
fn rms<T: SimdComplexField, D: Dim>(value: &OVector<T, D>) -> T
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, D>,
{
    value.map(|v| v.simd_powi(2)).sum().simd_sqrt()
}

#[cfg(test)]
mod tests {
    use nalgebra::ComplexField;

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
                Some(SVector::<_,1>::new(f64::NAN))
            }
        }

        assert_eq!(0, newton_rhapson_tolerance(&Equ, &mut SVector::zeros(), 1e-4));
    }
}