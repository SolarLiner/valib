use nalgebra::{ComplexField, Dim, OVector, SMatrix, SVector};

use crate::Scalar;

pub trait RootEq<T, const N: usize> {
    fn eval(&self, input: &SVector<T, N>) -> SVector<T, N>;

    fn j_inv(&self, input: &SVector<T, N>) -> Option<SMatrix<T, N, N>>;
}

#[inline]
pub fn nr_step<T: Scalar + ComplexField, const N: usize>(
    eq: &impl RootEq<T, N>,
    input: &SVector<T, N>,
) -> Option<SVector<T, N>> {
    eq.j_inv(input).map(|jinv| jinv * eq.eval(input))
}

#[inline]
pub fn newton_rhapson_steps<T: Scalar + ComplexField, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    iter: usize,
) -> T {
    let Some(mut step ) = nr_step(eq, value) else { return T::zero() };
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

#[inline]
pub fn newton_rhapson_tolerance<T: Scalar + ComplexField, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
) -> usize {
    let mut i = 0;

    while let Some(step) = nr_step(eq, value) {
        i += 1;
        *value -= step;
        if rms(&step) < tol {
            break;
        }
    }

    i
}

#[inline]
pub fn newton_rhapson_tol_max_iter<T: Scalar + ComplexField, const N: usize>(
    eq: &impl RootEq<T, N>,
    value: &mut SVector<T, N>,
    tol: T,
    max_iter: usize
) {
    for _ in 0..max_iter {
        let Some(step) = nr_step(eq, value) else {
            break;
        };
        if rms(&step) < tol {
            break;
        }
        *value -= step;
    }
}

#[inline]
fn rms<T: ComplexField, D: Dim>(value: &OVector<T, D>) -> T
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<T, D>,
{
    value.map(|v| v.powi(2)).sum().sqrt()
}
