//! Module for working with numerical root finding, using the Newton-Rhapson method.
use crate::math;
use crate::Scalar;
use nalgebra as na;
use nalgebra::{Dim, OMatrix, OVector, VectorView, VectorViewMut};
use num_traits::Float;
use simba::simd::{SimdBool, SimdPartialOrd};
use std::num::NonZeroUsize;

/// Trait desciring a multivariate root equation. Root equations are solved with numerical methods such as
/// Newton-Rhapson, when linear algebra cannot be used (e.g. in the case of nonlinear systems).
pub trait RootEq
where
    na::DefaultAllocator:
        na::allocator::Allocator<Self::Dim> + na::allocator::Allocator<Self::Dim, Self::Dim>,
{
    /// Scalar type of the equation
    type Scalar: Scalar;

    /// Equation dimension, typed using [`nalgebra`] dimensions.
    type Dim: Dim;

    /// Evaluate the equation at the given input vector.
    fn eval(
        &self,
        input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
    ) -> OVector<Self::Scalar, Self::Dim>;

    /// Evaluate the **inverse** jacobian of the equation. When the jacobian cannot be computed, [`None`] should be returned instead.
    fn j_inv(
        &self,
        input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
    ) -> Option<OMatrix<Self::Scalar, Self::Dim, Self::Dim>>;
}

impl<'a, Equ: RootEq> RootEq for &'a Equ
where
    na::DefaultAllocator:
        na::allocator::Allocator<Equ::Dim> + na::allocator::Allocator<Equ::Dim, Equ::Dim>,
{
    type Scalar = Equ::Scalar;
    type Dim = Equ::Dim;

    fn eval(
        &self,
        input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
    ) -> OVector<Self::Scalar, Self::Dim> {
        Equ::eval(self, input)
    }

    fn j_inv(
        &self,
        input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
    ) -> Option<OMatrix<Self::Scalar, Self::Dim, Self::Dim>> {
        Equ::j_inv(self, input)
    }
}

/// Perform root-finding over an implicit equation with the Newton-Rhapson method.
#[derive(Debug)]
pub struct NewtonRhapson<Equ: RootEq>
where
    na::DefaultAllocator:
        na::allocator::Allocator<Equ::Dim> + na::allocator::Allocator<Equ::Dim, Equ::Dim>,
{
    /// Maximum tolerance accepted to terminate iteration
    pub tolerance: Option<Equ::Scalar>,
    /// Maximum number of iterations allowed to find the root
    pub max_iterations: Option<NonZeroUsize>,
    /// Implicit equation type
    pub equation: Equ,
}

impl<Equ: RootEq> NewtonRhapson<Equ>
where
    Equ::Scalar: Scalar<Element: Float>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Equ::Dim>
        + nalgebra::allocator::Allocator<Equ::Dim, Equ::Dim>,
    OVector<Equ::Scalar, Equ::Dim>: Copy,
{
    /// Create a new solver, given the equation and settings.
    ///
    /// # Arguments
    ///
    /// * `equation`: Implicit equation to solve
    /// * `tolerance`: Maximum tolerance allowed to terminate the iteration
    /// * `max_iterations`: Maximum number of iterations allowed
    ///
    /// returns: NewtonRhapson<Equ>
    pub fn new(
        equation: Equ,
        tolerance: Option<Equ::Scalar>,
        max_iterations: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            tolerance,
            max_iterations,
            equation,
        }
    }

    /// Run the root-finding algorithm, given the initial guess.
    ///
    /// # Arguments
    ///
    /// * `initial_guess`: Initial guess to use as first value into the iteration scheme.
    ///     Performance depends a lot on this value being a good guess for a root of the equation.
    ///
    /// returns: Matrix<<Equ as RootEq>::Scalar, <Equ as RootEq>::Dim, Const<1>, <DefaultAllocator as Allocator<<Equ as RootEq>::Dim, Const<1>>>::Buffer<<Equ as RootEq>::Scalar>>
    pub fn run(
        &self,
        mut initial_guess: OVector<Equ::Scalar, Equ::Dim>,
    ) -> OVector<Equ::Scalar, Equ::Dim> {
        let view = initial_guess.as_view_mut();
        self.run_in_place(view);
        initial_guess
    }

    /// Run the root-finding algorithm, using the provided view as initial guess and result.
    ///
    /// # Arguments
    ///
    /// * `value`:  Initial guess to use as first value into the iteration scheme.
    ///     Performance depends a lot on this value being a good guess for a root of the equation.
    ///
    /// returns: usize
    pub fn run_in_place(
        &self,
        mut value: VectorViewMut<Equ::Scalar, Equ::Dim, impl Dim, impl Dim>,
    ) -> usize {
        debug_assert!(
            self.tolerance.is_some() || self.max_iterations.is_some(),
            "Current Newron-Rhapson solver configuration would lead to infinite loop"
        );

        for i in self.iterations_iter() {
            let Some(ret) = self
                .equation
                .j_inv(value.as_view())
                .map(|jinv| jinv * self.equation.eval(value.as_view()))
            else {
                return i;
            };
            let all_finite = ret
                .iter()
                .copied()
                .flat_map(|v| v.into_iter())
                .all(|v| v.is_finite());

            value -= ret;
            if !all_finite || self.check_tolerance(ret.as_view()) {
                return i;
            }
        }
        self.max_iterations.map(|m| m.get()).unwrap_or(0)
    }

    fn iterations_iter(&self) -> impl Iterator<Item = usize> {
        struct Iter {
            max: Option<usize>,
            current: usize,
        }
        impl Iterator for Iter {
            type Item = usize;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(max) = self.max {
                    if self.current > max {
                        return None;
                    }
                }
                let ret = Some(self.current);
                self.current += 1;
                ret
            }
        }

        Iter {
            max: self.max_iterations.map(|m| m.get()),
            current: 0,
        }
    }

    fn check_tolerance(
        &self,
        value: VectorView<Equ::Scalar, Equ::Dim, impl Dim, impl Dim>,
    ) -> bool {
        if let Some(tol) = self.tolerance {
            math::rms(value).simd_lt(tol).all()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::vector_view_mut;
    use nalgebra::SVector;

    struct SqrtNumerical {
        squared: f64,
    }

    impl RootEq for SqrtNumerical {
        type Scalar = f64;
        type Dim = na::U1;

        fn eval(
            &self,
            input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
        ) -> OVector<Self::Scalar, Self::Dim> {
            [self.squared - input[0].powi(2)].into()
        }

        fn j_inv(
            &self,
            input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
        ) -> Option<OMatrix<Self::Scalar, Self::Dim, Self::Dim>> {
            Some([-2.0 * input[0]].map(|x| (x + 1e-6).recip()).into())
        }

        // fn eval(&self, input: &SVector<f64, 1>) -> SVector<f64, 1> {
        //     SVector::<_, 1>::new(self.squared - input[0].powi(2))
        // }
        //
        // fn j_inv(&self, input: &SVector<f64, 1>) -> Option<SMatrix<f64, 1, 1>> {
        //     let diff = -2.0 * input[0];
        //     Some(SVector::<_, 1>::new((diff + 1e-6).recip()))
        // }
    }

    #[test]
    fn test_solve() {
        let equ = SqrtNumerical { squared: 4.0 };
        let nr = NewtonRhapson::new(equ, Some(1e-4), None);
        let mut actual: SVector<_, 1> = na::zero();
        let iters = nr.run_in_place(vector_view_mut(&mut actual));
        let expected = 2.0;

        println!("Iterations: {iters} | {actual}");
        assert!((expected - actual[0].abs()).abs() <= 1e-4);
    }

    #[test]
    fn test_detect_nan() {
        struct Equ;

        impl RootEq for Equ {
            type Scalar = f64;
            type Dim = na::U1;

            fn eval(
                &self,
                _input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
            ) -> OVector<Self::Scalar, Self::Dim> {
                na::zero()
            }

            fn j_inv(
                &self,
                _input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
            ) -> Option<OMatrix<Self::Scalar, Self::Dim, Self::Dim>> {
                Some([f64::NAN].into())
            }
        }

        let mut actual = na::zero();
        let nr = NewtonRhapson::new(Equ, None, NonZeroUsize::new(2));

        assert_eq!(0, nr.run_in_place(vector_view_mut(&mut actual)));
    }
}
