//! # Antiderivative Anti-Aliasing
//!
//! Methods which suppresses aliasing by relying on 1st and 2nd order antiderivatives
use numeric_literals::replace_float_literals;
use valib_core::simd::SimdBool;

use crate::{Asinh, Blend, Clipper, Saturator, Tanh};
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::Scalar;

/// Trait for functions that have antiderivatives.
pub trait Antiderivative<T> {
    /// Evaluate the function itself.
    ///
    /// # Arguments
    ///
    /// * `x`: Input to the function
    ///
    /// returns: T
    fn evaluate(&self, x: T) -> T;

    /// Evaluate the antiderivative of the function. The extra constant is assumed to be zero.
    ///
    /// # Arguments
    ///
    /// * `x`: Input value
    ///
    /// returns: T
    fn antiderivative(&self, x: T) -> T;
}

/// Trait for functions which have a 2nd-order antiderivative.
pub trait Antiderivative2<T>: Antiderivative<T> {
    /// Evaluate the 2nd-order antiderivative. The additional constants are assumed to be zero.
    ///
    /// # Arguments
    ///
    /// * `x`: Function input
    ///
    /// returns: T
    fn antiderivative2(&self, x: T) -> T;
}

impl<T: Scalar> Antiderivative<T> for Tanh {
    fn evaluate(&self, x: T) -> T {
        x.simd_tanh()
    }

    fn antiderivative(&self, x: T) -> T {
        x.simd_cosh().simd_ln()
    }
}

impl<T: Scalar> Antiderivative<T> for Asinh {
    fn evaluate(&self, x: T) -> T {
        x.simd_asinh()
    }

    fn antiderivative(&self, x: T) -> T {
        let x0 = x * x + T::one();
        x * x.simd_asinh() - x0.simd_sqrt()
    }
}

impl<T: Scalar> Antiderivative2<T> for Asinh {
    #[replace_float_literals(T::from_f64(literal))]
    fn antiderivative2(&self, x: T) -> T {
        let x0 = x * x + 1.0;
        let x1 = 2.0 * x * x - 1.0;
        let x2 = -x * 3.0 * x0.simd_sqrt() / 4.0;
        let x3 = x1 * x.simd_asinh() / 4.0;
        x2 + x3
    }
}

impl<T: Scalar> Antiderivative<T> for Clipper<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn evaluate(&self, x: T) -> T {
        x.simd_clamp(self.min, self.max)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn antiderivative(&self, x: T) -> T {
        let lower = -x;
        let upper = x;
        let middle = (0.5) * (x * x + T::one());
        let is_lower = x.simd_lt(self.min);
        let is_higher = x.simd_gt(self.max);
        lower.select(is_lower, upper.select(is_higher, middle))
    }
}

impl<T: Scalar> Antiderivative2<T> for Clipper<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn antiderivative2(&self, x: T) -> T {
        let lower = -0.5 * x * (x - 2.0);
        let upper = x * x / 2.0 * x * T::simd_recip(3.0);
        let middle = T::simd_recip(6.0) * (x * x * x + 9.0 * x + 1.0);
        let is_lower = x.simd_lt(self.min);
        let is_higher = x.simd_gt(self.max);
        lower.select(is_lower, upper.select(is_higher, middle))
    }
}

impl<T: Scalar, S: Antiderivative<T>> Antiderivative<T> for Blend<T, S> {
    fn evaluate(&self, x: T) -> T {
        x + (self.inner.evaluate(x) - x) * self.amt
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn antiderivative(&self, x: T) -> T {
        let ad_x = 0.5 * x * x;
        self.inner.antiderivative(x) * self.amt + ad_x * (1.0 + self.amt)
    }
}

impl<T: Scalar, S: Antiderivative2<T>> Antiderivative2<T> for Blend<T, S> {
    #[replace_float_literals(T::from_f64(literal))]
    fn antiderivative2(&self, x: T) -> T {
        let ad_x = x + x * x * x / 6.0;
        self.inner.antiderivative2(x) * self.amt + ad_x * (1.0 + self.amt)
    }
}

/// Antiderivative Anti-Aliasing implementation
#[derive(Debug, Copy, Clone)]
pub struct Adaa<T, S, const ORDER: usize> {
    /// Minimum input difference to use the antiderivative instead of calling the saturator directly
    pub epsilon: T,
    /// Inner saturator
    pub inner: S,
    memory: [T; ORDER],
}

impl<T: Scalar, S, const ORDER: usize> Adaa<T, S, ORDER> {
    /// Create a new ADAA saturator, wrapping an inner saturator.
    ///
    /// # Arguments
    ///
    /// * `inner`: Inner saturator
    pub fn new(inner: S) -> Self {
        Self {
            epsilon: T::from_f64(1e-3),
            inner,
            memory: [T::zero(); ORDER],
        }
    }
}

impl<T: Scalar, S: Default, const ORDER: usize> Default for Adaa<T, S, ORDER> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

impl<T: Scalar, S: Antiderivative<T>> Adaa<T, S, 1> {
    /// Compute the next sample, without updating the inner saturator state.
    ///
    /// Uses the 1st order antiderivative of the inner saturator.
    ///
    /// # Arguments
    ///
    /// * `x`: Function input
    ///
    /// returns: T
    #[replace_float_literals(T::from_f64(literal))]
    pub fn next_sample_immutable(&self, x: T) -> T {
        let den = x - self.memory[0];
        let below = den.simd_abs().simd_lt(self.epsilon);
        below.if_else(
            || self.inner.evaluate((x + self.memory[0]) / 2.0),
            || {
                let num = self.inner.antiderivative(x) - self.inner.antiderivative(self.memory[0]);
                num / den
            },
        )
    }

    /// Commit the input sample.
    ///
    /// Uses the 1st order antiderivative of the inner saturator.
    ///
    /// # Arguments
    ///
    /// * `x`: Input sample
    ///
    /// returns: ()
    pub fn commit_sample(&mut self, x: T) {
        self.memory = [x];
    }

    /// Shortcut for calling [`Sample::next_sample_immutable`], then [`Sample::commit_sample`].
    ///
    /// Uses the 1st order antiderivative of the inner saturator.
    ///
    /// # Arguments
    ///
    /// * `x`: Input sample
    ///
    /// returns: T
    pub fn next_sample(&mut self, x: T) -> T {
        let y = self.next_sample_immutable(x);
        self.commit_sample(x);
        y
    }
}

impl<T: Scalar, S: Antiderivative2<T>> Adaa<T, S, 2> {
    /// Compute the next sample, without updating the inner saturator state.
    ///
    /// Uses the 1st order antiderivative of the inner saturator.
    ///
    /// # Arguments
    ///
    /// * `x`: Function input
    ///
    /// returns: T
    #[replace_float_literals(T::from_f64(literal))]
    #[profiling::function]
    pub fn next_sample_immutable(&self, x: T) -> T {
        let [x1, x2] = self.memory;
        let den1 = x - x1;
        let den2 = x1 - x2;
        let den3 = x - x2;
        let below1 = den1.simd_abs().simd_lt(self.epsilon);
        let below2 = den2.simd_abs().simd_lt(self.epsilon);
        let below3 = den3.simd_abs().simd_lt(self.epsilon);
        (below1 | below2 | below3).if_else(
            || self.inner.evaluate((x + x1) / 2.0),
            || {
                let num1 = self.inner.antiderivative(x) - self.inner.antiderivative2(x1);
                let num2 = self.inner.antiderivative2(x1) - self.inner.antiderivative2(x2);
                den3.simd_recip() * (num1 / den1 + num2 / den2)
            },
        )
    }

    /// Commit the input sample.
    ///
    /// Uses the 1st order antiderivative of the inner saturator.
    ///
    /// # Arguments
    ///
    /// * `x`: Input sample
    ///
    /// returns: ()
    pub fn commit_sample(&mut self, x: T) {
        self.memory.swap(0, 1);
        self.memory[0] = x;
    }

    /// Shortcut for calling [`Sample::next_sample_immutable`], then [`Sample::commit_sample`].
    ///
    /// Uses the 1st order antiderivative of the inner saturator.
    ///
    /// # Arguments
    ///
    /// * `x`: Input sample
    ///
    /// returns: T
    pub fn next_sample(&mut self, x: T) -> T {
        let y = self.next_sample_immutable(x);
        self.commit_sample(x);
        y
    }
}

#[profiling::all_functions]
impl<T: Scalar, S: Antiderivative<T> + Saturator<T>> Saturator<T> for Adaa<T, S, 1> {
    fn saturate(&self, x: T) -> T {
        self.next_sample_immutable(x)
    }

    fn update_state(&mut self, x: T, y: T) {
        self.commit_sample(x);
        self.inner.update_state(x, y);
    }

    fn sat_diff(&self, x: T) -> T {
        self.inner.sat_diff(x)
    }
}

impl<T: Scalar, S> DSPMeta for Adaa<T, S, 1> {
    type Sample = T;
}

#[profiling::all_functions]
impl<T: Scalar, S: Antiderivative<T>> DSPProcess<1, 1> for Adaa<T, S, 1>
where
    Self: DSPMeta<Sample = T>,
{
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.next_sample(x)]
    }
}

#[profiling::all_functions]
impl<T: Scalar, S: Antiderivative2<T> + Saturator<T>> Saturator<T> for Adaa<T, S, 2> {
    #[replace_float_literals(T::from_f64(literal))]
    fn saturate(&self, x: T) -> T {
        self.next_sample_immutable(x)
    }

    fn update_state(&mut self, x: T, y: T) {
        self.commit_sample(x);
        self.inner.update_state(x, y);
    }

    fn sat_diff(&self, x: T) -> T {
        self.inner.sat_diff(x)
    }
}

impl<T: Scalar, S> DSPMeta for Adaa<T, S, 2> {
    type Sample = T;

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.memory.fill_with(T::zero);
    }
}

#[profiling::all_functions]
impl<T: Scalar, S: Antiderivative2<T>> DSPProcess<1, 1> for Adaa<T, S, 2>
where
    Self: DSPMeta<Sample = T>,
{
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = self.next_sample(x);
        [y]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::f64::consts::TAU;

    #[rstest]
    #[case("tanh", Adaa::< _, Tanh, 1 >::default())]
    #[case("asinh", Adaa::< _, Asinh, 1 >::default())]
    #[case("clipper", Adaa::< _, Clipper<f64>, 1 >::default())]
    fn test_adaa1<S: Antiderivative<f64> + Saturator<f64>>(
        #[case] name: &str,
        #[case] mut adaa: Adaa<f64, S, 1>,
    ) {
        let samplerate = 100.0;
        let f = 10.0;
        let input: [_; 100] =
            std::array::from_fn(|i| i as f64 / samplerate).map(|t| 3.0 * f64::sin(TAU * f * t));
        let output = input.map(|x| {
            let y = adaa.saturate(x);
            adaa.update_state(x, y);
            y
        });

        let name = format!("test_adaa1_{name}");
        insta::assert_csv_snapshot!(name, &output as &[_], { "[]" => insta::rounded_redaction(3) })
    }

    #[rstest]
    #[case("clipper", Adaa::< _, Clipper<f64>, 2 >::default())]
    #[case("asinh", Adaa::< _, Asinh, 2 >::default())]
    fn test_adaa2<S: Antiderivative2<f64> + Saturator<f64>>(
        #[case] name: &str,
        #[case] mut adaa: Adaa<f64, S, 2>,
    ) {
        let samplerate = 100.0;
        let f = 10.0;
        let input: [_; 100] =
            std::array::from_fn(|i| i as f64 / samplerate).map(|t| 3.0 * f64::sin(TAU * f * t));
        let output = input.map(|x| {
            let y = adaa.saturate(x);
            adaa.update_state(x, y);
            y
        });

        let name = format!("test_adaa2_{name}",);
        insta::assert_csv_snapshot!(name, &output as &[_], { "[]" => insta::rounded_redaction(3) })
    }
}
