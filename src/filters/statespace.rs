//! Linear state-space model implementation for arbitrary I/O.
//!
//! # Example
//!
//! ```rust
//! use nalgebra::SMatrix;
//! use valib::dsp::DSPProcess;
//! use valib::filters::statespace::StateSpace;
//! use valib::Scalar;
//!
//! /// Implements a 1-pole lowpass filter as a linear state-space model
//! fn create_filter(fc: f32) -> StateSpace<f32, 1, 1, 1> {
//!     let new = SMatrix::<_, 1, 1>::new;
//!     StateSpace ::new(
//!         new(-(fc - 2.0) / (fc + 2.0)),
//!         new(1.0),
//!         new(-fc * (fc - 2.0) / (fc + 2.0).powi(2) + fc / (fc + 2.0)),
//!         new(fc / (fc + 2.0)),
//!     )
//! }
//!
//! let mut filter = create_filter(0.25);
//! let output = filter.process([0.0]);
//! ```
use std::f64::NAN;

use nalgebra::{Complex, SMatrix, SVector, SimdComplexField};
use num_traits::Zero;

use crate::dsp::{analysis::DspAnalysis, DSPMeta, DSPProcess};
use crate::Scalar;

// TODO: Add saturators
/// Linear discrete state-space method implementation with direct access to the state space matrices.
#[derive(Debug, Copy, Clone)]
pub struct StateSpace<T: nalgebra::Scalar, const IN: usize, const STATE: usize, const OUT: usize> {
    pub a: SMatrix<T, STATE, STATE>,
    pub b: SMatrix<T, STATE, IN>,
    pub c: SMatrix<T, OUT, STATE>,
    pub d: SMatrix<T, OUT, IN>,
    state: SVector<T, STATE>,
}

impl<
        T: Copy + Scalar + nalgebra::RealField,
        const IN: usize,
        const STATE: usize,
        const OUT: usize,
    > DspAnalysis<IN, OUT> for StateSpace<T, IN, STATE, OUT>
{
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; OUT]; IN] {
        let a = self.a.map(Complex::from_simd_real);
        let b = self.b.map(Complex::from_simd_real);
        let c = self.c.map(Complex::from_simd_real);
        let d = self.d.map(Complex::from_simd_real);
        let zi = SMatrix::<_, STATE, STATE>::identity() * z;
        let mut zia = zi - a;
        if !zia.try_inverse_mut() {
            zia.iter_mut()
                .for_each(|v| *v = Complex::from_simd_real(<T as Scalar>::from_f64(NAN)));
        }

        let h = c * zia * b + d;
        h.into()
    }
}

impl<T: nalgebra::Scalar + Zero, const IN: usize, const STATE: usize, const OUT: usize>
    StateSpace<T, IN, STATE, OUT>
{
    /// Create a zero state-space system, which blocks all inputs.
    pub fn zeros() -> Self {
        Self {
            a: SMatrix::zeros(),
            b: SMatrix::zeros(),
            c: SMatrix::zeros(),
            d: SMatrix::zeros(),
            state: SVector::zeros(),
        }
    }

    /// Create a state-space system with the provided A, B, C and D matrices.
    pub fn new(
        a: SMatrix<T, STATE, STATE>,
        b: SMatrix<T, STATE, IN>,
        c: SMatrix<T, OUT, STATE>,
        d: SMatrix<T, OUT, IN>,
    ) -> Self {
        Self {
            a,
            b,
            c,
            d,
            state: SVector::zeros(),
        }
    }
}

impl<T: Copy + nalgebra::Scalar, const IN: usize, const STATE: usize, const OUT: usize>
    StateSpace<T, IN, STATE, OUT>
{
    /// Update the matrices of this state space instance by copying them from another instance.
    /// This is useful to be able to reuse constructors as a mean to fully update the state space.
    pub fn update_matrices(&mut self, other: &Self) {
        self.a = other.a;
        self.b = other.b;
        self.c = other.c;
        self.d = other.d;
    }
}

impl<T: Scalar, const IN: usize, const STATE: usize, const OUT: usize> DSPMeta
    for StateSpace<T, IN, STATE, OUT>
{
    type Sample = T;
}

impl<T: Scalar, const IN: usize, const STATE: usize, const OUT: usize> DSPProcess<IN, OUT>
    for StateSpace<T, IN, STATE, OUT>
{
    fn process(&mut self, x: [Self::Sample; IN]) -> [Self::Sample; OUT] {
        let x = SVector::from(x);
        let y = self.c * self.state + self.d * x;
        self.state = self.a * self.state + self.b * x;
        y.into()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::ComplexField;
    use numeric_literals::replace_float_literals;

    use crate::dsp::buffer::AudioBuffer;
    use crate::dsp::DSPProcessBlock;

    use super::*;

    struct RC<T: nalgebra::Scalar>(StateSpace<T, 1, 1, 1>);

    impl<T: Scalar> DSPMeta for RC<T> {
        type Sample = T;
    }

    impl<T: Scalar> DSPProcess<1, 1> for RC<T> {
        fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
            self.0.process(x)
        }
    }

    impl<T: Scalar> RC<T> {
        #[replace_float_literals(T::from_f64(literal))]
        fn new(fc: T) -> Self {
            let new = SMatrix::<_, 1, 1>::new;
            Self(StateSpace {
                a: new(-(fc - 2.0) / (fc + 2.0)),
                b: new(1.0),
                c: new(-fc * (fc - 2.0) / (fc + 2.0).simd_powi(2) + fc / (fc + 2.0)),
                d: new(fc / (fc + 2.0)),
                ..StateSpace::zeros()
            })
        }
    }

    #[test]
    fn test_rc_filter() {
        let mut filter = RC::new(0.25);
        let mut input = AudioBuffer::zeroed(1024);
        let mut output = input.clone();
        input[0][0] = 1.0;
        filter.process_block(input.as_ref(), output.as_mut());
        insta::assert_csv_snapshot!(output.get_channel(0), { "[]" => insta::rounded_redaction(3) });
    }

    #[test]
    fn test_rc_filter_hz() {
        let filter = RC::new(0.25);
        let freq_response: [_; 512] = std::array::from_fn(|i| i as f64)
            .map(|f| filter.0.freq_response(1024.0, f)[0][0].abs());
        insta::assert_csv_snapshot!(&freq_response as &[_], { "[]" => insta::rounded_redaction(3)})
    }
}
