//! Linear state-space model implementation for arbitrary I/O.
//!
//! # Example
//!
//! ```rust
//! use nalgebra::SMatrix;
//! use valib_core::dsp::DSPProcess;
//! use valib_filters::statespace::StateSpace;
//! use valib_core::Scalar;
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
use nalgebra::{Complex, SMatrix, SVector, SimdComplexField};
use num_traits::Zero;

use valib_core::dsp::{analysis::DspAnalysis, DSPMeta, DSPProcess};
use valib_core::Scalar;
use valib_saturators::{Linear, MultiSaturator};

/// Linear discrete state-space method implementation with direct access to the state space matrices.
#[derive(Debug, Copy, Clone)]
pub struct StateSpace<
    T: Scalar,
    const IN: usize,
    const STATE: usize,
    const OUT: usize,
    S: MultiSaturator<T, STATE> = Linear,
> {
    /// Internal state matrix
    pub a: SMatrix<T, STATE, STATE>,
    /// Input -> state matrix
    pub b: SMatrix<T, STATE, IN>,
    /// State -> output matrix
    pub c: SMatrix<T, OUT, STATE>,
    /// Input -> Output matriw
    pub d: SMatrix<T, OUT, IN>,
    state: SVector<T, STATE>,
    saturators: S,
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
                .for_each(|v| *v = Complex::from_simd_real(<T as Scalar>::from_f64(f64::NAN)));
        }

        let h = c * zia * b + d;
        h.into()
    }
}

impl<T: Scalar + Zero, const IN: usize, const STATE: usize, const OUT: usize>
    StateSpace<T, IN, STATE, OUT, Linear>
{
    /// Create a zero state-space system, which blocks all inputs.
    pub fn zeros() -> Self {
        Self {
            a: SMatrix::zeros(),
            b: SMatrix::zeros(),
            c: SMatrix::zeros(),
            d: SMatrix::zeros(),
            state: SVector::zeros(),
            saturators: Linear,
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
            saturators: Linear,
        }
    }
}

impl<
        T: Scalar,
        const IN: usize,
        const STATE: usize,
        const OUT: usize,
        S: MultiSaturator<T, STATE>,
    > StateSpace<T, IN, STATE, OUT, S>
{
    /// Update the matrices of this state space instance by copying them from another instance.
    /// This is useful to be able to reuse constructors as a mean to fully update the state space.
    pub fn update_matrices<S2: MultiSaturator<T, STATE>>(
        &mut self,
        other: &StateSpace<T, IN, STATE, OUT, S2>,
    ) {
        self.a = other.a;
        self.b = other.b;
        self.c = other.c;
        self.d = other.d;
    }

    /// Replace the state saturators with the given ones
    ///
    /// # Arguments
    ///
    /// * `saturators`: New multi-saturator
    ///
    /// returns: StateSpace<T, { IN }, { STATE }, { OUT }, S2>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn with_saturators<S2: MultiSaturator<T, STATE>>(
        self,
        saturators: S2,
    ) -> StateSpace<T, IN, STATE, OUT, S2> {
        let Self {
            a, b, c, d, state, ..
        } = self;
        StateSpace {
            a,
            b,
            c,
            d,
            state,
            saturators,
        }
    }
}

impl<
        T: Scalar,
        S: MultiSaturator<T, STATE>,
        const IN: usize,
        const STATE: usize,
        const OUT: usize,
    > DSPMeta for StateSpace<T, IN, STATE, OUT, S>
{
    type Sample = T;
}

#[profiling::all_functions]
impl<
        T: Scalar,
        S: MultiSaturator<T, STATE>,
        const IN: usize,
        const STATE: usize,
        const OUT: usize,
    > DSPProcess<IN, OUT> for StateSpace<T, IN, STATE, OUT, S>
{
    fn process(&mut self, x: [Self::Sample; IN]) -> [Self::Sample; OUT] {
        let x = SVector::from(x);
        let y = self.c * self.state + self.d * x;
        self.state = self.a * self.state + self.b * x;
        let ys = self.saturators.multi_saturate(self.state.into());
        self.saturators.update_state_multi(self.state.into(), ys);
        self.state = SVector::from(ys);
        y.into()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::ComplexField;
    use numeric_literals::replace_float_literals;

    use valib_core::dsp::buffer::AudioBuffer;
    use valib_core::dsp::{BlockAdapter, DSPProcessBlock};

    use super::*;

    struct RC<T: Scalar>(StateSpace<T, 1, 1, 1>);

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
        let mut filter = BlockAdapter(RC::new(0.25));
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
