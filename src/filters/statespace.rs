use nalgebra::{SMatrix, SVector};
use num_traits::Zero;
use crate::dsp::DSP;
use crate::Scalar;

#[derive(Debug, Copy, Clone)]
pub struct StateSpace<T: nalgebra::Scalar, const IN: usize, const STATE: usize, const OUT: usize> {
    pub a: SMatrix<T, STATE, STATE>,
    pub b: SMatrix<T, STATE, IN>,
    pub c: SMatrix<T, OUT, STATE>,
    pub d: SMatrix<T, OUT, IN>,
    state: SVector<T, STATE>,
}

impl<T: nalgebra::Scalar + Zero, const IN: usize, const STATE: usize, const OUT: usize> StateSpace<T, IN, STATE, OUT> {
    pub fn zeros() -> Self {
        Self {
            a: SMatrix::zeros(),
            b: SMatrix::zeros(),
            c: SMatrix::zeros(),
            d: SMatrix::zeros(),
            state: SVector::zeros(),
        }
    }
}

impl<T: Copy + nalgebra::Scalar, const IN: usize, const STATE: usize, const OUT: usize> StateSpace<T, IN, STATE, OUT> {
    pub fn update_matrices(&mut self, other: &Self) {
        self.a = other.a;
        self.b = other.b;
        self.c = other.c;
        self.d = other.d;
    }
}

impl<T: Scalar, const IN: usize, const STATE: usize, const OUT: usize> DSP<IN, OUT> for StateSpace<T, IN, STATE, OUT> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; IN]) -> [Self::Sample; OUT] {
        let x = SVector::from(x);
        let y = self.c * self.state + self.d * x;
        self.state = self.a * self.state + self.b * x;
        y.into()
    }
}

#[cfg(test)]
mod tests {
    use numeric_literals::replace_float_literals;
    use crate::dsp::{DSPBlock, utils::{slice_to_mono_block, slice_to_mono_block_mut}};

    use super::*;

    struct RC<T: nalgebra::Scalar>(StateSpace<T, 1, 1, 1>);

    impl<T: Scalar> DSP<1, 1> for RC<T> {
        type Sample = T;

        fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
            self.0.process(x)
        }
    }

    impl<T: Scalar> RC<T> {
        #[replace_float_literals(T::from_f64(literal))]
        fn new(fc: T) -> Self {
            let new = SMatrix::<_, 1, 1>::new;
            Self(StateSpace {
                a: new(-(fc-2.0)/(fc+2.0)),
                b: new(1.0),
                c: new(-fc*(fc-2.0)/(fc+2.0).simd_powi(2) + fc/(fc+2.0)),
                d: new(fc / (fc + 2.0)),
                ..StateSpace::zeros()
            })
        }
    }

    #[test]
    fn test_rc_filter() {
        let mut filter = RC::new(0.25);
        let mut input = [0.0; 1024];
        let mut output = [0.0; 1024];
        input[0] = 1.0;
        filter.process_block(slice_to_mono_block(&input), slice_to_mono_block_mut(&mut output));
        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(3) });
    }
}