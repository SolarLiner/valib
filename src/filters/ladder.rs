//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use crate::{saturators::Saturator, Scalar, dsp::{DSP, analysis::DspAnalysis}};
use nalgebra::{SVector, Complex};
use numeric_literals::replace_float_literals;
use std::fmt;

#[derive(Debug, Copy, Clone)]
pub struct Ladder<T, S> {
    g: T,
    s: SVector<T, 4>,
    sat: [S; 4],
    k: T,
    pub compensated: bool,
}

impl<T: Scalar, S> Ladder<T, S> {
    pub fn new(samplerate: T, cutoff: T, resonance: T) -> Self where S: Default {
        let mut this = Self {
            g: T::zero(),
            s: SVector::zeros(),
            sat: std::array::from_fn(|_| S::default()),
            k: resonance,
            compensated: false,
        };
        this.set_cutoff(samplerate, cutoff);
        this
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_cutoff(&mut self, samplerate: T, frequency: T) {
        self.g = frequency / (2.0 * samplerate);
    }

    pub fn set_resonance(&mut self, k: T) {
        self.k = k;
    }
}

impl<T: Scalar + fmt::Debug, S: Saturator<T>> DSP<1, 1> for Ladder<T, S> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let input_gain = if self.compensated {
            self.k + 1.0
        } else {
            1.0
        };
        let x = input_gain * x[0];
        let y0 = x - self.k * self.s[3];
        let yd = SVector::<T, 4>::from_column_slice(&[
            y0 - self.s[0],
            self.s[0] - self.s[1],
            self.s[1] - self.s[2],
            self.s[2] - self.s[3],
        ]);
        let sout = SVector::<T, 4>::from_fn(|i, _| self.sat[i].saturate(yd[i]));
        let y = sout * self.g + self.s;
        self.s = y;
        for (i,s) in self.sat.iter_mut().enumerate() {
            s.update_state(yd[i], sout[i]);
        }
        [y[3]]
    }

    fn latency(&self) -> usize {
        4
    }

    fn reset(&mut self) {
        self.s = SVector::zeros();
    }
}

impl<T: Scalar, S: Saturator<T>> DspAnalysis<1, 1> for Ladder<T, S> {
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, z: [nalgebra::Complex<Self::Sample>; 1]) -> [nalgebra::Complex<Self::Sample>; 1] {
        let input_gain = if self.compensated {
            Complex::from(self.k) + 1.0
        } else {
            1.0
        };
        let z = input_gain * z[0];
        let lp = z * self.g / (z - 1.0);
        let ff = lp.powi(4);
        [ff / (1.0 + ff * self.k)]
    }
}

#[cfg(test)]
mod tests {
    use crate::{saturators::Linear, dsp::{DSPBlock, utils::{slice_to_mono_block, slice_to_mono_block_mut}}};

    use super::*;

    #[test]
    fn test_ladder_lp() {
        let mut filter = Ladder::<_, Linear>::new(1024.0, 200.0, 2.0);
        filter.compensated = true;
        let mut input = [1.0; 1024];
        let mut output = [0.0; 1024];
        input[0] = 0.0;
        filter.process_block(slice_to_mono_block(&input), slice_to_mono_block_mut(&mut output));

        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(3) })
    }
}