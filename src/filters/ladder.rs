//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use crate::{saturators::{Saturator, Tanh}, Scalar, dsp::{DSP, analysis::DspAnalysis}, math::bilinear_prewarming_bounded};
use nalgebra::{SVector, Complex, SimdComplexField};
use numeric_literals::replace_float_literals;
use std::{fmt, marker::PhantomData};

pub trait LadderTopology<T>: Default {
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Ideal;

impl<T: Scalar> LadderTopology<T> for Ideal {
    #[replace_float_literals(T::from_f64(literal))]
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4> {
        let yd = SVector::from([
            y[0] - y0,
            y[1] - y[0],
            y[2] - y[1],
            y[3] - y[2],
        ]).map(|x| x.simd_clamp(-1.0, 1.0));
        y - yd * wc
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OTA<S>([S; 4]);

impl<T: Scalar, S: Saturator<T>> LadderTopology<T> for OTA<S> {
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4> {
        let yd = SVector::from([
            y[0] - y0,
            y[1] - y[0],
            y[2] - y[1],
            y[3] - y[2],
        ]);
        let sout = SVector::from_fn(|i, _| self.0[i].saturate(yd[i]));
        for (i, s) in self.0.iter_mut().enumerate() {
            s.update_state(yd[i], sout[i]);
        }
        y - sout * wc
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Transistor<S>([S; 5]);

impl<T: Scalar, S: Saturator<T>> LadderTopology<T> for Transistor<S> {
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4> {
        let y0sat = wc * self.0[4].saturate(y0);
        let ysat = SVector::<_, 4>::from_fn(|i, _| wc * self.0[i].saturate(y[0]));
        let yd = SVector::from([
            ysat[0] - y0sat,
            ysat[1] - ysat[0],
            ysat[2] - ysat[1],
            ysat[3] - ysat[2],
        ]);
        for (i, s) in self.0.iter_mut().enumerate() {
            s.update_state(y[i], ysat[i]);
        }
        self.0[4].update_state(y0, y0sat);
        y - yd
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Ladder<T, Topo=OTA<Tanh>> {
    g: T,
    s: SVector<T, 4>,
    topology: Topo,
    k: T,
    pub compensated: bool,
}

impl<T: Scalar, Topo: LadderTopology<T>> Ladder<T, Topo> {
    pub fn new(samplerate: T, cutoff: T, resonance: T) -> Self {
        let mut this = Self {
            g: T::zero(),
            s: SVector::zeros(),
            topology: Topo::default(),
            k: resonance,
            compensated: false,
        };
        this.set_cutoff(samplerate, cutoff);
        this
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_cutoff(&mut self, samplerate: T, frequency: T) {
        let wc = bilinear_prewarming_bounded(samplerate, T::simd_two_pi() * frequency);
        self.g = wc / (2.0 * samplerate);
    }

    pub fn set_resonance(&mut self, k: T) {
        self.k = k;
    }
}

impl<T: Scalar + fmt::Debug, Topo: LadderTopology<T>> DSP<1, 1> for Ladder<T, Topo> {
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
        self.s = self.topology.next_output(self.g, y0, self.s);
        [self.s[3]]
    }

    fn latency(&self) -> usize {
        4
    }

    fn reset(&mut self) {
        self.s = SVector::zeros();
    }
}

impl<T: Scalar, Topo: LadderTopology<T>> DspAnalysis<1, 1> for Ladder<T, Topo> {
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, z: [nalgebra::Complex<Self::Sample>; 1]) -> [nalgebra::Complex<Self::Sample>; 1] {
        let input_gain = if self.compensated {
            (Complex::from(self.k) + 1.0) *  0.707_945_784
        } else {
            1.0
        };
        let z = z[0];
        let lp = z * self.g / (z - 1.0);
        let ff = lp.powi(4);
        [input_gain * ff / (1.0 - ff * self.k)]
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use crate::dsp::{DSPBlock, utils::{slice_to_mono_block, slice_to_mono_block_mut}};

    use super::*;

    #[test]
    fn test_ladder_lp() {
        let mut filter = Ladder::<_, Ideal>::new(1024.0, 200.0, 2.0);
        filter.compensated = true;
        let mut input = [1.0; 1024];
        let mut output = [0.0; 1024];
        input[0] = 0.0;
        filter.process_block(slice_to_mono_block(&input), slice_to_mono_block_mut(&mut output));

        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(3) })
    }

    #[test]
    fn test_ladder_hz() {
        let filter = Ladder::<_, Ideal>::new(1024.0, 200.0, 2.0);
        let response: [_; 512] = std::array::from_fn(|i| i as f64 / 1024.0).map(|f| filter.freq_response([TAU * f])[0].simd_abs()).map(|x| 20.0 * x.log10());

        insta::assert_csv_snapshot!(&response as &[_], { "[]" => insta::rounded_redaction(3) })
    }

    #[test]
    fn test_ladder_hz_compensated() {
        let mut filter = Ladder::<_, Ideal>::new(1024.0, 200.0, 2.0);
        filter.compensated = true;
        let response: [_; 512] = std::array::from_fn(|i| i as f64 / 1024.0).map(|f| filter.freq_response([TAU * f])[0].simd_abs()).map(|x| 20.0 * x.log10());

        insta::assert_csv_snapshot!(&response as &[_], { "[]" => insta::rounded_redaction(3) })
    }
}