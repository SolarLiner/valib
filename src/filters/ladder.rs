//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use std::fmt;

use nalgebra::{Complex, SVector, SimdComplexField};
use numeric_literals::replace_float_literals;

use crate::{
    dsp::{analysis::DspAnalysis, DSP},
    math::bilinear_prewarming_bounded,
    saturators::{Saturator, Tanh},
    Scalar,
};

/// Ladder topology struct. Internal state of the saturators will be held within instances of this trait.
pub trait LadderTopology<T>: Default {
    /// Provide the next output vector, given the input sample, last output vector and the current normalized angular
    /// frequency.
    ///
    /// # Arguments
    ///
    /// * `wc`: Angular frequency
    /// * `y0`: Input sample coming into the filter
    /// * `y`: Last output vector
    ///
    /// returns the next output vector for each integrator.
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4>;
}

/// Ideal ladder topology, no nonlinearities per se, just a hard clipping of the output to prevent runaway feedback.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ideal;

impl<T: Scalar> LadderTopology<T> for Ideal {
    #[replace_float_literals(T::from_f64(literal))]
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4> {
        let yd = SVector::from([y[0] - y0, y[1] - y[0], y[2] - y[1], y[3] - y[2]])
            .map(|x| x.simd_clamp(-1.0, 1.0));
        y - yd * wc
    }
}

/// Operational Transconductance Amplifier ladder topology. For aiming at realism, the [`Saturator`] instance should be
/// [`Tanh`], which mimics the output saturation of an OTA chip.
#[derive(Debug, Clone, Copy, Default)]
pub struct OTA<S>(pub [S; 4]);

impl<T: Scalar, S: Saturator<T>> LadderTopology<T> for OTA<S> {
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4> {
        let yd = SVector::from([y[0] - y0, y[1] - y[0], y[2] - y[1], y[3] - y[2]]);
        let sout = SVector::from_fn(|i, _| self.0[i].saturate(yd[i]));
        for (i, s) in self.0.iter_mut().enumerate() {
            s.update_state(yd[i], sout[i]);
        }
        y - sout * wc
    }
}

/// Transistor ladder, the most famous topology in synth history.
#[derive(Debug, Clone, Copy, Default)]
pub struct Transistor<S>(pub [S; 5]);

impl<T: Scalar, S: Saturator<T>> LadderTopology<T> for Transistor<S> {
    fn next_output(&mut self, wc: T, y0: T, y: SVector<T, 4>) -> SVector<T, 4> {
        let y0sat = wc * self.0[4].saturate(y0);
        let ysat = SVector::<_, 4>::from_fn(|i, _| wc * self.0[i].saturate(y[i]));
        let yd = SVector::from([
            ysat[0] - y0sat,
            ysat[1] - ysat[0],
            ysat[2] - ysat[1],
            ysat[3] - ysat[2],
        ]);
        for (s, (y, ysat)) in self
            .0
            .iter_mut()
            .zip(y.iter().copied().zip(ysat.iter().copied()))
        {
            s.update_state(y, ysat);
        }
        self.0[4].update_state(y0, y0sat);
        y - yd
    }
}

/// Ladder filter. This [`DSP`] instance implements a saturated 4-pole lowpass filter, with feedback negatively added
/// back into the input.
#[derive(Debug, Copy, Clone)]
pub struct Ladder<T, Topo = OTA<Tanh>> {
    g: T,
    s: SVector<T, 4>,
    topology: Topo,
    k: T,
    /// Whether or not the DC gain loss due to higher resonance values is compensated.
    pub compensated: bool,
}

impl<T: Scalar, Topo: LadderTopology<T>> Ladder<T, Topo> {
    /// Create a new instance of this filter.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Signal sampling rate (Hz)
    /// * `cutoff`: Cutoff frequency (Hz)
    /// * `resonance`: Resonance amount (0..4)
    ///
    /// returns: Ladder<T, Topo>
    ///
    /// # Examples
    ///
    /// ```
    /// use valib::filters::ladder::{Ideal, Ladder, OTA, Transistor};
    /// use valib::saturators::clippers::DiodeClipperModel;
    /// use valib::saturators::Tanh;
    /// let ota_ladder = Ladder::<_, OTA<Tanh>>::new(48000.0, 440.0, 1.0);
    /// let ideal_ladder = Ladder::<_, Ideal>::new(48000.0, 440.0, 1.0);
    /// let transistor_ladder = Ladder::<_, Transistor<DiodeClipperModel<_>>>::new(48000.0, 440.0, 1.0);
    /// ```
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

    pub fn with_topology<T2>(self, topology: T2) -> Ladder<T, T2> {
        let Self {
            g,
            s,
            k,
            compensated,
            ..
        } = self;
        Ladder {
            g,
            s,
            k,
            compensated,
            topology,
        }
    }

    /// Sets the cutoff frequency of the filter. Uses bounded prewarping to help setting the normalized cutoff frequency
    /// at the right frequency.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Signal sampling rate (Hz)
    /// * `frequency`: Cutoff frequency (Hz)
    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_cutoff(&mut self, samplerate: T, frequency: T) {
        let wc = bilinear_prewarming_bounded(samplerate, T::simd_two_pi() * frequency);
        self.g = wc / (2.0 * samplerate);
    }

    /// Sets the resonance amount.
    ///
    /// # Arguments
    ///
    /// * `k`: Resonance (0.., starts self-oscillation at 4)
    pub fn set_resonance(&mut self, k: T) {
        self.k = k;
    }
}

impl<T: Scalar + fmt::Debug, Topo: LadderTopology<T>> DSP<1, 1> for Ladder<T, Topo> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let input_gain = if self.compensated { self.k + 1.0 } else { 1.0 };
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
    fn h_z(&self, z: nalgebra::Complex<Self::Sample>) -> [[nalgebra::Complex<Self::Sample>; 1]; 1] {
        let input_gain = if self.compensated {
            (Complex::from(self.k) + 1.0) * 0.707_945_784
        } else {
            1.0
        };
        let lp = z * self.g / (z - 1.0);
        let ff = lp.powi(4);
        [[input_gain * ff / (1.0 - ff * self.k)]]
    }
}

#[cfg(test)]
mod tests {
    use num_traits::real::Real;
    use rstest::rstest;

    use crate::dsp::{
        utils::{slice_to_mono_block, slice_to_mono_block_mut},
        DSPBlock,
    };
    use crate::saturators::clippers::DiodeClipperModel;

    use super::*;

    #[rstest]
    fn test_ladder_ir<Topo: LadderTopology<f64>>(
        #[values(Ideal, OTA([Tanh; 4]), Transistor([DiodeClipperModel::new_silicon(1, 1); 5]))]
        topology: Topo,
        #[values(false, true)] compensated: bool,
        #[values(0.0, 0.1, 0.5, 1.0)] resonance: f64,
    ) {
        let mut filter =
            Ladder::<f64, Ideal>::new(1024.0, 200.0, resonance).with_topology::<Topo>(topology);
        filter.compensated = compensated;
        let mut input = [1.0; 1024];
        let mut output = [0.0; 1024];
        input[0] = 0.0;
        filter.process_block(
            slice_to_mono_block(&input),
            slice_to_mono_block_mut(&mut output),
        );

        let topo = std::any::type_name::<Topo>()
            .replace("::", "__")
            .replace('<', "_")
            .replace('>', "_");
        let name = format!("test_ladder_ir_{topo}_c{compensated}_r{resonance}");
        insta::assert_csv_snapshot!(name, &output as &[_], { "[]" => insta::rounded_redaction(3) })
    }

    #[rstest]
    fn test_ladder_hz(
        #[values(false, true)] compensated: bool,
        #[values(0.0, 0.1, 0.2, 0.5, 1.0)] resonance: f64,
    ) {
        let mut filter = Ladder::<_, Ideal>::new(1024.0, 200.0, resonance);
        filter.compensated = compensated;
        let response: [_; 512] = std::array::from_fn(|i| i as f64)
            .map(|f| filter.freq_response(1024.0, f)[0][0].simd_abs())
            .map(|x| 20.0 * x.log10());

        let name = format!("test_ladder_hz_c{compensated}_r{resonance}");
        insta::assert_csv_snapshot!(name, &response as &[_], { "[]" => insta::rounded_redaction(3) })
    }
}
