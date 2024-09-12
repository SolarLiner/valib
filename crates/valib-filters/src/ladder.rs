//! Implementation of various blocks of DSP code from the VA Filter Design book.
//!
//! Downloaded from <https://www.discodsp.net/VAFilterDesign_2.1.2.pdf>
//!
//! # Example
//!
//! ```rust
//! use valib_core::dsp::DSPProcess;
//! use valib_filters::ladder::{Ladder, OTA};
//! use valib_saturators::Tanh;
//! let mut filter = Ladder::<f32, OTA<Tanh>>::new(44100.0, 300.0, 0.5);
//! let output = filter.process([0.0]);
//! ```

use std::fmt;

use nalgebra::{Complex, SVector};
use numeric_literals::replace_float_literals;
use valib_core::dsp::analysis::DspAnalysis;
use valib_core::dsp::parameter::HasParameters;
use valib_core::dsp::DSPMeta;
use valib_core::dsp::{parameter::ParamId, parameter::ParamName, DSPProcess};
use valib_core::math::bilinear_prewarming_bounded;
use valib_core::Scalar;
use valib_saturators::{Saturator, Tanh};

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

#[profiling::all_functions]
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

#[profiling::all_functions]
impl<T: Scalar, S: Default + Saturator<T>> LadderTopology<T> for OTA<S> {
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

#[profiling::all_functions]
impl<T: Scalar, S: Default + Saturator<T>> LadderTopology<T> for Transistor<S> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ParamName)]
pub enum LadderParams {
    Cutoff,
    Resonance,
}

/// Ladder filter. This [`DSPProcess`] instance implements a saturated 4-pole lowpass filter, with feedback negatively added
/// back into the input.
#[derive(Debug, Copy, Clone)]
pub struct Ladder<T, Topo = OTA<Tanh>> {
    wc: T,
    samplerate: T,
    inv_2fs: T,
    s: SVector<T, 4>,
    topology: Topo,
    k: T,
    /// Whether or not the DC gain loss due to higher resonance values is compensated.
    pub compensated: bool,
}

impl<T: Scalar, Topo: LadderTopology<T>> HasParameters for Ladder<T, Topo> {
    type Name = LadderParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            LadderParams::Cutoff => self.set_cutoff(T::from_f64(value as _)),
            LadderParams::Resonance => self.set_resonance(T::from_f64(value as _)),
        }
    }
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
    /// use valib_filters::ladder::{Ideal, Ladder, OTA, Transistor};
    /// use valib_saturators::clippers::DiodeClipperModel;
    /// use valib_saturators::Tanh;
    /// let ota_ladder = Ladder::<_, OTA<Tanh>>::new(48000.0, 440.0, 1.0);
    /// let ideal_ladder = Ladder::<_, Ideal>::new(48000.0, 440.0, 1.0);
    /// let transistor_ladder = Ladder::<_, Transistor<DiodeClipperModel<_>>>::new(48000.0, 440.0, 1.0);
    /// ```
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: impl Into<f64>, cutoff: T, resonance: T) -> Self {
        let samplerate = T::from_f64(samplerate.into());
        let mut this = Self {
            inv_2fs: T::simd_recip(2.0 * samplerate),
            samplerate,
            wc: cutoff,
            s: SVector::zeros(),
            topology: Topo::default(),
            k: resonance,
            compensated: false,
        };
        this.set_cutoff(cutoff);
        this
    }

    pub fn with_topology<T2>(self, topology: T2) -> Ladder<T, T2> {
        let Self {
            inv_2fs,
            samplerate,
            wc: fc,
            s,
            k,
            compensated,
            ..
        } = self;
        Ladder {
            inv_2fs,
            samplerate,
            wc: fc,
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
    pub fn set_cutoff(&mut self, frequency: T) {
        self.wc = bilinear_prewarming_bounded(
            self.samplerate,
            T::from_f64(2.0) * T::simd_two_pi() * frequency,
        );
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

impl<T: Scalar, Topo: LadderTopology<T>> DSPMeta for Ladder<T, Topo> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = T::from_f64(samplerate as _);
        self.inv_2fs = T::simd_recip(self.samplerate + self.samplerate);
    }

    fn latency(&self) -> usize {
        4
    }

    fn reset(&mut self) {
        self.s = SVector::zeros();
    }
}

fn quad_falloff<T: Scalar>(t: T) -> T {
    T::simd_powi(T::one() - t.simd_clamp(T::zero(), T::one()), 2)
}

#[profiling::all_functions]
impl<T: Scalar + fmt::Debug, Topo: LadderTopology<T>> DSPProcess<1, 1> for Ladder<T, Topo> {
    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let input_gain = if self.compensated { self.k + 1.0 } else { 1.0 };
        let x = input_gain * x[0];
        let q_correction = quad_falloff(self.wc * self.inv_2fs / T::simd_two_pi());
        let y0 = x - self.k * self.s[3] * (q_correction);
        let g = self.wc * self.inv_2fs;
        self.s = self.topology.next_output(g, y0, self.s);
        [self.s[3]]
    }
}

impl<T: Scalar, Topo: LadderTopology<T>> DspAnalysis<1, 1> for Ladder<T, Topo> {
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 1] {
        let input_gain = if self.compensated {
            (Complex::from(self.k) + 1.0) * 0.707_945_784
        } else {
            1.0
        };
        let g = self.wc * self.inv_2fs;
        let lp = z * g / (z - 1.0);
        let ff = lp.powi(4);
        [[input_gain * ff / (1.0 - ff * self.k)]]
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use valib_core::simd::SimdComplexField;

    use valib_core::{
        dsp::{buffer::AudioBuffer, BlockAdapter, DSPProcessBlock},
        util::tests::{Plot, Series},
    };
    use valib_saturators::clippers::DiodeClipperModel;
    use valib_saturators::Tanh;

    use super::*;

    #[rstest]
    fn test_ladder_ir<Topo: LadderTopology<f64>>(
        #[values(Ideal, OTA([Tanh; 4]), Transistor([DiodeClipperModel::new_silicon(1, 1); 5]))]
        topology: Topo,
        #[values(false, true)] compensated: bool,
        #[values(0.0, 0.1, 0.5, 1.0)] resonance: f64,
    ) {
        use plotters::prelude::*;

        let samplerate = 4096.0;
        let mut filter = BlockAdapter(
            Ladder::<f64, Ideal>::new(samplerate, 100.0, resonance).with_topology::<Topo>(topology),
        );
        filter.0.compensated = compensated;
        let input = AudioBuffer::new([std::iter::once(0.0)
            .chain(std::iter::repeat(1.0))
            .take(1024)
            .collect::<Box<_>>()])
        .unwrap();
        let mut output = AudioBuffer::zeroed(1024);
        filter.process_block(input.as_ref(), output.as_mut());

        let topo = std::any::type_name::<Topo>()
            .replace("::", "__")
            .replace(['<', '>'], "_");
        let name = format!("test_ladder_ir_{topo}_c{compensated}_r{resonance}");
        let plot_title = format!("Ladder {topo} c={compensated} r={resonance}");
        let inp_f32 = input
            .get_channel(0)
            .iter()
            .copied()
            .map(|x| x as f32)
            .collect::<Box<[_]>>();
        let out_f32 = output
            .get_channel(0)
            .iter()
            .copied()
            .map(|x| x as f32)
            .collect::<Box<[_]>>();
        Plot {
            title: &plot_title,
            bode: false,
            series: &[
                Series {
                    label: "Input",
                    samplerate: samplerate as f32,
                    series: &*inp_f32,
                    color: &BLUE,
                },
                Series {
                    label: "Output",
                    samplerate: samplerate as f32,
                    series: &*out_f32,
                    color: &RED,
                },
            ],
        }
        .create_svg(format!("plots/ladder/{name}.svg"));
        insta::assert_csv_snapshot!(name, output.get_channel(0), { "[]" => insta::rounded_redaction(3) })
    }

    #[rstest]
    fn test_ladder_hz(
        #[values(false, true)] compensated: bool,
        #[values(0.0, 0.1, 0.2, 0.5, 1.0)] resonance: f64,
    ) {
        use plotters::prelude::*;
        use valib_core::util::tests::Series;

        let samplerate = 1024.0;
        let mut filter = Ladder::<_, Ideal>::new(samplerate, 200.0, resonance);
        filter.compensated = compensated;
        let response: [_; 511] = std::array::from_fn(|i| (i + 1) as f64)
            .map(|f| filter.freq_response(1024.0, f)[0][0].simd_abs());
        let response_db = response.map(|x| 20.0 * x.log10());
        let responsef32 = response.map(|x| x as f32);

        let name = format!("test_ladder_hz_c{compensated}_r{resonance}");
        let plot_title = format!("Ladder filter: compensated={compensated}, resonance={resonance}");
        Plot {
            title: &plot_title,
            bode: true,
            series: &[Series {
                label: "Frequency response",
                samplerate,
                series: &responsef32,
                color: &BLUE,
            }],
        }
        .create_svg(format!(
            "plots/ladder/freq_response__c{compensated}__r{resonance}.svg"
        ));
        insta::assert_csv_snapshot!(name, &response_db as &[_], { "[]" => insta::rounded_redaction(3) })
    }
}
