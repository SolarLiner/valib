//! Implementation of various blocks of DSP code from the VA Filter Design book.
//!
//! Downloaded from <https://www.discodsp.net/VAFilterDesign_2.1.2.pdf>
//! All references in this module, unless specified otherwise, are taken from this book.

use nalgebra::Complex;
use num_traits::One;
use numeric_literals::replace_float_literals;
use valib_core::dsp::{
    analysis::DspAnalysis,
    parameter::{HasParameters, ParamId, ParamName},
    DSPMeta, DSPProcess,
};
use valib_core::Scalar;
use valib_saturators::{Linear, Saturator};

/// Parameter type for the SVF filter
#[derive(Debug, Clone, Copy, PartialEq, Eq, ParamName)]
pub enum SvfParams {
    /// Cutoff frequency (Hz)
    Cutoff,
    /// Resonance
    Resonance,
}

/// SVF topology filter, with optional non-linearities.
#[derive(Debug, Copy, Clone)]
pub struct Svf<T, Mode = Linear> {
    s: [T; 2],
    r: T,
    fc: T,
    g: T,
    g1: T,
    d: T,
    w_step: T,
    samplerate: T,
    saturator: Mode,
}

impl<T: Scalar, Mode: Saturator<T>> HasParameters for Svf<T, Mode> {
    type Name = SvfParams;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        let value = T::from_f64(value as _);
        match param {
            SvfParams::Cutoff => self.set_cutoff(value),
            SvfParams::Resonance => self.set_r(value),
        }
    }
}

impl<T: Scalar, Mode: Saturator<T>> DSPMeta for Svf<T, Mode> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.w_step = T::simd_pi() / T::from_f64(samplerate as _);
        self.update_coefficients();
    }

    fn latency(&self) -> usize {
        2
    }

    fn reset(&mut self) {
        self.s.fill(T::zero());
    }
}

#[profiling::all_functions]
impl<T: Scalar, S: Saturator<T>> DSPProcess<1, 3> for Svf<T, S> {
    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 3] {
        let [s1, s2] = self.s;

        let bpp = self.saturator.saturate(s1);
        let bpl = (self.r - 1.) * s1;
        let bp1 = 2. * (bpp + bpl);
        let hp = (x[0] - bp1 - s2) * self.d;
        self.saturator.update_state(s1, bpp);

        let v1 = self.g * hp;
        let bp = v1 + s1;
        let s1 = bp + v1;

        let v2 = self.g * bp;
        let lp = v2 + s2;
        let s2 = lp + v2;

        self.s = [s1, s2];
        [lp, bp, hp]
    }
}

impl<T: Scalar, S: Saturator<T>> DspAnalysis<1, 3> for Svf<T, S> {
    #[replace_float_literals(T::from_f64(literal))]
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 3]; 1] {
        let omega_c = 2.0 * self.fc * self.w_step;
        let x0 = z + Complex::one();
        let x1 = x0.powi(2) * omega_c.simd_powi(2);
        let x2 = z - Complex::one();
        let x3 = x2.powi(2) * 4.0;
        let x4 = x0 * x2 * omega_c;
        let x5 = Complex::<T>::one() / (-x4 * 4.0 * self.r + x1 + x3);
        [[x1 * x5, -x4 * x5 * 2.0, x3 * x5]]
    }
}

impl<T: Scalar> Svf<T, Linear> {
    /// Create a new SVF filter with the provided sample rate, frequency cutoff (in Hz) and resonance amount
    /// (in 0..1 for stable filters, otherwise use bounded nonlinearities).
    pub fn new(samplerate: T, fc: T, r: T) -> Self {
        let mut this = Self {
            s: [T::zero(); 2],
            r: r + r,
            fc,
            g: T::zero(),
            g1: T::zero(),
            d: T::zero(),
            samplerate,
            w_step: T::simd_pi() / samplerate,
            saturator: Linear,
        };
        this.update_coefficients();
        this
    }
}

impl<T: Scalar, C> Svf<T, C> {
    /// Set the new filter cutoff frequency (in Hz).
    pub fn set_cutoff(&mut self, freq: T) {
        self.fc = freq;
        self.update_coefficients();
    }

    /// Set the resonance amount (in 0..1 for stable filters, otherwise use bounded nonlinearities).
    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_r(&mut self, r: T) {
        self.r = 2. * r;
        self.update_coefficients();
    }

    #[profiling::function]
    #[replace_float_literals(T::from_f64(literal))]
    fn update_coefficients(&mut self) {
        self.g = self.w_step * self.fc;
        self.g1 = 2. * self.r + self.g;
        self.d = (1. + 2. * self.r * self.g + self.g * self.g).simd_recip();
    }
}

impl<T: Scalar, S: Saturator<T>> Svf<T, S> {
    /// Apply these new saturators to this SVF instance, returning a new instance of it.
    pub fn set_saturator(&mut self, sat: S) {
        self.saturator = sat;
    }

    /// Replace the saturators in this Biquad instance with the provided values.
    pub fn with_saturator<S2: Saturator<T>>(self, saturator: S2) -> Svf<T, S2> {
        let Self {
            s,
            r,
            fc,
            g,
            g1,
            d,
            w_step,
            samplerate,
            ..
        } = self;
        Svf {
            s,
            r,
            fc,
            g,
            g1,
            d,
            w_step,
            samplerate,
            saturator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::ComplexField;
    use plotters::prelude::*;
    use valib_core::util::tests::*;

    #[test]
    fn test_svf_hz() {
        const SAMPLERATE: f64 = 1024.0;
        const FC: f64 = 10.0;
        let filter = Svf::<_, Linear>::new(SAMPLERATE, FC, 0.15);
        let hz: [_; 512] = std::array::from_fn(|i| i as f64)
            .map(|f| filter.freq_response(SAMPLERATE, f)[0].map(|c| c.abs()));
        let lp = hz.map(|[lp, _, _]| lp as _);
        let bp = hz.map(|[_, bp, _]| bp as _);
        let hp = hz.map(|[_, _, hp]| hp as _);
        Plot {
            title: "SVF Frequency Response",
            bode: true,
            series: &[
                Series {
                    label: "Low pass",
                    color: &BLUE,
                    samplerate: SAMPLERATE as _,
                    series: &lp,
                },
                Series {
                    label: "Band pass",
                    color: &GREEN,
                    samplerate: SAMPLERATE as _,
                    series: &bp,
                },
                Series {
                    label: "High pass",
                    color: &full_palette::ORANGE,
                    samplerate: SAMPLERATE as _,
                    series: &hp,
                },
            ],
        }
        .create_svg("plots/svf/freq_response_hz.svg");
        insta::assert_csv_snapshot!(&hz as &[_], { "[][]" => insta::rounded_redaction(3)})
    }
}
