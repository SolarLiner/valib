use std::sync::Arc;

use nih_plug::prelude::*;
use numeric_literals::replace_float_literals;
use realfft::num_complex::Complex;

use valib::{dsp::analysis::DspAnalysis, math::freq_to_z, saturators::clippers::DiodeClipperModel};
use valib::dsp::DSP;
use valib::filters::svf::Svf;
use valib::saturators::{Dynamic, Saturator};
use valib::Scalar;
use valib::simd::SimdValue;

use crate::Sample;

#[derive(Debug, Copy, Clone, Enum, Eq, PartialEq)]
pub enum FilterType {
    Bypass,
    Lowpass,
    Bandpass,
    Highpass,
    Lowshelf,
    Highshelf,
    PeakSharp,
    PeakShelf,
    Notch,
    Allpass,
}

impl FilterType {
    #[replace_float_literals(Complex::from(< T as Scalar >::from_f64(literal)))]
    pub(crate) fn h_z<T: Scalar + valib::simd::SimdRealField, S: Saturator<T>>(
        &self,
        samplerate: T,
        filter: &Svf<T, S>,
        amp: <T as SimdValue>::Element,
        z: Complex<T>,
    ) -> Complex<T> {
        let amp = Complex::from(T::splat(amp));
        let [[lp, bp, hp]] = filter.h_z(samplerate, z);
        match self {
            Self::Bypass => 1.0,
            Self::Lowpass => lp,
            Self::Bandpass => bp,
            Self::Highpass => hp,
            Self::PeakSharp => lp - hp,
            Self::PeakShelf => {
                let g = amp - 1.;
                1. + bp * g
            }
            Self::Notch => 1. - bp,
            Self::Allpass => 1.,
            Self::Lowshelf => 1. + lp * (amp - 1.),
            Self::Highshelf => 1. + hp * (amp - 1.),
        }
    }

    pub(crate) fn freq_response<T: Scalar + nalgebra::RealField, S: Saturator<T>>(
        &self,
        samplerate: T,
        filter: &Svf<T, S>,
        amp: <T as SimdValue>::Element,
        f: T,
    ) -> Complex<T> {
        self.h_z(samplerate, filter, amp, freq_to_z(samplerate, f))
    }

    #[replace_float_literals(Sample::from_f64(literal))]
    fn mix(&self, amp: Sample, x: Sample, [lp, bp, hp]: [Sample; 3]) -> Sample {
        match self {
            Self::Bypass => x,
            Self::Lowpass => lp,
            Self::Bandpass => bp,
            Self::Highpass => hp,
            Self::PeakSharp => lp - hp,
            Self::PeakShelf => {
                let g = amp - 1.;
                x + bp * g
            }
            Self::Notch => x - bp,
            Self::Allpass => x - 2. * bp,
            Self::Lowshelf => x + lp * (amp - 1.),
            Self::Highshelf => x + hp * (amp - 1.),
        }
    }
}

impl Default for FilterType {
    fn default() -> Self {
        Self::PeakShelf
    }
}

#[derive(Debug, Copy, Clone, Enum, Eq, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum ResonanceClip {
    Clean,
    I,
    II,
    III,
}

impl Default for ResonanceClip {
    fn default() -> Self {
        Self::I
    }
}

impl ResonanceClip {
    pub fn as_dynamic_type(&self) -> Dynamic<Sample> {
        match self {
            Self::Clean => Dynamic::Linear,
            Self::I => Dynamic::DiodeClipper(DiodeClipperModel::new_led(2, 3)),
            Self::II => Dynamic::DiodeClipper(DiodeClipperModel::new_germanium(1, 2)),
            Self::III => Dynamic::DiodeClipper(DiodeClipperModel::new_silicon(2, 1)),
        }
    }
}

#[derive(Debug, Params)]
pub struct FilterParams {
    #[id = "fc"]
    pub(crate) cutoff: FloatParam,
    #[id = "q"]
    pub(crate) q: FloatParam,
    #[id = "amp"]
    pub(crate) amp: FloatParam,
    #[id = "type"]
    pub(crate) ftype: EnumParam<FilterType>,
    #[id = "rclip"]
    pub(crate) resclip: EnumParam<ResonanceClip>,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            cutoff: FloatParam::new("Filter Cutoff", 300., Self::cutoff_range())
                .with_value_to_string(formatters::v2s_f32_hz_then_khz_with_note_name(2, false))
                .with_string_to_value(formatters::s2v_f32_hz_then_khz())
                .with_smoother(SmoothingStyle::Exponential(50.)),
            q: FloatParam::new(
                "Q",
                0.5,
                FloatRange::Skewed {
                    min: 0.,
                    max: 1.,
                    factor: FloatRange::skew_factor(1.5),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_value_to_string(formatters::v2s_f32_percentage(2))
            .with_unit(" %"),
            amp: FloatParam::new(
                "Gain",
                1.,
                FloatRange::Skewed {
                    min: 1e-2,
                    max: 100.,
                    factor: FloatRange::gain_skew_factor(-40., 40.),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_unit(" dB"),
            ftype: EnumParam::new("Filter Type", FilterType::default()),
            resclip: EnumParam::new("Resonance clipping", ResonanceClip::default()),
        }
    }
}

impl FilterParams {
    pub fn cutoff_range() -> FloatRange {
        FloatRange::Skewed {
            min: 20.,
            max: 20e3,
            factor: FloatRange::skew_factor(-2.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Filter {
    pub params: Arc<FilterParams>,
    pub scale: <Sample as SimdValue>::Element,
    svf: Svf<Sample, Dynamic<Sample>>,
    clippers: [Dynamic<Sample>; 3],
    last_resclip: ResonanceClip,
}

impl DspAnalysis<1, 1> for Filter {
    fn h_z(
        &self,
        samplerate: Self::Sample,
        z: Complex<Self::Sample>,
    ) -> [[Complex<Self::Sample>; 1]; 1] {
        [[self
            .params
            .ftype
            .value()
            .h_z(samplerate, &self.svf, self.scale, z)]]
    }
}

impl DSP<1, 1> for Filter {
    type Sample = Sample;

    fn latency(&self) -> usize {
        self.svf.latency()
    }

    fn reset(&mut self) {
        self.svf.reset();
    }

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let mut x = x[0];
        self.update_coefficients_sample();
        let amps = Sample::splat(util::db_to_gain(
            util::gain_to_db(self.params.amp.smoothed.next()) * self.scale,
        ));
        let filter_out = self.svf.process([x]);
        let filter_out = std::array::from_fn(|i| self.clippers[i].saturate(filter_out[i]));
        x = self.params.ftype.value().mix(amps, x, filter_out);
        [x]
    }
}

impl Filter {
    pub fn new(samplerate: f32, params: Arc<FilterParams>) -> Self {
        let samplerate = Sample::splat(samplerate);
        let fc = Sample::splat(params.cutoff.default_plain_value());
        let q = Sample::splat(params.q.default_plain_value());
        let clippers = [params.resclip.default_plain_value().as_dynamic_type(); 3];
        let last_resclip = params.resclip.value();
        Self {
            params,
            scale: 1.0,
            svf: Svf::new(samplerate, fc, Sample::from_f64(1.0) - q),
            clippers,
            last_resclip,
        }
    }

    #[replace_float_literals(Sample::from_f64(literal))]
    pub fn reset(&mut self, samplerate: f32) {
        let samplerate = Sample::splat(samplerate);
        let fc = Sample::splat(self.params.cutoff.smoothed.next());
        let q = Sample::splat(self.params.q.value());
        let nl = self.params.resclip.value().as_dynamic_type();
        let f = &mut self.svf;
        f.reset();
        f.set_cutoff(fc);
        f.set_r(1. - q);
        f.set_samplerate(samplerate);
        f.set_saturators(nl, nl);
    }

    #[replace_float_literals(Sample::from_f64(literal))]
    pub fn update_coefficients_sample(&mut self) {
        let fc = Sample::splat(self.params.cutoff.smoothed.next());
        let q = if let FilterType::Notch = self.params.ftype.value() {
            0.707
        } else {
            Sample::splat(self.params.q.smoothed.next())
        };
        let resclip = self.params.resclip.value();
        if self.last_resclip != resclip {
            let nl = resclip.as_dynamic_type();
            self.clippers = [nl, nl, nl];
            self.svf.set_saturators(nl, nl);
            self.last_resclip = resclip;
        }
        let f = &mut self.svf;
        f.set_cutoff(fc);
        f.set_r(1. - q);
    }
}
