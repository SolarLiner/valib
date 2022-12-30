use std::sync::Arc;

use nih_plug::buffer::Block;
use nih_plug::prelude::*;
use realfft::num_complex::Complex;

use valib::saturators::{Dynamic, Saturator};
use valib::svf::Svf;
use valib::{DspAnalysis, DSP};

#[derive(Debug, Copy, Clone, Enum, Eq, PartialEq)]
pub enum FilterType {
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
    pub(crate) fn freq_response<S: Saturator<f32>>(
        &self,
        filter: &Svf<f32, S>,
        amp: f32,
        jw: f32,
    ) -> Complex<f32> {
        let [lp, bp, hp] = filter.freq_response([jw]);
        match self {
            Self::Lowpass => lp,
            Self::Bandpass => bp,
            Self::Highpass => hp,
            Self::PeakSharp => lp - hp,
            Self::PeakShelf => {
                let g = amp - 1.;
                1. + bp * g
            }
            Self::Notch => 1. - bp,
            Self::Allpass => Complex::from(1.),
            Self::Lowshelf => 1. + lp * (amp - 1.),
            Self::Highshelf => 1. + hp * (amp - 1.),
        }
    }

    fn mix(&self, amp: f32, x: f32, [lp, bp, hp]: [f32; 3]) -> f32 {
        match self {
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
pub enum DirtyType {
    Linear,
    Tanh,
}

impl Default for DirtyType {
    fn default() -> Self {
        Self::Tanh
    }
}

impl DirtyType {
    pub fn as_dynamic_type(&self) -> Dynamic {
        match self {
            Self::Linear => Dynamic::Linear,
            Self::Tanh => Dynamic::Tanh,
        }
    }

    pub fn equal_loudness(&self) -> f32 {
        match self {
            Self::Linear => 1.,
            Self::Tanh => util::db_to_gain(-40.),
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
    #[id = "dirty"]
    pub(crate) fdirty: EnumParam<DirtyType>,
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
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.)),
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
            .with_unit("dB"),
            ftype: EnumParam::new("Filter Type", FilterType::default()),
            fdirty: EnumParam::new("Nonlinear", DirtyType::default()),
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
pub struct Filter<const N: usize> {
    pub params: Arc<FilterParams>,
    svf: [Svf<f32, Dynamic>; N],
}

impl<const N: usize> Filter<N> {
    pub fn new(samplerate: f32, params: Arc<FilterParams>) -> Self {
        let fc = params.cutoff.default_plain_value();
        let q = params.q.default_plain_value();
        Self {
            params,
            svf: std::array::from_fn(|_| Svf::new(samplerate, fc, 1. - q)),
        }
    }

    pub fn reset(&mut self, samplerate: f32) {
        let fc = self.params.cutoff.smoothed.next();
        let q = self.params.q.value();
        let nl = self.params.fdirty.value().as_dynamic_type();
        for f in &mut self.svf {
            f.reset();
            f.set_cutoff(fc);
            f.set_r(1. - q);
            f.set_samplerate(samplerate);
            f.set_saturators(nl, nl);
        }
    }

    pub fn update_coefficients_sample(&mut self) {
        let fc = self.params.cutoff.smoothed.next();
        let q = if let FilterType::Notch = self.params.ftype.value() {
            0.5
        } else {
            self.params.q.smoothed.next()
        };
        let nl = self.params.fdirty.value().as_dynamic_type();
        for f in &mut self.svf {
            f.set_cutoff(fc);
            f.set_r(1. - q);
            f.set_saturators(nl, nl);
        }
    }

    #[inline(always)]
    pub fn process_block<const SIZE: usize>(&mut self, block: &mut Block, scale: [f32; SIZE]) {
        for (samples, scale) in block.iter_samples().zip(scale) {
            self.process_sample(samples, scale);
        }
    }

    pub fn process_sample<'a>(
        &mut self,
        samples: impl IntoIterator<Item = &'a mut f32>,
        scale: f32,
    ) {
        self.update_coefficients_sample();
        let equal_loudness = self.params.fdirty.value().equal_loudness();
        // for (sample, filt) in samples.into_iter().zip(&mut self.filters) {
        //     *sample = filt.process([*sample * equal_loudness])[0] / equal_loudness;
        // }
        let amps = util::db_to_gain(util::gain_to_db(self.params.amp.smoothed.next()) * scale);
        for (sample, filt) in samples.into_iter().zip(&mut self.svf) {
            *sample *= equal_loudness;
            *sample = self
                .params
                .ftype
                .value()
                .mix(amps, *sample, filt.process([*sample]))
                / equal_loudness;
        }
    }
}
