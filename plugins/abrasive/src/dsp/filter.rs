use nalgebra::{Complex, SMatrix, SVector};
use nih_plug::formatters;
use nih_plug::params::{EnumParam, FloatParam, Params};
use nih_plug::prelude::*;
use num_traits::Float;
use numeric_literals::replace_float_literals;
use realfft::num_traits::One;
use std::sync::Arc;
use valib::contrib::nih_plug::ValueAs;
use valib::dsp::analysis::DspAnalysis;
use valib::dsp::{DSPMeta, DSPProcess};
use valib::filters::svf::Svf;
use valib::saturators::{bjt, Asinh, Saturator};
use valib::Scalar;

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

pub struct FilterMixer<T> {
    pub filter_type: FilterType,
    pub amp: T,
}

impl<T> FilterMixer<T> {
    fn new(filter_type: FilterType, amp: T) -> Self {
        Self { filter_type, amp }
    }
}

impl<T: Scalar> DSPMeta for FilterMixer<T> {
    type Sample = T;
}

impl<T: Scalar> DSPProcess<4, 1> for FilterMixer<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, [x, lp, bp, hp]: [Self::Sample; 4]) -> [Self::Sample; 1] {
        let y = match self.filter_type {
            FilterType::Bypass => x,
            FilterType::Lowpass => lp,
            FilterType::Bandpass => bp,
            FilterType::Highpass => hp,
            FilterType::PeakSharp => lp - hp,
            FilterType::PeakShelf => {
                let g = self.amp - 1.;
                x + bp * g
            }
            FilterType::Notch => x - bp,
            FilterType::Allpass => x - 2. * bp,
            FilterType::Lowshelf => x + lp * (self.amp - 1.),
            FilterType::Highshelf => x + hp * (self.amp - 1.),
        };
        [y]
    }
}

impl<T: Scalar> DspAnalysis<4, 1> for FilterMixer<T> {
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, _: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 4] {
        let g = Complex::from(self.amp) - 1.;
        let inner = match self.filter_type {
            FilterType::Bypass => [1.0, 0.0, 0.0, 0.0],
            FilterType::Lowpass => [0.0, 1.0, 0.0, 0.0],
            FilterType::Bandpass => [0.0, 0.0, 1.0, 0.0],
            FilterType::Highpass => [0.0, 0.0, 0.0, 1.0],
            FilterType::Lowshelf => [1.0, g, 0.0, 0.0],
            FilterType::Highshelf => [1.0, 0.0, 0.0, g],
            FilterType::PeakSharp => [0.0, 1.0, 0.0, -1.0],
            FilterType::PeakShelf => [1.0, 0.0, g, 0.0],
            FilterType::Notch => [1.0, 0.0, -1.0, 1.0],
            FilterType::Allpass => [1.0, 0.0, -2.0, 0.0],
        };
        inner.map(|x| [x])
    }
}

impl Default for FilterType {
    fn default() -> Self {
        Self::PeakShelf
    }
}

struct Sinh;

impl<T: Scalar> Saturator<T> for Sinh {
    fn saturate(&self, x: T) -> T {
        x.simd_sinh()
    }
}

pub struct FilterModule<T> {
    params: Arc<FilterParams>,
    input_clip: bjt::CommonCollector<T>,
    svf: Svf<T, Sinh>,
    mixer: FilterMixer<T>,
    scale: T,
}

impl<T: Scalar> DSPMeta for FilterModule<T> {
    type Sample = T;
    fn set_samplerate(&mut self, samplerate: f32) {
        self.svf.set_samplerate(samplerate);
        self.mixer.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.svf.latency() + self.mixer.latency()
    }

    fn reset(&mut self) {
        self.svf.reset();
        self.mixer.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for FilterModule<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.mixer.filter_type = self.params.ftype.value();
        self.mixer.amp = self.scale * T::from_f64(self.params.amp.smoothed.next() as _);
        self.svf.set_cutoff(self.params.cutoff.value_as());
        self.svf.set_r(T::one() - self.params.q.value_as::<T>());

        let x = self.input_clip.saturate(x);
        let [lp, bp, hp] = self.svf.process([x]);
        self.mixer.process([x, lp, bp, hp])
    }
}

impl<T: Scalar> DspAnalysis<1, 1> for FilterModule<T> {
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 1] {
        let h = SMatrix::<_, 3, 1>::from(self.svf.h_z(z));
        let h = h.insert_row(0, Complex::one());
        let y = SMatrix::<_, 1, 4>::from(self.mixer.h_z(z)) * h;
        [[y[0]]]
    }
}

impl<T: Scalar> FilterModule<T> {
    pub fn new(samplerate: T, params: Arc<FilterParams>) -> Self {
        let svf = Svf::new(samplerate, params.cutoff.value_as(), params.q.value_as())
            .with_saturator(Sinh);
        let mixer = FilterMixer::new(params.ftype.value(), params.amp.value_as());
        Self {
            params,
            input_clip: Default::default(),
            svf,
            mixer,
            scale: T::one(),
        }
    }

    pub fn set_scale(&mut self, scale: T) {
        self.scale = scale;
    }

    pub fn use_param_values(&mut self, use_modulated: bool) {
        self.mixer.filter_type = self.params.ftype.value();
        self.svf.set_cutoff(T::from_f64(if use_modulated {
            self.params.cutoff.modulated_plain_value() as _
        } else {
            self.params.cutoff.unmodulated_plain_value() as _
        }));
        self.svf.set_r(T::from_f64(if use_modulated {
            self.params.q.modulated_plain_value() as _
        } else {
            self.params.q.unmodulated_plain_value() as _
        }));
    }
}
