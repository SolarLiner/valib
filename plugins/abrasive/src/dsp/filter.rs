use nalgebra::{Complex, SMatrix};
use nih_plug::formatters;
use nih_plug::params::{EnumParam, FloatParam, Params};
use nih_plug::prelude::*;
use nih_plug::util::db_to_gain;
use numeric_literals::replace_float_literals;
use realfft::num_traits::One;
use std::sync::Arc;
use valib::contrib::nih_plug::ValueAs;
use valib::dsp::analysis::DspAnalysis;
use valib::dsp::blocks::ModMatrix;
use valib::dsp::{DSPMeta, DSPProcess};
use valib::filters::svf::Svf;
use valib::saturators::{bjt, Saturator};
use valib::Scalar;

#[derive(Debug, Copy, Clone, Enum, Eq, PartialEq)]
pub enum FilterType {
    Bypass,
    #[name = "Low pass"]
    Lowpass,
    #[name = "Band pass"]
    Bandpass,
    #[name = "High pass"]
    Highpass,
    #[name = "Low shelf"]
    Lowshelf,
    #[name = "High shelf"]
    Highshelf,
    #[name = "Band shelf"]
    Band,
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
                    min: db_to_gain(-40.),
                    max: db_to_gain(40.),
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
    mixer: ModMatrix<T, 4, 1>,
}

impl<T: Scalar> FilterMixer<T> {
    pub fn new(filter_type: FilterType, amp: T) -> Self {
        Self {
            mixer: ModMatrix {
                weights: Self::get_weights(filter_type, amp),
            },
        }
    }

    pub fn set_mixer(&mut self, filter_type: FilterType, amp: T) {
        self.mixer.weights = Self::get_weights(filter_type, amp);
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn get_weights(filter_type: FilterType, amp: T) -> SMatrix<T, 1, 4> {
        let inner = match filter_type {
            FilterType::Bypass => [1.0, 0.0, 0.0, 0.0],
            FilterType::Lowpass => [0.0, 1.0, 0.0, 0.0],
            FilterType::Bandpass => [0.0, 0.0, 1.0, 0.0],
            FilterType::Highpass => [0.0, 0.0, 0.0, 1.0],
            FilterType::Lowshelf => {
                // m^-4 = amp => m = amp^4
                // m^-2 = (m^-4)^(1/2) = sqrt(m^-4) = sqrt(amp)
                let mp2 = amp.simd_sqrt();
                [0.0, amp, mp2, 1.0]
            }
            FilterType::Highshelf => {
                // m^4 = amp
                // m^2 = sqrt(amp)
                let m2 = amp.simd_sqrt();
                [0.0, 1., m2, amp]
            }
            FilterType::Band => [1.0, 0.0, amp - 1., 0.0],
            FilterType::Notch => [1.0, 0.0, -1.0, 0.0],
            FilterType::Allpass => [1.0, 0.0, -2.0, 0.0],
        };
        SMatrix::from(inner.map(|x| [x]))
    }
}

impl<T: Scalar> DSPMeta for FilterMixer<T> {
    type Sample = T;
}

impl<T: Scalar> DSPProcess<4, 1> for FilterMixer<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 4]) -> [Self::Sample; 1] {
        self.mixer.process(x)
    }
}

impl<T: Scalar> DspAnalysis<4, 1> for FilterMixer<T> {
    #[replace_float_literals(Complex::from(T::from_f64(literal)))]
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 4] {
        self.mixer.h_z(z)
    }
}

impl Default for FilterType {
    fn default() -> Self {
        Self::Bypass
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
        let filter_type = self.params.ftype.value();
        let amp = self.scale * T::from_f64(self.params.amp.smoothed.next() as _);
        self.mixer.set_mixer(filter_type, amp);
        self.svf.set_cutoff(self.params.cutoff.value_as());
        let r = T::one() - self.params.q.value_as::<T>();
        self.svf.set_r(r);

        let x_sat = self.input_clip.saturate(x);
        self.input_clip.update_state(x, x_sat);
        let [lp, bp, hp] = self.svf.process([x_sat]);
        let bp1 = (r + r) * bp;
        self.mixer.process([x, lp, bp1, hp])
    }
}

impl<T: Scalar> DspAnalysis<1, 1> for FilterModule<T> {
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 1] {
        let mut h = SMatrix::<_, 3, 1>::from(self.svf.h_z(z));
        h[1] *= self.svf.get_2r();
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
        self.svf.set_cutoff(T::from_f64(if use_modulated {
            self.params.cutoff.modulated_plain_value() as _
        } else {
            self.params.cutoff.unmodulated_plain_value() as _
        }));
        self.svf.set_r(T::from_f64(if use_modulated {
            1.0 - self.params.q.modulated_plain_value() as f64
        } else {
            1.0 - self.params.q.unmodulated_plain_value() as f64
        }));

        let filter_type = self.params.ftype.value();
        let amp = T::from_f64(if use_modulated {
            self.params.amp.modulated_plain_value() as f64
        } else {
            self.params.amp.unmodulated_plain_value() as f64
        });
        self.mixer.set_mixer(filter_type, amp);
    }
}
