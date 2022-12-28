#![feature(default_free_fn)]

use nih_plug::prelude::*;
use std::default::default;
use std::sync::Arc;
use valib::saturators::Dynamic;
use valib::{biquad::Biquad, DSP};

#[derive(Debug, Enum, Eq, PartialEq)]
enum FilterType {
    Lowpass,
    Bandpass,
    Highpass,
}

#[derive(Debug, Enum, Eq, PartialEq)]
enum NLType {
    Linear,
    Clipped,
    Tanh,
}

impl NLType {
    pub fn as_dynamic_saturator(&self) -> Dynamic {
        match self {
            Self::Linear => Dynamic::Linear,
            Self::Clipped => Dynamic::Clipper,
            Self::Tanh => Dynamic::Tanh,
        }
    }
}

#[derive(Debug, Params)]
struct PluginParams {
    #[id = "fc"]
    fc: FloatParam,
    #[id = "q"]
    q: FloatParam,
    #[id = "type"]
    filter_type: EnumParam<FilterType>,
    #[id = "nl"]
    nonlinearity: EnumParam<NLType>,
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
            fc: FloatParam::new(
                "Frequency",
                300.,
                FloatRange::Skewed {
                    min: 20.,
                    max: 20e3,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_hz_then_khz_with_note_name(2, false))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_smoother(SmoothingStyle::Exponential(10.)),
            q: FloatParam::new(
                "Q",
                0.5,
                FloatRange::Linear {
                    min: 1e-2,
                    max: 30.,
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.)),
            filter_type: EnumParam::new("Filter Type", FilterType::Lowpass),
            nonlinearity: EnumParam::new("Nonlinearity", NLType::Linear),
        }
    }
}

#[derive(Debug)]
struct Plugin {
    params: Arc<PluginParams>,
    biquad: [Biquad<f32, Dynamic>; 2],
}

impl Plugin {
    fn reset_filters(&mut self, samplerate: f32) {
        let fc = self.params.fc.value() / samplerate;
        let q = self.params.fc.value();
        let nltype = self.params.nonlinearity.value().as_dynamic_saturator();
        for x in &mut self.biquad {
            *x = match self.params.filter_type.value() {
                FilterType::Lowpass => Biquad::lowpass(fc, q),
                FilterType::Bandpass => Biquad::bandpass_peak0(fc, q),
                FilterType::Highpass => Biquad::highpass(fc, q),
            };
            x.set_saturators(nltype, nltype);
        }
    }

    fn update_filters_sample(&mut self, samplerate: f32) {
        let fc = self.params.fc.smoothed.next() / samplerate;
        let q = self.params.q.smoothed.next();

        let filter = match self.params.filter_type.value() {
            FilterType::Lowpass => Biquad::lowpass(fc, q),
            FilterType::Bandpass => Biquad::bandpass_peak0(fc, q),
            FilterType::Highpass => Biquad::highpass(fc, q),
        };

        let nltype = self.params.nonlinearity.value().as_dynamic_saturator();
        for f in &mut self.biquad {
            f.update_coefficients(&filter);
            f.set_saturators(nltype, nltype);
        }
    }
}

impl Default for Plugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        Self {
            params,
            biquad: std::array::from_fn(move |_| Biquad::new(default(), default())),
        }
    }
}

impl nih_plug::prelude::Plugin for Plugin {
    const NAME: &'static str = "SVF Mixer";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = "0.0.0";
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        config.num_input_channels == 2 && config.num_output_channels == 2
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.reset_filters(buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        for f in &mut self.biquad {
            f.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        ctx: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for samples in buffer.iter_samples() {
            self.update_filters_sample(ctx.transport().sample_rate);
            for (ch, f) in samples.into_iter().zip(&mut self.biquad) {
                *ch = f.process([*ch])[0];
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Plugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.DirtyBiquad";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for Plugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDirTYBiqUAD";
    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_clap!(Plugin);
nih_export_vst3!(Plugin);
