use nih_plug::prelude::*;
use std::sync::Arc;
use valib::{Clean, Driven, Svf, DSP, P1};

#[derive(Debug, Params)]
struct PluginParams {
    #[id = "fc"]
    fc: FloatParam,
    #[id = "q"]
    q: FloatParam,
    #[id = "lp"]
    lp_gain: FloatParam,
    #[id = "bp"]
    bp_gain: FloatParam,
    #[id = "hp"]
    hp_gain: FloatParam,
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
            q: FloatParam::new("Q", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_smoother(SmoothingStyle::Exponential(50.)),
            lp_gain: FloatParam::new(
                "LP Gain",
                0.,
                FloatRange::SymmetricalSkewed {
                    min: -1.,
                    max: 1.,
                    factor: 2.,
                    center: 0.,
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_unit("dB")
            .with_smoother(SmoothingStyle::Exponential(50.)),
            bp_gain: FloatParam::new(
                "BP Gain",
                0.,
                FloatRange::SymmetricalSkewed {
                    min: -1.,
                    max: 1.,
                    factor: 2.,
                    center: 0.,
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_unit("dB")
            .with_smoother(SmoothingStyle::Exponential(50.)),
            hp_gain: FloatParam::new(
                "HP Gain",
                0.,
                FloatRange::SymmetricalSkewed {
                    min: -1.,
                    max: 1.,
                    factor: 2.,
                    center: 0.,
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_unit("dB")
            .with_smoother(SmoothingStyle::Exponential(50.)),
        }
    }
}

#[derive(Debug)]
struct Plugin {
    params: Arc<PluginParams>,
    svf: [Svf<f32, Driven>; 2],
}

impl Default for Plugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        let fc = params.fc.default_plain_value();
        let q = params.q.default_plain_value();
        Self {
            params: params.clone(),
            svf: std::array::from_fn(move |_| Svf::new(1., fc, 1. - q)),
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
        for f in &mut self.svf {
            f.set_samplerate(buffer_config.sample_rate);
        }
        true
    }

    fn reset(&mut self) {
        for f in &mut self.svf {
            f.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for samples in buffer.iter_samples() {
            let fc = self.params.fc.smoothed.next();
            let q = self.params.q.smoothed.next();
            let lp_gain = self.params.lp_gain.smoothed.next();
            let bp_gain = self.params.bp_gain.smoothed.next();
            let hp_gain = self.params.hp_gain.smoothed.next();
            for (ch, f) in samples.into_iter().zip(&mut self.svf) {
                f.set_fc(fc);
                f.set_r(1. - q);
                let [lp, bp, hp] = f.process([*ch]);
                *ch = lp * lp_gain + bp * bp_gain + hp * hp_gain;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Plugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.SVFMixer";
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
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibSvfMixerSLN";
    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_clap!(Plugin);
nih_export_vst3!(Plugin);
