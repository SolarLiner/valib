use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug::util::db_to_gain;

use valib::contrib::nih_plug::{enum_int_param, process_buffer_simd, NihParamsController};
use valib::dsp::DSPMeta;

use crate::dsp::{DspParameters, FilterType, SaturatorType};

mod dsp;

const OVERSAMPLE: usize = 2;
const MAX_BLOCK_SIZE: usize = 512;

struct Plugin {
    params: Arc<NihParamsController<dsp::Dsp>>,
    dsp: dsp::Dsp,
}

impl Default for Plugin {
    fn default() -> Self {
        let dsp = dsp::create(44100.0);
        let params = NihParamsController::new(&dsp, |param, _name| match param {
            DspParameters::Drive => FloatParam::new(
                "Drive",
                1.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: db_to_gain(40.0),
                    factor: FloatRange::gain_skew_factor(0.0, 40.0),
                },
            )
            .with_unit(" dB")
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .into(),
            DspParameters::Cutoff => FloatParam::new(
                "Cutoff",
                3000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20e3,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_value_to_string(formatters::v2s_f32_hz_then_khz_with_note_name(2, true))
            .into(),
            DspParameters::Resonance => FloatParam::new(
                "Resonance",
                0.5,
                FloatRange::Skewed {
                    min: 0.02,
                    max: 30.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .into(),
            DspParameters::FilterType => {
                enum_int_param::<FilterType>("Filter type", FilterType::Lowpass).into()
            }
            DspParameters::SaturatorType => {
                enum_int_param::<SaturatorType>("Saturator type", SaturatorType::Linear).into()
            }
        });
        Self {
            params: Arc::new(params),
            dsp,
        }
    }
}

impl nih_plug::prelude::Plugin for Plugin {
    const NAME: &'static str = "Dirty Biquad";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = "0.0.0";
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(2)),
        main_output_channels: Some(new_nonzero_u32(2)),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames {
            layout: Some("Stereo"),
            main_input: Some("Input"),
            main_output: Some("Output"),
            aux_inputs: &[],
            aux_outputs: &[],
        },
    }];
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_config: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.dsp.set_samplerate(buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        self.dsp.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        ctx: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        ctx.set_latency_samples(self.dsp.latency() as _);
        process_buffer_simd::<_, _, MAX_BLOCK_SIZE>(&mut self.dsp, buffer);
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
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(Plugin);
nih_export_vst3!(Plugin);
