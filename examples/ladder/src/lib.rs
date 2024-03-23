use std::sync::Arc;

use nih_plug::prelude::*;

use valib::contrib::nih_plug::{enum_int_param, process_buffer_simd, NihParamsController};
use valib::dsp::DSPMeta;

use crate::dsp::{DspParameters, LadderType};

mod dsp;

const MAX_BUFFER_SIZE: usize = 512;
const OVERSAMPLE: usize = 16;

struct LadderFilterPlugin {
    params: Arc<NihParamsController<dsp::Dsp>>,
    dsp: dsp::Dsp,
}

impl Default for LadderFilterPlugin {
    fn default() -> Self {
        let dsp = dsp::create(44100.0);
        let params = NihParamsController::new(&dsp, |param, _name| match param {
            DspParameters::LadderType => {
                enum_int_param::<LadderType>("Topology", LadderType::default()).into()
            }
            DspParameters::Drive => FloatParam::new(
                "Drive",
                1.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 100.0,
                    factor: FloatRange::gain_skew_factor(0.0, 40.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .into(),
            DspParameters::Cutoff => FloatParam::new(
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
            .into(),
            DspParameters::Resonance => {
                FloatParam::new("Q", 0.5, FloatRange::Linear { min: 0., max: 1.25 })
                    .with_unit(" %")
                    .with_value_to_string(formatters::v2s_f32_percentage(2))
                    .with_string_to_value(formatters::s2v_f32_percentage())
                    .into()
            }
            DspParameters::Compensated => BoolParam::new("Compensated", false).into(),
        });
        Self {
            params: Arc::new(params),
            dsp,
        }
    }
}

impl Plugin for LadderFilterPlugin {
    const NAME: &'static str = "Ladder filter";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(2)),
        main_output_channels: Some(new_nonzero_u32(2)),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames {
            layout: Some("Stereo"),
            main_input: Some("input"),
            main_output: Some("output"),
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
        _audio_io_layout: &AudioIOLayout,
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
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        context.set_latency_samples(self.dsp.latency() as _);
        process_buffer_simd::<_, _, MAX_BUFFER_SIZE>(&mut self.dsp, buffer);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for LadderFilterPlugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.LadderFilter";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for LadderFilterPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibLaddrFltSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(LadderFilterPlugin);
nih_export_vst3!(LadderFilterPlugin);
