use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug::util::db_to_gain;

use valib::contrib::nih_plug::{process_buffer_simd, BindToParameter};
use valib::dsp::parameter::{RemoteControl, RemoteControlled};
use valib::dsp::DSPMeta;

use crate::dsp::{DspParameters, FilterType, SaturatorType};

mod dsp;

const OVERSAMPLE: usize = 2;
const MAX_BLOCK_SIZE: usize = 512;

#[derive(Params)]
#[allow(dead_code)]
struct DirtyBiquadParams {
    #[id = "drive"]
    drive: FloatParam,
    #[id = "ftype"]
    filter_type: EnumParam<FilterType>,
    #[id = "fc"]
    cutoff: FloatParam,
    #[id = "res"]
    resonance: FloatParam,
    #[id = "stype"]
    saturator_type: EnumParam<SaturatorType>,
}

impl DirtyBiquadParams {
    fn new(remote: &RemoteControl<DspParameters>) -> Arc<Self> {
        Arc::new(Self {
            drive: FloatParam::new(
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
            .bind_to_parameter(remote, DspParameters::Drive),
            cutoff: FloatParam::new(
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
            .bind_to_parameter(remote, DspParameters::Cutoff),
            resonance: FloatParam::new(
                "Resonance",
                0.5,
                FloatRange::Skewed {
                    min: 0.02,
                    max: 30.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .bind_to_parameter(remote, DspParameters::Resonance),
            filter_type: {
                //enum_int_param::<FilterType>("Filter type", FilterType::Lowpass).bind_to_parameter(remote, )
                EnumParam::new("Filter type", FilterType::Lowpass)
                    .bind_to_parameter(remote, DspParameters::FilterType)
            },
            saturator_type: {
                //enum_int_param::<SaturatorType>("Saturator type", SaturatorType::Linear).into()
                EnumParam::new("Saturator type", SaturatorType::Linear)
                    .bind_to_parameter(remote, DspParameters::SaturatorType)
            },
        })
    }
}

struct DirtyBiquadPlugin {
    dsp: RemoteControlled<dsp::Dsp>,
    params: Arc<DirtyBiquadParams>,
}

impl Default for DirtyBiquadPlugin {
    fn default() -> Self {
        let dsp = dsp::create(44100.0);
        let params = DirtyBiquadParams::new(&dsp.proxy);
        Self { params, dsp }
    }
}

impl nih_plug::prelude::Plugin for DirtyBiquadPlugin {
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

impl ClapPlugin for DirtyBiquadPlugin {
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

impl Vst3Plugin for DirtyBiquadPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDirTYBiqUAD";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(DirtyBiquadPlugin);
nih_export_vst3!(DirtyBiquadPlugin);
