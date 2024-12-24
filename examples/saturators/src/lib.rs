use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug::util::db_to_gain;

use dsp::Dsp;
use valib::contrib::nih_plug::{process_buffer_simd, BindToParameter};
use valib::dsp::parameter::{RemoteControl, RemoteControlled};
use valib::dsp::DSPMeta;

use crate::dsp::{create_dsp, DspInnerParams, DspParams, SaturatorType};

mod dsp;

const MAX_OVERSAMPLE: usize = 64;

const MAX_BLOCK_SIZE: usize = 64;

#[derive(Debug, Params)]
struct SaturatorsParams {
    #[id = "drive"]
    drive: FloatParam,
    #[id = "sat"]
    saturator: EnumParam<SaturatorType>,
    #[id = "fb"]
    feedback: FloatParam,
    #[id = "alvl"]
    adaa_level: FloatParam,
    #[id = "aeps"]
    adaa_epsilon: FloatParam,
    #[id = "osamt"]
    oversampling: IntParam,
    #[id = "dcblk"]
    dc_blocker: BoolParam,
}

impl SaturatorsParams {
    fn new(remote: &RemoteControl<DspParams>) -> Arc<Self> {
        Arc::new(Self {
            drive: FloatParam::new(
                "Drive",
                1.0,
                FloatRange::Skewed {
                    min: db_to_gain(-12.0),
                    max: db_to_gain(40.0),
                    factor: FloatRange::gain_skew_factor(-12.0, 40.0),
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .bind_to_parameter(remote, DspParams::InnerParam(DspInnerParams::Drive)),
            saturator: EnumParam::new("Saturator", SaturatorType::Tanh)
                .bind_to_parameter(remote, DspParams::InnerParam(DspInnerParams::Saturator)),
            feedback: FloatParam::new(
                "Feedback",
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -2.0,
                    max: 2.0,
                    factor: 1.0,
                    center: 0.0,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(2))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .bind_to_parameter(remote, DspParams::InnerParam(DspInnerParams::Feedback)),
            adaa_level: {
                FloatParam::new("ADAA Level", 2.0, FloatRange::Linear { min: 0.0, max: 2.0 })
                    .with_step_size(1.0)
                    .with_value_to_string(formatters::v2s_f32_rounded(0))
                    .bind_to_parameter(remote, DspParams::InnerParam(DspInnerParams::AdaaLevel))
            },
            adaa_epsilon: FloatParam::new(
                "ADAA Epsilon",
                1e-4,
                FloatRange::Skewed {
                    min: 1e-10,
                    max: 1.0,
                    factor: -2.0,
                },
            )
            .with_value_to_string(Arc::new(|f| format!("{f:.1e}")))
            .bind_to_parameter(remote, DspParams::InnerParam(DspInnerParams::AdaaEpsilon)),
            oversampling: IntParam::new(
                "Oversampling",
                usize::ilog2(MAX_OVERSAMPLE) as _,
                IntRange::Linear {
                    min: 0,
                    max: usize::ilog2(MAX_OVERSAMPLE) as _,
                },
            )
            .with_unit("x")
            .with_string_to_value(formatters::s2v_i32_power_of_two())
            .with_value_to_string(formatters::v2s_i32_power_of_two())
            .bind_to_parameter(remote, DspParams::Oversampling),
            dc_blocker: BoolParam::new("Block DC", true)
                .bind_to_parameter(remote, DspParams::DcBlocker),
        })
    }
}

struct SaturatorsPlugin {
    dsp: RemoteControlled<Dsp>,
    params: Arc<SaturatorsParams>,
}

impl Default for SaturatorsPlugin {
    fn default() -> Self {
        let dsp = create_dsp(44100.0, MAX_OVERSAMPLE, MAX_BLOCK_SIZE);
        let params = SaturatorsParams::new(&dsp.proxy);
        Self { dsp, params }
    }
}

impl Plugin for SaturatorsPlugin {
    const NAME: &'static str = "Diode Clipper";
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
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        _context.set_latency_samples(self.dsp.latency() as _);
        process_buffer_simd::<_, _, MAX_BLOCK_SIZE>(&mut self.dsp, buffer);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for SaturatorsPlugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.Saturators";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for SaturatorsPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibSaturatrSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(SaturatorsPlugin);
nih_export_vst3!(SaturatorsPlugin);
