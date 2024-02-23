use std::sync::Arc;

use enum_map::Enum;
use nih_plug::prelude::*;
use nih_plug::util::db_to_gain;

use dsp::Dsp;
use valib::contrib::nih_plug::{process_buffer_simd, NihParamsController};
use valib::dsp::DSPBlock;

use crate::dsp::{create_dsp, DiodeType, DspParams};

mod dsp;

#[cfg(debug_assertions)]
const OVERSAMPLE: usize = 4;
#[cfg(not(debug_assertions))]
const OVERSAMPLE: usize = 16;

const MAX_BLOCK_SIZE: usize = 512;

struct ClipperPlugin {
    dsp: Dsp,
    params: Arc<NihParamsController<Dsp>>,
}

impl Default for ClipperPlugin {
    fn default() -> Self {
        let dsp = create_dsp(44100.0, OVERSAMPLE, MAX_BLOCK_SIZE);
        let params = NihParamsController::new(&dsp, |k, _| match k {
            DspParams::Drive => FloatParam::new(
                "Drive",
                0.0,
                FloatRange::Skewed {
                    min: db_to_gain(-12.0),
                    max: db_to_gain(40.0),
                    factor: FloatRange::gain_skew_factor(-12.0, 40.0),
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .into(),
            DspParams::ModelSwitch => {
                IntParam::new("Model", 0, IntRange::Linear { min: 0, max: 1 })
                    .with_value_to_string(Arc::new(|x| {
                        match x {
                            0 => "Newton-Rhapson",
                            1 => "Model",
                            _ => "INVALID",
                        }
                        .to_string()
                    }))
                    .into()
            }
            DspParams::DiodeType => {
                IntParam::new("Diode type", 0, IntRange::Linear { min: 0, max: 2 })
                    .with_value_to_string(Arc::new(|x| {
                        DiodeType::from_usize(nih_dbg!(x as _)).name()
                    }))
                    .into()
            }
            DspParams::NumForward => {
                IntParam::new("# Forward", 1, IntRange::Linear { min: 1, max: 5 }).into()
            }
            DspParams::NumBackward => {
                IntParam::new("# Backward", 1, IntRange::Linear { min: 1, max: 5 }).into()
            }
            DspParams::ForceReset => BoolParam::new("Force reset", false).into(),
        });
        Self {
            dsp,
            params: Arc::new(params),
        }
    }
}

impl Plugin for ClipperPlugin {
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

impl ClapPlugin for ClipperPlugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.DiodeClipper";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ClipperPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDiodeClpSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(ClipperPlugin);
nih_export_vst3!(ClipperPlugin);
