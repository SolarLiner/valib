use std::sync::Arc;

use nih_plug::prelude::*;

use valib::contrib::nih_plug::{process_buffer_simd, NihParamsController};
use valib::dsp::DSPMeta;

use valib::oversample::Oversample;

use crate::dsp::{Dsp, DspInner, DspParam};

mod dsp;

const MAX_BUFFER_SIZE: usize = 512;
const OVERSAMPLE: usize = 2;

struct SvfMixerPlugin {
    params: Arc<NihParamsController<Dsp>>,
    dsp: Dsp,
}

impl Default for SvfMixerPlugin {
    fn default() -> Self {
        let samplerate = 44100.0;
        let dsp_inner = DspInner::new(samplerate);
        let dsp = Oversample::new(OVERSAMPLE, MAX_BUFFER_SIZE).with_dsp(samplerate, dsp_inner);
        let params_controller = NihParamsController::new(&dsp, |param, _| match param {
            DspParam::Drive => FloatParam::new(
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
            .with_unit(" dB")
            .into(),
            DspParam::Cutoff => FloatParam::new(
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
            DspParam::Resonance => {
                FloatParam::new("Q", 0.5, FloatRange::Linear { min: 0., max: 1.25 }).into()
            }
            DspParam::LpGain => FloatParam::new(
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
            .with_unit(" dB")
            .into(),
            DspParam::BpGain => FloatParam::new(
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
            .with_unit(" dB")
            .into(),
            DspParam::HpGain => FloatParam::new(
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
            .with_unit(" dB")
            .into(),
        });
        Self {
            params: Arc::new(params_controller),
            dsp,
        }
    }
}

impl nih_plug::prelude::Plugin for SvfMixerPlugin {
    const NAME: &'static str = "SVF Mixer";
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
        let os_samplerate = self.dsp.os_factor() as f32 * buffer_config.sample_rate;
        self.dsp.inner.set_samplerate(os_samplerate);
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
        process_buffer_simd::<_, _, MAX_BUFFER_SIZE>(&mut self.dsp, buffer);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for SvfMixerPlugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.SVFMixer";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for SvfMixerPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibSvfMixerSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(SvfMixerPlugin);
nih_export_vst3!(SvfMixerPlugin);
