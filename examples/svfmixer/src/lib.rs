use std::sync::Arc;

use nih_plug::prelude::*;

use extend::FloatParamExt;

use valib::dsp::DSPBlock;
use valib::dsp::{
    buffer::{AudioBufferMut, AudioBufferRef},
    parameter::HasParameters,
};
use valib::oversample::Oversample;
use valib::Scalar;

use crate::dsp::{Dsp, DspInner, DspParam};

mod dsp;

mod extend {
    use std::sync::Arc;

    use nih_plug::params::FloatParam;

    use valib::dsp::parameter::Parameter;

    pub trait FloatParamExt {
        fn bind_to_parameter(self, param: &Parameter) -> Self;
    }

    impl FloatParamExt for FloatParam {
        fn bind_to_parameter(self, param: &Parameter) -> Self {
            let param = param.clone();
            self.with_callback(Arc::new(move |value| param.set_value(value)))
        }
    }
}

const MAX_BUFFER_SIZE: usize = 512;
const OVERSAMPLE: usize = 2;

#[derive(Debug, Params)]
struct PluginParams {
    #[id = "drive"]
    drive: FloatParam,
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

impl PluginParams {
    fn new(dsp: &Dsp) -> Self {
        Self {
            drive: FloatParam::new(
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
            .bind_to_parameter(dsp.get_parameter(DspParam::Drive)),
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
            .bind_to_parameter(dsp.get_parameter(DspParam::Cutoff)),
            q: FloatParam::new("Q", 0.5, FloatRange::Linear { min: 0., max: 1.25 })
                .bind_to_parameter(dsp.get_parameter(DspParam::Resonance)),
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
            .with_unit(" dB")
            .bind_to_parameter(dsp.get_parameter(DspParam::LpGain)),
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
            .with_unit(" dB")
            .bind_to_parameter(dsp.get_parameter(DspParam::BpGain)),
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
            .with_unit(" dB")
            .bind_to_parameter(dsp.get_parameter(DspParam::HpGain)),
        }
    }
}

struct SvfMixerPlugin {
    params: Arc<PluginParams>,
    dsp: Dsp,
}

impl Default for SvfMixerPlugin {
    fn default() -> Self {
        let dsp_inner = DspInner::new(44100.0);
        let dsp = Oversample::new(OVERSAMPLE, MAX_BUFFER_SIZE).with_dsp(dsp_inner);
        let params = PluginParams::new(&dsp);
        Self {
            params: Arc::new(params),
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
    type BackgroundTask = ();
    type SysExMessage = ();

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

fn process_buffer_simd<
    T: Scalar<Element = f32> + From<[f32; 2]>,
    Dsp: DSPBlock<1, 1, Sample = T>,
    const BUFSIZE: usize,
>(
    dsp: &mut Dsp,
    buffer: &mut Buffer,
) {
    let mut input = [T::from_f64(0.0); BUFSIZE];
    let mut output = input;
    for (_, mut block) in buffer.iter_blocks(BUFSIZE) {
        for (i, mut c) in block.iter_samples().enumerate() {
            input[i] = T::from(std::array::from_fn(|i| c.get_mut(i).copied().unwrap()));
            output[i] = input[i];
        }

        let input = &input[..block.samples()];
        let output = &mut output[..block.samples()];

        dsp.process_block(
            AudioBufferRef::from(input),
            AudioBufferMut::from(&mut *output),
        );

        for (i, mut c) in block.iter_samples().enumerate() {
            for (ch, s) in c.iter_mut().enumerate() {
                *s = output[i].extract(ch);
            }
        }
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
