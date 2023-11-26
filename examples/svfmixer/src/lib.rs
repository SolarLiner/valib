use std::sync::Arc;

use nih_plug::prelude::*;

use valib::{clippers::DiodeClipperModel, oversample::Oversample, svf::Svf};
use valib::dsp::DSP;

const MAX_BUFFER_SIZE: usize = 512;
const OVERSAMPLE: usize = 2;

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
            q: FloatParam::new("Q", 0.5, FloatRange::Linear { min: 0., max: 1.25 })
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

type Sample = AutoSimd<[f32; 2]>;
type Filter = Svf<Sample, DiodeClipperModel<Sample>>;

#[derive(Debug)]
struct Plugin {
    params: Arc<PluginParams>,
    svf: Filter,
    oversample: Oversample<Sample>,
}

impl Default for Plugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        let fc = Sample::splat(params.fc.default_plain_value());
        let q = Sample::splat(params.q.default_plain_value());
        Self {
            params,
            svf: Svf::new(Sample::from_f64(1.0), fc, Sample::from_f64(1.0) - q).with_saturators(DiodeClipperModel::new_germanium(3, 2), DiodeClipperModel::new_germanium(3, 2)),
            oversample: Oversample::new(OVERSAMPLE, MAX_BUFFER_SIZE),
        }
    }
}

impl nih_plug::prelude::Plugin for Plugin {
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
        self.svf.set_samplerate(Sample::splat(buffer_config.sample_rate * OVERSAMPLE as f32));
        true
    }

    fn reset(&mut self) {
        self.svf.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut fc = [0.; MAX_BUFFER_SIZE];
        let mut q = [0.; MAX_BUFFER_SIZE];
        let mut lp_gain = [0.; MAX_BUFFER_SIZE];
        let mut bp_gain = [0.; MAX_BUFFER_SIZE];
        let mut hp_gain = [0.; MAX_BUFFER_SIZE];
        let mut simd_slice = [Sample::from_f64(0.0); MAX_BUFFER_SIZE];
        for (_, mut block) in buffer.iter_blocks(MAX_BUFFER_SIZE) {
            for (i, mut sample) in block.iter_samples().enumerate() {
                simd_slice[i] = Sample::new(sample.get_mut(0).copied().unwrap(), sample.get_mut(1).copied().unwrap());
            }
            let len = block.samples();
            let os_len = OVERSAMPLE * len;

            self.params.fc.smoothed.next_block_exact(&mut fc[..len]);
            self.params.q.smoothed.next_block_exact(&mut q[..len]);
            self.params
                .lp_gain
                .smoothed
                .next_block_exact(&mut lp_gain[..len]);
            self.params
                .bp_gain
                .smoothed
                .next_block_exact(&mut bp_gain[..len]);
            self.params
                .hp_gain
                .smoothed
                .next_block_exact(&mut hp_gain[..len]);

            let mut os_fc = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_q = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_lp_gain = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_bp_gain = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_hp_gain = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];

            valib::util::lerp_block(&mut os_fc[..os_len], &fc[..len]);
            valib::util::lerp_block(&mut os_q[..os_len], &q[..len]);
            valib::util::lerp_block(&mut os_lp_gain[..os_len], &lp_gain[..len]);
            valib::util::lerp_block(&mut os_bp_gain[..os_len], &bp_gain[..len]);
            valib::util::lerp_block(&mut os_hp_gain[..os_len], &hp_gain[..len]);

            let buffer = &mut simd_slice[..len];
            let mut os_buffer = self.oversample.oversample(buffer);
            for (i, s) in os_buffer.iter_mut().enumerate() {
                let fc = os_fc[i];
                let q = os_q[i];
                let lp_gain = Sample::splat(os_lp_gain[i]);
                let bp_gain = Sample::splat(os_bp_gain[i]);
                let hp_gain = Sample::splat(os_hp_gain[i]);

                self.svf.set_cutoff(Sample::splat(fc));
                self.svf.set_r(Sample::splat(1. - q));
                let [lp, bp, hp] = self.svf.process([*s]);
                *s = lp * lp_gain + bp * bp_gain + hp * hp_gain;
            }
            os_buffer.finish(buffer);

            for (i, mut s) in block.iter_samples().enumerate() {
                *s.get_mut(0).unwrap() = buffer[i].extract(0);
                *s.get_mut(1).unwrap() = buffer[i].extract(1);
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
    ];
}

impl Vst3Plugin for Plugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibSvfMixerSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(Plugin);
nih_export_vst3!(Plugin);
