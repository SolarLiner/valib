use std::sync::Arc;

use nih_plug::prelude::*;

use valib::math::interpolation::{Cubic, Interpolate};
use valib::simd::{AutoSimd, SimdValue};
use valib::{
    dsp::blocks::{ModMatrix, Series2},
    saturators::{Clipper, Saturator, Slew},
};
use valib::{dsp::DSP, Scalar};
use valib::{filters::svf::Svf, oversample::Oversample};

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

impl Default for PluginParams {
    fn default() -> Self {
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
            .with_smoother(SmoothingStyle::Exponential(10.)),
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

#[derive(Debug, Clone, Copy)]
struct OpAmp<T>(Clipper, Slew<T>);

impl<T: Scalar> Default for OpAmp<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: Scalar> Saturator<T> for OpAmp<T>
where
    Clipper: Saturator<T>,
    Slew<T>: Saturator<T>,
{
    fn saturate(&self, x: T) -> T {
        self.1.saturate(self.0.saturate(x))
    }

    fn sat_diff(&self, x: T) -> T {
        self.0.sat_diff(x) * self.1.sat_diff(self.0.saturate(x))
    }

    fn update_state(&mut self, x: T, y: T) {
        let xc = self.0.saturate(x);
        self.0.update_state(x, xc);
        self.1.update_state(xc, y);
    }
}

type Sample = AutoSimd<[f32; 2]>;
type Filter = Svf<Sample, OpAmp<Sample>>;
type Dsp = Series2<Filter, ModMatrix<Sample, 3, 1>, 3>;

#[derive(Debug)]
struct SvfMixerPlugin {
    params: Arc<PluginParams>,
    oversample: Oversample<Sample>,
    dsp: Dsp,
}

impl Default for SvfMixerPlugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        let fc = Sample::splat(params.fc.default_plain_value());
        let q = Sample::splat(params.q.default_plain_value());
        let sat = OpAmp(Clipper, Slew::new(Sample::splat(util::db_to_gain(60.0))));
        let filter = Svf::new(Sample::from_f64(1.0), fc, Sample::from_f64(1.0) - q)
            .with_saturators(sat, sat);
        let mod_matrix = ModMatrix::default();
        Self {
            params,
            dsp: Series2::new(filter, mod_matrix),
            oversample: Oversample::new(OVERSAMPLE, MAX_BUFFER_SIZE),
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
        self.dsp
            .left_mut()
            .set_samplerate(buffer_config.sample_rate * OVERSAMPLE as f32);
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
        let mut drive = [0.; MAX_BUFFER_SIZE];
        let mut fc = [0.; MAX_BUFFER_SIZE];
        let mut q = [0.; MAX_BUFFER_SIZE];
        let mut lp_gain = [0.; MAX_BUFFER_SIZE];
        let mut bp_gain = [0.; MAX_BUFFER_SIZE];
        let mut hp_gain = [0.; MAX_BUFFER_SIZE];
        let mut simd_slice = [Sample::from_f64(0.0); MAX_BUFFER_SIZE];
        for (_, mut block) in buffer.iter_blocks(MAX_BUFFER_SIZE) {
            for (i, mut sample) in block.iter_samples().enumerate() {
                simd_slice[i] = Sample::new(
                    sample.get_mut(0).copied().unwrap(),
                    sample.get_mut(1).copied().unwrap(),
                );
            }
            let len = block.samples();
            let os_len = OVERSAMPLE * len;

            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
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

            let mut os_drive = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_fc = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_q = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_lp_gain = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_bp_gain = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_hp_gain = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];

            Cubic::interpolate_slice(&mut os_drive[..os_len], &drive[..len]);
            Cubic::interpolate_slice(&mut os_fc[..os_len], &fc[..len]);
            Cubic::interpolate_slice(&mut os_q[..os_len], &q[..len]);
            Cubic::interpolate_slice(&mut os_lp_gain[..os_len], &lp_gain[..len]);
            Cubic::interpolate_slice(&mut os_bp_gain[..os_len], &bp_gain[..len]);
            Cubic::interpolate_slice(&mut os_hp_gain[..os_len], &hp_gain[..len]);

            let buffer = &mut simd_slice[..len];
            let mut os_buffer = self.oversample.oversample(buffer);
            for (i, s) in os_buffer.iter_mut().enumerate() {
                let fc = os_fc[i];
                let q = os_q[i];
                let filter = self.dsp.left_mut();
                let drive = Sample::splat(os_drive[i]);
                filter.set_cutoff(Sample::splat(fc));
                filter.set_r(Sample::splat(1. - q));
                let mod_matrix = self.dsp.right_mut();
                mod_matrix.weights[(0, 0)] = Sample::splat(os_lp_gain[i]);
                mod_matrix.weights[(0, 1)] = Sample::splat(os_bp_gain[i]);
                mod_matrix.weights[(0, 2)] = Sample::splat(os_hp_gain[i]);
                *s = self.dsp.process([*s * drive])[0] / drive;
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
