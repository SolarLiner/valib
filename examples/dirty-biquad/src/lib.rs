#![feature(default_free_fn)]

use std::default::default;
use std::sync::Arc;

use nih_plug::prelude::*;

use valib::clippers::DiodeClipperModel;
use valib::oversample::Oversample;
use valib::saturators::Dynamic;
use valib::util::lerp_block;
use valib::{biquad::Biquad, DSP};

const OVERSAMPLE: usize = 2;
const MAX_BLOCK_SIZE: usize = 512;

#[derive(Debug, Enum, Eq, PartialEq)]
enum FilterType {
    Lowpass,
    Bandpass,
    Highpass,
}

#[derive(Debug, Enum, Eq, PartialEq)]
enum NLType {
    Linear,
    Clipped,
    Tanh,
    Diode,
}

impl NLType {
    pub fn as_dynamic_saturator(&self) -> Dynamic<f32> {
        match self {
            Self::Linear => Dynamic::Linear,
            Self::Clipped => Dynamic::HardClipper,
            Self::Tanh => Dynamic::Tanh,
            Self::Diode => Dynamic::DiodeClipper(DiodeClipperModel::new_silicon(1, 1)),
        }
    }
}

#[derive(Debug, Params)]
struct PluginParams {
    #[id = "fc"]
    fc: FloatParam,
    #[id = "q"]
    q: FloatParam,
    #[id = "drive"]
    drive: FloatParam,
    #[id = "type"]
    filter_type: EnumParam<FilterType>,
    #[id = "nl"]
    nonlinearity: EnumParam<NLType>,
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
            q: FloatParam::new(
                "Q",
                0.5,
                FloatRange::Linear {
                    min: 1e-2,
                    max: 30.,
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.)),
            drive: FloatParam::new(
                "Drive",
                1.,
                FloatRange::SymmetricalSkewed {
                    min: util::db_to_gain(-36.),
                    max: util::db_to_gain(36.),
                    center: 1.,
                    factor: FloatRange::skew_factor(-2.),
                },
            )
            .with_unit(" dB")
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_smoother(SmoothingStyle::Exponential(100.)),
            filter_type: EnumParam::new("Filter Type", FilterType::Lowpass),
            nonlinearity: EnumParam::new("Nonlinearity", NLType::Linear),
        }
    }
}

#[derive(Debug)]
struct Plugin {
    params: Arc<PluginParams>,
    biquad: [Biquad<f32, Dynamic<f32>>; 2],
    oversample: [Oversample<f32>; 2],
}

impl Plugin {
    fn reset_filters(&mut self, samplerate: f32) {
        let fc = self.params.fc.value() / samplerate;
        let q = self.params.fc.value();
        let nltype = self.params.nonlinearity.value().as_dynamic_saturator();
        for x in &mut self.biquad {
            *x = match self.params.filter_type.value() {
                FilterType::Lowpass => Biquad::lowpass(fc, q),
                FilterType::Bandpass => Biquad::bandpass_peak0(fc, q),
                FilterType::Highpass => Biquad::highpass(fc, q),
            };
            x.set_saturators(nltype, nltype);
        }
        for os in &mut self.oversample {
            os.reset();
        }
    }
}

impl Default for Plugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        Self {
            params,
            biquad: std::array::from_fn(move |_| Biquad::new(default(), default())),
            oversample: std::array::from_fn(|_| Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE)),
        }
    }
}

impl nih_plug::prelude::Plugin for Plugin {
    const NAME: &'static str = "Dirty Biquad";
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
        self.reset_filters(buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        for f in &mut self.biquad {
            f.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        ctx: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let os_samplerate = ctx.transport().sample_rate * OVERSAMPLE as f32;
        let mut fc = [0.; MAX_BLOCK_SIZE];
        let mut q = [0.; MAX_BLOCK_SIZE];
        let mut drive = [0.; MAX_BLOCK_SIZE];
        let mut os_fc = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
        let mut os_q = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
        let mut os_drive = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
        for (_, mut block) in buffer.iter_blocks(MAX_BLOCK_SIZE) {
            let len = block.samples();
            let os_len = OVERSAMPLE * len;
            self.params.fc.smoothed.next_block_exact(&mut fc[..len]);
            self.params.q.smoothed.next_block_exact(&mut q[..len]);
            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
            lerp_block(&mut os_fc[..os_len], &fc[..len]);
            lerp_block(&mut os_q[..os_len], &q[..len]);
            lerp_block(&mut os_drive[..os_len], &drive[..len]);

            for ch in 0..2 {
                let buffer = block.get_mut(ch).unwrap();
                let mut os_buffer = self.oversample[ch].oversample(buffer);
                for (i, s) in os_buffer.iter_mut().enumerate() {
                    let fc = os_fc[i] / os_samplerate;
                    let q = os_q[i];
                    let drive = os_drive[i];
                    let f = &mut self.biquad[ch];
                    let filter = match self.params.filter_type.value() {
                        FilterType::Lowpass => Biquad::lowpass(fc, q),
                        FilterType::Bandpass => Biquad::bandpass_peak0(fc, q),
                        FilterType::Highpass => Biquad::highpass(fc, q),
                    };

                    let nltype = self.params.nonlinearity.value().as_dynamic_saturator();
                    f.update_coefficients(&filter);
                    f.set_saturators(nltype, nltype);
                    *s = f.process([*s * drive])[0] / drive;
                }
                os_buffer.finish(buffer);
            }
        }

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
    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_clap!(Plugin);
nih_export_vst3!(Plugin);
