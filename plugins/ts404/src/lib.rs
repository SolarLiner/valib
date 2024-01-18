use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug::util::{db_to_gain_fast, gain_to_db, MINUS_INFINITY_DB, MINUS_INFINITY_GAIN};
use num_traits::Zero;
use valib::dsp::blocks::Series;
use valib::dsp::utils::{slice_to_mono_block, slice_to_mono_block_mut};
use valib::dsp::{DSPBlock, DSP};
use valib::oversample::Oversample;
use valib::simd::{AutoF64x2, SimdValue};
use valib::util::lerp;
use valib::Scalar;

use crate::dsp::{Bypass, ClipperStage, InputStage, OutputStage, ToneStage};

mod dsp;
mod gen;

type Sample = AutoF64x2;
type Dsp = Series<(
    Bypass<InputStage<Sample>>,
    Bypass<Series<(ClipperStage<Sample>, ToneStage<Sample>)>>,
    OutputStage<Sample>,
)>;

const OVERSAMPLE: usize = 4;
const MAX_BLOCK_SIZE: usize = 512;

struct Ts404 {
    params: Arc<Ts404Params>,
    dsp: Dsp,
    oversample: Oversample<Sample>,
}

#[derive(Params)]
struct Ts404Params {
    #[id = "drive"]
    drive: FloatParam,
    #[id = "dist"]
    dist: FloatParam,
    #[id = "tone"]
    tone: FloatParam,
    #[id = "level"]
    out_level: FloatParam,
    #[id = "cmpmat"]
    component_matching: FloatParam,
    #[id = "bypass"]
    bypass: BoolParam,
    #[id = "byp_io"]
    io_bypass: BoolParam,
}

impl Default for Ts404 {
    fn default() -> Self {
        let samplerate = Sample::splat(OVERSAMPLE as f64 * 44100.0);
        Self {
            params: Arc::new(Ts404Params::default()),
            dsp: Series((
                Bypass::new(InputStage::new(samplerate, Sample::splat(1.0))),
                Bypass::new(Series((
                    ClipperStage::new(samplerate, Sample::splat(0.1)),
                    ToneStage::new(samplerate, Sample::splat(0.5)),
                ))),
                OutputStage::new(samplerate, Sample::splat(1.0)),
            )),
            oversample: Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE),
        }
    }
}

impl Default for Ts404Params {
    fn default() -> Self {
        Self {
            drive: FloatParam::new(
                "Drive",
                1.0,
                FloatRange::Skewed {
                    min: 0.5,
                    max: 100.0,
                    factor: FloatRange::gain_skew_factor(gain_to_db(0.5), gain_to_db(100.0)),
                },
            )
            .with_unit("dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            // .with_smoother(SmoothingStyle::Linear(50.0)),
            dist: FloatParam::new("Distortion", 0.1, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                // .with_smoother(SmoothingStyle::Linear(50.0))
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            tone: FloatParam::new("Tone", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                // .with_smoother(SmoothingStyle::Linear(50.0))
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            out_level: FloatParam::new(
                "Output Level",
                0.158,
                FloatRange::Skewed {
                    min: MINUS_INFINITY_GAIN,
                    max: 1.0,
                    factor: FloatRange::gain_skew_factor(MINUS_INFINITY_DB, gain_to_db(1.0)),
                },
            )
            .with_unit("dB")
            // .with_smoother(SmoothingStyle::Linear(50.0))
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            component_matching: FloatParam::new(
                "Component Matching",
                1.,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit("%")
            // .with_smoother(SmoothingStyle::Linear(10.0))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_value_to_string(formatters::v2s_f32_percentage(0)),
            bypass: BoolParam::new("Bypass", false),
            io_bypass: BoolParam::new("I/O Buffers Bypass", false),
        }
    }
}

fn component_matching_slew_rate(samplerate: Sample, normalized: Sample) -> Sample {
    let min = Sample::splat(db_to_gain_fast(30.0) as _);
    let max = Sample::splat(db_to_gain_fast(100.0) as _);
    lerp(normalized, min, max) / samplerate
}

fn component_matching_crossover(normalized: Sample) -> (Sample, Sample) {
    let min = Sample::splat(10.0);
    let max = Sample::splat(0.5);
    let a = lerp(normalized, min, max);
    let b = Sample::from_f64(10.0);
    (a, b)
}

impl Plugin for Ts404 {
    const NAME: &'static str = "TS-404";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "me@solarliner.dev";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
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
        let samplerate = Sample::splat(buffer_config.sample_rate as _);
        let Series((
            Bypass {
                inner: input,
                active: input_active,
            },
            Bypass {
                inner: Series((clipping, tone)),
                active: main_active,
            },
            output,
        )) = &mut self.dsp;

        let component_matching = Sample::from_f64(self.params.component_matching.value() as _);

        let io_active = !self.params.io_bypass.value();
        *main_active = !self.params.bypass.value();
        *input_active = io_active;
        output.inner.active = io_active;
        input.gain = Sample::splat(self.params.drive.value() as _);
        clipping.set_params(samplerate, Sample::splat(self.params.dist.value() as _));
        clipping.crossover = component_matching_crossover(component_matching);
        clipping.slew.max_diff = component_matching_slew_rate(samplerate, component_matching);
        tone.update_params(samplerate, Sample::splat(self.params.tone.value() as _));
        output.gain = Sample::splat(self.params.out_level.value() as _);
        true
    }

    fn reset(&mut self) {
        DSP::reset(&mut self.dsp);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let samplerate = Sample::splat(context.transport().sample_rate as _);
        let Series((
            Bypass {
                inner: input,
                active: input_active,
            },
            Bypass {
                inner: Series((clipping, tone)),
                active: main_active,
            },
            output,
        )) = &mut self.dsp;

        let component_matching = Sample::from_f64(self.params.component_matching.value() as _);
        let io_active = !self.params.io_bypass.value();
        *main_active = !self.params.bypass.value();
        *input_active = io_active;
        output.inner.active = io_active;
        input.gain = Sample::splat(self.params.drive.value() as _);
        clipping.set_params(samplerate, Sample::splat(self.params.dist.value() as _));
        clipping.crossover = component_matching_crossover(component_matching);
        clipping.slew.max_diff = component_matching_slew_rate(samplerate, component_matching);
        tone.update_params(samplerate, Sample::splat(self.params.tone.value() as _));
        output.gain = Sample::splat(self.params.out_level.value() as _);

        context.set_latency_samples(DSP::latency(&self.dsp) as _);

        let mut inner_buffer = [Sample::zero(); MAX_BLOCK_SIZE];
        let mut os_block_copy = [Sample::zero(); OVERSAMPLE * MAX_BLOCK_SIZE];
        for (_, mut block) in buffer.iter_blocks(MAX_BLOCK_SIZE) {
            for (i, mut frame) in block.iter_samples().enumerate() {
                let stereo = Sample::new(
                    *frame.get_mut(0).unwrap() as _,
                    *frame.get_mut(1).unwrap() as _,
                );
                inner_buffer[i] = stereo;
            }

            let inner_buffer = &mut inner_buffer[..block.samples()];
            let os_block_copy = &mut os_block_copy[..block.samples() * OVERSAMPLE];
            let mut os_block = self.oversample.oversample(inner_buffer);
            os_block_copy.copy_from_slice(&os_block);
            self.dsp.process_block(
                slice_to_mono_block(os_block_copy),
                slice_to_mono_block_mut(&mut os_block),
            );
            os_block.finish(inner_buffer);

            for (i, s) in inner_buffer.iter().copied().enumerate() {
                *block.get_mut(0).unwrap().get_mut(i).unwrap() = s.extract(0) as _;
                *block.get_mut(1).unwrap().get_mut(i).unwrap() = s.extract(1) as _;
            }
        }

        safety_clipper(buffer);

        ProcessStatus::Normal
    }
}

fn safety_clipper(buffer: &mut Buffer) {
    let max_ampl = db_to_gain_fast(8.0);
    for sample in buffer.iter_samples() {
        for s in sample.into_iter() {
            if !s.is_finite() || s.abs() > max_ampl {
                nih_debug_assert_failure!("Safety clip triggered");
                s.set_zero();
            }
        }
    }
}

impl ClapPlugin for Ts404 {
    const CLAP_ID: &'static str = "dev.solarliner.ts404";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("An inspired but fantasy screamer guitar pedal emulation");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for Ts404 {
    const VST3_CLASS_ID: [u8; 16] = *b"SolNrPluginTs404";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(Ts404);
nih_export_vst3!(Ts404);
