use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug::util::{gain_to_db, MINUS_INFINITY_DB, MINUS_INFINITY_GAIN};
use valib::contrib::nih_plug::{BindToParameter, process_buffer_simd64};
use valib::dsp::DSPMeta;
use valib::dsp::parameter::{RemoteControl, RemoteControlled};
use valib::simd::{AutoF32x2, AutoF64x2};

use dsp::MAX_BLOCK_SIZE;

use crate::dsp::{Dsp, DspParams};

mod dsp;
mod gen;

const TARGET_SAMPLERATE: f32 = 96000.;

type Sample = AutoF64x2;
type Sample32 = AutoF32x2;

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

impl Ts404Params {
    fn new(remote: &RemoteControl<DspParams>) -> Arc<Self> {
        Arc::new(Self {
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
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .bind_to_parameter(remote, DspParams::InputGain),
            dist: FloatParam::new("Distortion", 0.1, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_string_to_value(formatters::s2v_f32_percentage())
                .bind_to_parameter(remote, DspParams::Distortion),
            tone: FloatParam::new("Tone", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_string_to_value(formatters::s2v_f32_percentage())
                .bind_to_parameter(remote, DspParams::Tone),
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
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .bind_to_parameter(remote, DspParams::OutputGain),
            component_matching: FloatParam::new(
                "Component Matching",
                1.,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit("%")
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .bind_to_parameter(remote, DspParams::ComponentMismatch),
            bypass: BoolParam::new("Bypass", false).bind_to_parameter(remote, DspParams::Bypass),
            io_bypass: BoolParam::new("I/O Buffers Bypass", false)
                .bind_to_parameter(remote, DspParams::BufferBypass),
        })
    }
}

struct Ts404 {
    params: Arc<Ts404Params>,
    dsp: RemoteControlled<Dsp<Sample>>,
}

impl Default for Ts404 {
    fn default() -> Self {
        let default_samplerate = 44100.0;
        let dsp = Dsp::new(default_samplerate, TARGET_SAMPLERATE);
        let params = Ts404Params::new(&dsp.proxy);
        Self { dsp, params }
    }
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
        let dsp = Dsp::new(buffer_config.sample_rate, TARGET_SAMPLERATE);
        self.dsp.inner = dsp.inner;

        let dsp = &self.dsp;
        dsp.proxy
            .set_parameter(DspParams::InputGain, self.params.drive.value());
        dsp.proxy
            .set_parameter(DspParams::Distortion, self.params.dist.value());
        dsp.proxy
            .set_parameter(DspParams::Tone, self.params.tone.value());
        dsp.proxy
            .set_parameter(DspParams::OutputGain, self.params.out_level.value());
        dsp.proxy.set_parameter(
            DspParams::ComponentMismatch,
            self.params.component_matching.value(),
        );
        dsp.proxy.set_parameter(
            DspParams::Bypass,
            if self.params.bypass.value() { 1.0 } else { 0.0 },
        );
        dsp.proxy.set_parameter(
            DspParams::BufferBypass,
            if self.params.io_bypass.value() {
                1.0
            } else {
                0.0
            },
        );

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
        process_buffer_simd64::<_, _, MAX_BLOCK_SIZE>(&mut self.dsp, buffer);
        //safety_clipper(buffer);

        ProcessStatus::Normal
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
