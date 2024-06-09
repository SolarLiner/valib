use valib::dsp::parameter::RemoteControl;
use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver, Sender};
use nih_plug::params::{BoolParam, FloatParam, Params};
use nih_plug::prelude::{AtomicF32, FloatRange};
use nih_plug::util::{gain_to_db, MINUS_INFINITY_DB, MINUS_INFINITY_GAIN};
use nih_plug::formatters;
use nih_plug_iced::IcedState;
use valib::contrib::nih_plug::BindToParameter;
use crate::dsp::DspParams;

#[derive(Params)]
pub(crate) struct Ts404Params {
    #[id = "drive"]
    pub(crate) drive: FloatParam,
    #[id = "dist"]
    pub(crate) dist: FloatParam,
    #[id = "tone"]
    pub(crate) tone: FloatParam,
    #[id = "level"]
    pub(crate) out_level: FloatParam,
    #[id = "cmpmat"]
    pub(crate) component_matching: FloatParam,
    #[id = "bypass"]
    pub(crate) bypass: BoolParam,
    #[id = "byp_io"]
    pub(crate) io_bypass: BoolParam,
    #[persist="editor_state"]
    pub(crate) editor_state: Arc<IcedState>,
}

impl Ts404Params {
    pub(crate) fn new(remote: &RemoteControl<DspParams>) -> Arc<Self> {
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
            editor_state: crate::editor::default_state(),
        })
    }
}
