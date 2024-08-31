use crate::dsp::{DspParams, InputLevelMatching};
use nih_plug::formatters;
use nih_plug::params::{BoolParam, EnumParam, FloatParam, Params};
use nih_plug::prelude::{AtomicF32, FloatRange};
use nih_plug::util::{gain_to_db, MINUS_INFINITY_DB, MINUS_INFINITY_GAIN};
use nih_plug_vizia::ViziaState;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use valib::contrib::nih_plug::BindToParameter;
use valib::dsp::parameter::RemoteControl;

pub(crate) const MAX_AGE: f32 = 100.;

#[derive(Params)]
pub(crate) struct Ts404Params {
    #[id = "drive"]
    pub(crate) input_mode: EnumParam<InputLevelMatching>,
    #[id = "dist"]
    pub(crate) dist: FloatParam,
    #[id = "tone"]
    pub(crate) tone: FloatParam,
    #[id = "cmpmat"]
    pub(crate) component_matching: FloatParam,
    #[id = "bypass"]
    pub(crate) bypass: BoolParam,
    #[persist = "editor_state"]
    pub(crate) editor_state: Arc<ViziaState>,
}

impl Ts404Params {
    pub(crate) fn new(remote: &RemoteControl<DspParams>) -> Arc<Self> {
        Arc::new(Self {
            input_mode: EnumParam::new("Input mode", InputLevelMatching::Line)
                .bind_to_parameter(remote, DspParams::InputMode),
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
            component_matching: FloatParam::new(
                "Age",
                1.,
                FloatRange::Linear {
                    min: 0.0,
                    max: MAX_AGE,
                },
            )
            .with_unit(" yr")
            .bind_to_parameter(remote, DspParams::ComponentMismatch),
            bypass: BoolParam::new("Bypass", false).bind_to_parameter(remote, DspParams::Bypass),
            editor_state: crate::editor::default_state(),
        })
    }
}
