use crate::{
    OVERSAMPLE, POLYMOD_FILTER_CUTOFF, POLYMOD_OSC_AMP, POLYMOD_OSC_PITCH_COARSE,
    POLYMOD_OSC_PITCH_FINE,
};
use nih_plug::prelude::*;
use nih_plug::util::{db_to_gain, MINUS_INFINITY_DB};
use std::sync::Arc;
use valib::dsp::parameter::{ParamId, ParamName};

#[derive(Debug, Copy, Clone, Eq, PartialEq, ParamName, Enum)]
pub enum OscShape {
    Sine,
    Triangle,
    Square,
    Saw,
}

#[derive(Debug, Params)]
pub struct OscParams {
    #[id = "shp"]
    pub shape: EnumParam<OscShape>,
    #[id = "amp"]
    pub amplitude: FloatParam,
    #[id = "pco"]
    pub pitch_coarse: FloatParam,
    #[id = "pfi"]
    pub pitch_fine: FloatParam,
    #[id = "pw"]
    pub pulse_width: FloatParam,
}

impl OscParams {
    fn new(osc_index: usize, oversample: Arc<AtomicF32>) -> Self {
        Self {
            shape: EnumParam::new("Shape", OscShape::Saw),
            amplitude: FloatParam::new(
                "Amplitude",
                0.8,
                FloatRange::Skewed {
                    min: db_to_gain(MINUS_INFINITY_DB),
                    max: 1.0,
                    factor: FloatRange::gain_skew_factor(MINUS_INFINITY_DB, 0.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversample.clone(),
                &SmoothingStyle::Exponential(10.),
            ))
            .with_poly_modulation_id(POLYMOD_OSC_AMP[osc_index]),
            pitch_coarse: FloatParam::new(
                "Pitch (Coarse)",
                0.0,
                FloatRange::Linear {
                    min: -24.,
                    max: 24.,
                },
            )
            .with_step_size(1.)
            .with_unit(" st")
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversample.clone(),
                &SmoothingStyle::Exponential(10.),
            ))
            .with_poly_modulation_id(POLYMOD_OSC_PITCH_COARSE[osc_index]),
            pitch_fine: FloatParam::new(
                "Pitch (Fine)",
                0.0,
                FloatRange::Linear {
                    min: -0.5,
                    max: 0.5,
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(3))
            .with_unit(" st")
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversample.clone(),
                &SmoothingStyle::Exponential(10.),
            ))
            .with_poly_modulation_id(POLYMOD_OSC_PITCH_FINE[osc_index]),
            pulse_width: FloatParam::new(
                "Pulse Width",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(2))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversample.clone(),
                &SmoothingStyle::Linear(10.),
            )),
        }
    }
}

#[derive(Debug, Params)]
pub struct FilterParams {
    #[id = "fc"]
    pub cutoff: FloatParam,
    #[id = "res"]
    pub resonance: FloatParam,
}

impl FilterParams {
    fn new(oversample: Arc<AtomicF32>) -> Self {
        Self {
            cutoff: FloatParam::new(
                "Cutoff",
                3000.,
                FloatRange::Skewed {
                    min: 20.,
                    max: 20e3,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_hz_then_khz_with_note_name(2, true))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversample.clone(),
                &SmoothingStyle::Exponential(10.),
            ))
            .with_poly_modulation_id(POLYMOD_FILTER_CUTOFF),
            resonance: FloatParam::new(
                "Resonance",
                0.1,
                FloatRange::Linear {
                    min: 0.0,
                    max: 1.25,
                },
            )
            .with_value_to_string(formatters::v2s_f32_percentage(1))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_unit(" %")
            .with_smoother(SmoothingStyle::OversamplingAware(
                oversample.clone(),
                &SmoothingStyle::Linear(10.),
            )),
        }
    }
}

#[derive(Debug, Params)]
pub struct PolysynthParams {
    #[nested(array)]
    pub osc_params: [Arc<OscParams>; 2],
    #[nested]
    pub filter_params: Arc<FilterParams>,
    pub oversample: Arc<AtomicF32>,
}

impl Default for PolysynthParams {
    fn default() -> Self {
        let oversample = Arc::new(AtomicF32::new(OVERSAMPLE as _));
        Self {
            osc_params: std::array::from_fn(|i| Arc::new(OscParams::new(i, oversample.clone()))),
            filter_params: Arc::new(FilterParams::new(oversample.clone())),
            oversample,
        }
    }
}
