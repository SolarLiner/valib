use std::sync::{atomic::Ordering, Arc};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::{context::DrawContext, prelude::*, vg};

use crate::dsp::{filter::FilterParams, Equalizer};
use crate::AbrasiveParams;
use valib::dsp::analysis::DspAnalysis;
use valib::simd::SimdComplexField;

pub struct LogRange {
    base: f32,
    log_min: f32,
    log_max: f32,
}

impl LogRange {
    pub fn new(base: f32, min: f32, max: f32) -> Self {
        Self {
            base,
            log_min: min.log(base),
            log_max: max.log(base),
        }
    }

    pub fn normalize(&self, x: f32) -> f32 {
        let x = x.log(self.base);
        (x - self.log_min) / (self.log_max - self.log_min)
    }

    pub fn unnormalize(&self, x: f32) -> f32 {
        let x = self.log_min + x * (self.log_max - self.log_min);
        self.base.powf(x)
    }
}

#[derive(Debug, Clone)]
struct EqData {
    samplerate: Arc<AtomicF32>,
    params: Arc<AbrasiveParams>,
    frequency_range: FloatRange,
    // gain_range: FloatRange,
    modulated: bool,
}

impl View for EqData {
    fn element(&self) -> Option<&'static str> {
        Some("eq")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let samplerate = self.samplerate.load(Ordering::Relaxed);
        let range = LogRange::new(2.0, 20., samplerate.min(24e3));

        let mut dsp = Equalizer::new(samplerate, self.params.dsp_params.clone());
        dsp.use_param_values(self.modulated);
        let bounds = cx.bounds();
        let paint = vg::Paint::color(cx.font_color().into()).with_line_width(cx.border_width());
        let mut path = vg::Path::new();

        for j in 0..4 * bounds.w as usize {
            let x = j as f32 / (4. * bounds.w);
            let freq = range.unnormalize(x);
            let [[y]] = dsp.freq_response(samplerate, freq);
            let y = (util::gain_to_db(y.simd_abs()) + 24.) / 48.;
            if j == 0 {
                path.move_to(bounds.x + bounds.w * x, bounds.y + bounds.h * (1. - y));
            } else {
                path.line_to(bounds.x + bounds.w * x, bounds.y + bounds.h * (1. - y));
            }
        }

        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.stroke_path(&path, &paint);
    }
}

impl EqData {
    pub fn new(samplerate: Arc<AtomicF32>, params: Arc<AbrasiveParams>, modulated: bool) -> Self {
        Self {
            samplerate,
            params,
            frequency_range: FilterParams::cutoff_range(),
            // gain_range: FloatRange::Skewed {
            //     min: util::db_to_gain(-24.),
            //     max: util::db_to_gain(24.),
            //     factor: FloatRange::gain_skew_factor(-24., 24.),
            // },
            modulated,
        }
    }
}

pub(crate) fn build(
    cx: &mut Context,
    samplerate: impl Res<Arc<AtomicF32>>,
    params: impl Res<Arc<AbrasiveParams>>,
) -> Handle<impl View> {
    let samplerate = samplerate.get_val(cx);
    let params = params.get_val(cx);

    ZStack::new(cx, |cx| {
        EqData::new(samplerate.clone(), params.clone(), true)
            .build(cx, |_| ())
            .class("modulated");
        EqData::new(samplerate, params, false)
            .build(cx, |_| ())
            .class("unmodulated");
    })
}
