use std::sync::{atomic::Ordering, Arc};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::{cache::BoundingBox, prelude::*, vg};

use crate::filter::FilterParams;

pub struct SpectrumAnalyzer {
    spectrum: super::SpectrumUI,
    samplerate: Arc<AtomicF32>,
    frange: FloatRange,
}

impl SpectrumAnalyzer {
    pub fn new(
        cx: &mut Context,
        spectrum: super::SpectrumUI,
        samplerate: Arc<AtomicF32>,
    ) -> Handle<Self> {
        Self {
            spectrum,
            samplerate,
            frange: FilterParams::cutoff_range(),
        }
        .build(cx, |_cx| ())
    }

    fn draw_analyzer(&self, cx: &mut DrawContext, canvas: &mut Canvas, bounds: BoundingBox) {
        let line_width = cx.style.dpi_factor as f32 * 1.5;
        let line_paint =
            vg::Paint::color(cx.font_color().cloned().unwrap_or(Color::white()).into())
                .with_line_width(line_width);

        let mut path = vg::Path::new();

        let mut spectrum = self.spectrum.lock().unwrap();
        let spectrum = spectrum.read();
        let nyquist = self.samplerate.load(Ordering::Relaxed) / 2.0;
        for (i, y) in spectrum.data.iter().copied().enumerate() {
            if i == 0 {
                path.move_to(bounds.x - 100., bounds.y + bounds.h);
                continue;
            }

            let freq_norm = i as f32 / spectrum.data.len() as f32;
            let frequency = freq_norm * nyquist;
            let x = self.frange.normalize(frequency);
            let slope = 3.;
            let octavediff = frequency.log2() - 1000f32.log2();
            let octavegain = slope * octavediff;
            let h = (octavegain + util::gain_to_db(y) + 80.) / 80.;

            path.line_to(bounds.x + bounds.w * x, bounds.y + bounds.h * (1. - h));
        }

        canvas.stroke_path(&mut path, &line_paint);
    }
}

impl View for SpectrumAnalyzer {
    fn element(&self) -> Option<&'static str> {
        Some("spectrum")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w == 0.0 || bounds.h == 0.0 {
            return;
        }

        self.draw_analyzer(cx, canvas, bounds);
    }
}
