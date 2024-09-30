use crate::editor::eq::LogRange;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::{prelude::*, vg};

pub struct SpectrumAnalyzer {
    spectrum: super::SpectrumUI,
}

impl SpectrumAnalyzer {
    pub fn new(cx: &mut Context, spectrum: super::SpectrumUI) -> Handle<Self> {
        Self { spectrum }.build(cx, |_cx| ())
    }

    fn draw_analyzer(&self, cx: &mut DrawContext, canvas: &mut Canvas, bounds: BoundingBox) {
        let mut spectrum = self.spectrum.lock().unwrap();
        let spectrum = spectrum.read();
        let samplerate = spectrum.samplerate;
        let nyquist = samplerate / 2.0;
        let range = LogRange::new(2.0, 20.0, 24e3);
        let line_paint = vg::Paint::color(cx.font_color().into());

        let mut path = vg::Path::new();

        for (i, y) in spectrum.data.iter().copied().enumerate() {
            if i == 0 {
                path.move_to(bounds.x - 100., bounds.y + bounds.h);
                continue;
            }

            let freq_norm = i as f32 / spectrum.data.len() as f32;
            let frequency = freq_norm * nyquist;
            let x = range.normalize(frequency);
            let slope = 3.;
            let octavediff = frequency.log2() - 1000f32.log2();
            let octavegain = slope * octavediff;
            let h = (octavegain + util::gain_to_db(y) + 80.) / 80.;

            path.line_to(bounds.x + bounds.w * x, bounds.y + bounds.h * (1. - h));
        }

        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.stroke_path(&path, &line_paint);
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
