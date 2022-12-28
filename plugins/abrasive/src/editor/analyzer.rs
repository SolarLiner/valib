use crate::{filter::FilterParams, spectrum::Spectrum};
use atomic_float::AtomicF32;
use nih_plug::{params, prelude::*};
use nih_plug_vizia::vizia::{cache::BoundingBox, prelude::*, vg};
use std::sync::{atomic::Ordering, Arc, Mutex};
use triple_buffer::Output;

/// A very abstract spectrum analyzer. This draws the magnitude spectrum's bins as vertical lines
/// with the same distirubtion as the filter frequency parmaeter..
pub struct SpectrumAnalyzer {
    spectrum: super::SpectrumUI,
    sample_rate: Arc<atomic_float::AtomicF32>,
    frequency_range: FloatRange,
}

impl SpectrumAnalyzer {
    /// Creates a new [`SpectrumAnalyzer`]. The uses custom drawing.
    pub fn new<LSpectrum, LRate>(
        cx: &mut Context,
        spectrum: LSpectrum,
        sample_rate: LRate,
    ) -> Handle<Self>
    where
        LSpectrum: Lens<Target = Arc<Mutex<Output<Spectrum>>>>,
        LRate: Lens<Target = Arc<AtomicF32>>,
    {
        Self {
            spectrum: spectrum.get(cx),
            sample_rate: sample_rate.get(cx),

            frequency_range: FilterParams::cutoff_range(),
        }
        .build(
            cx,
            // This is an otherwise empty element only used for custom drawing
            |_cx| (),
        )
    }

    fn draw_analyzer(&self, cx: &mut DrawContext, canvas: &mut Canvas, bounds: BoundingBox) {
        let line_width = cx.style.dpi_factor as f32 * 1.5;
        let line_paint =
            vg::Paint::color(cx.font_color().cloned().unwrap_or(Color::white()).into())
                .with_line_width(line_width);

        let mut path = vg::Path::new();
        path.move_to(bounds.x, bounds.y + bounds.h);

        let mut spectrum = self.spectrum.lock().unwrap();
        let spectrum = spectrum.read();
        let nyquist = self.sample_rate.load(Ordering::Relaxed) / 2.0;
        for (bin_idx, y) in spectrum.data.iter().copied().enumerate() {
            let freq_norm = bin_idx as f32 / spectrum.data.len() as f32;
            let frequency = freq_norm * nyquist;
            let x = self.frequency_range.normalize(frequency);
            if x < 0.0 || x >= 1.0 {
                continue;
            }
            let slope = 4.5;
            let octavegain = util::db_to_gain(slope) * f32::log2(bin_idx as f32 + 1.) / 20.;
            let h = (util::gain_to_db(y * octavegain) + 80.) / 80.;

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
