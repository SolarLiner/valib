use crate::editor::eq::LogRange;
use atomic_float::AtomicF32;
use nih_plug::{formatters, nih_error};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::vizia::vg::{ErrorKind, FontId, Paint};
use std::cell::Cell;
use std::hint;
use std::sync::Arc;

pub(crate) struct Background {
    samplerate: Arc<AtomicF32>,
    entity: Entity,
    font_id: Cell<Option<FontId>>,
}

impl Background {
    pub fn new(cx: &mut Context, samplerate: Arc<AtomicF32>) -> Handle<Self> {
        let entity = cx.current();
        Self {
            samplerate,
            entity,
            font_id: Cell::new(None),
        }
        .build(cx, |_| ())
    }

    fn anchor_text_bottom(
        canvas: &mut Canvas,
        bounds: BoundingBox,
        paint: &Paint,
        text: impl AsRef<str>,
    ) -> Option<f32> {
        match canvas.measure_text(0., 0., text, paint) {
            Ok(measure) => {
                let y = bounds.bottom() - measure.height();
                Some(y)
            }
            Err(err) => {
                nih_error!("Cannot draw text: {err} ({err:?})");
                None
            }
        }
    }

    fn get_font_id(&self, canvas: &mut Canvas) -> Option<FontId> {
        if self.font_id.get().is_none() {
            let path = super::util::resolve_asset_file("fonts/Metrophobic-Regular.ttf".as_ref());
            match canvas.add_font(path) {
                Ok(id) => {
                    self.font_id.set(Some(id));
                }
                Err(err) => {
                    nih_error!("Cannot add font: {err} {err:?}");
                }
            }
        }
        self.font_id.get()
    }
}

impl View for Background {
    fn element(&self) -> Option<&'static str> {
        Some("background")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let freq_range = LogRange::new(2.0, 20., 24e3);
        let gain_range = |db| /* -24..24 */ (db + 24.) / 48.;
        let bounds = cx.bounds();

        // Background
        let mut view_path = cx.build_path();
        cx.draw_backdrop_filter(canvas, &mut view_path);
        cx.draw_background(canvas, &mut view_path);

        // Grid
        let paint =
            vg::Paint::color(cx.font_color().into()).with_font_size(cx.font_size(self.entity));
        let paint = if let Some(id) = self.get_font_id(canvas) {
            paint.with_font(&[id])
        } else {
            paint
        };
        let mut grid_lines = vg::Path::new();
        // Vertical lines
        let freq_format = formatters::v2s_f32_hz_then_khz(0);
        for i in 1..=4 {
            let freq_base = 10f32.powi(i);
            let text = freq_format(freq_base);
            if let Some(y) = Self::anchor_text_bottom(canvas, bounds, &paint, &text) {
                let x = freq_range.normalize(freq_base);
                let x = bounds.left() + x * bounds.width();
                canvas.fill_text(x, y, &text, &paint).unwrap();
            }
            for s in 0..10 {
                let f = freq_base * s as f32;
                let x = freq_range.normalize(f);
                let x = bounds.left() + bounds.width() * x;
                grid_lines.move_to(x, bounds.top());
                grid_lines.line_to(x, bounds.bottom());
            }
        }
        // Horizontal lines
        for gain in (-20..20).step_by(10) {
            let text = format!("{gain:+2} dB");
            let y = gain_range(gain as f32);
            let y = bounds.bottom() - y * bounds.height();
            if let Err(err) = canvas.fill_text(0.0, y, &text, &paint) {
                nih_error!("Cannot draw text: {err} ({err:?})");
            }
            grid_lines.move_to(bounds.left(), y);
            grid_lines.line_to(bounds.right(), y);
        }
        canvas.stroke_path(&grid_lines, &paint);
    }
}
