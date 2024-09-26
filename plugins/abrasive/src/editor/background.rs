use crate::editor::eq::LogRange;
use atomic_float::AtomicF32;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub(crate) struct Background {
    samplerate: Arc<AtomicF32>,
}

impl Background {
    pub fn new(cx: &mut Context, samplerate: Arc<AtomicF32>) -> Handle<Self> {
        Self { samplerate }.build(cx, |_| ())
    }
}

impl View for Background {
    fn element(&self) -> Option<&'static str> {
        Some("background")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let samplerate = self.samplerate.load(Ordering::Relaxed);
        let range = LogRange::new(2.0, 20., samplerate.min(24e3));
        // Background
        let paint = vg::Paint::color(cx.background_color().into());
        let mut bg_path = vg::Path::new();
        let bounds = cx.bounds();
        bg_path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.fill_path(&bg_path, &paint);

        // Grid
        let mut grid_lines = vg::Path::new();
        for i in 1..=4 {
            for s in 0..10 {
                let f = 10f32.powi(i) * s as f32;
                let x = range.normalize(f);
                let x = bounds.left() + bounds.width() * x;
                grid_lines.move_to(x, bounds.top());
                grid_lines.line_to(x, bounds.bottom());
            }
        }
        let paint = vg::Paint::color(cx.font_color().into());
        canvas.stroke_path(&grid_lines, &paint);
    }
}
