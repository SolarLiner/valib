use std::sync::Arc;
use std::sync::atomic::Ordering;

use nih_plug::prelude::AtomicF32;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;

pub(crate) struct Led {
    drive_led: Arc<AtomicF32>,
}

impl Led {
    pub(crate) fn new(cx: &mut Context, drive_led: Arc<AtomicF32>) -> Handle<Self> {
        Self { drive_led }.build(cx, |_| ())
    }
}

impl View for Led {
    fn element(&self) -> Option<&'static str> {
        Some("led")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        let (x, y) = bounds.center();
        let size = bounds.w.min(bounds.h) / 2.;

        let color = cx.background_color();
        let drive_led = self.drive_led.load(Ordering::Relaxed);
        let brightness = 1. - f32::exp(-drive_led / 50.);
        let paint = vg::Paint::color(vg::Color::rgba(
            color.r(),
            color.g(),
            color.b(),
            (brightness * 255.) as _,
        ));

        let mut path = vg::Path::new();
        path.circle(x, y, size);

        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.fill_path(&path, &paint);
    }
}
