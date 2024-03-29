use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;

pub(crate) struct Background;

impl Background {
    #[cfg(never)]
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self.build(cx, |_| ())
    }
}

impl View for Background {
    fn element(&self) -> Option<&'static str> {
        Some("background")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let paint = vg::Paint::color(cx.background_color().into());
        let mut bg_path = vg::Path::new();
        let bounds = cx.bounds();
        bg_path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.fill_path(&bg_path, &paint);
    }
}
