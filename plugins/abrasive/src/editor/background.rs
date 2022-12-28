use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;

pub(crate) struct Background;

impl Background {
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self.build(cx, |_| ())
    }
}

impl View for Background {
    fn element(&self) -> Option<&'static str> {
        Some("background")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let paint = vg::Paint::color(cx.background_color().copied().unwrap_or(Color::black()).into());
        let mut bg_path = vg::Path::new();
        let bounds = cx.bounds();
        bg_path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.fill_path(&mut bg_path, &paint);
    }
}