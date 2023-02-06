use std::f32::consts::{PI, TAU};

use nih_plug::nih_error;
use nih_plug_vizia::vizia::{prelude::*, vg};

pub struct Knob {
    value: f32,
    bipolar: bool,
}

impl Knob {
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self {
            value: 0.,
            bipolar: true,
        }
        .build(cx, |_| ())
    }
}

pub trait HandleKnobExt {
    fn value<L: Res<f32>>(self, value: L) -> Self;
    fn bipolar<L: Res<bool>>(self, value: L) -> Self;
}

impl HandleKnobExt for Handle<'_, Knob> {
    fn value<L: Res<f32>>(self, value: L) -> Self {
        let entity = self.entity;
        value.set_or_bind(self.cx, entity, move |cx, entity, value| {
            if let Some(view) = cx.views.get_mut(&entity) {
                if let Some(knob) = view.downcast_mut::<Knob>() {
                    knob.value = value;
                    cx.need_redraw();
                }
            }
        });
        self
    }

    fn bipolar<L: Res<bool>>(self, value: L) -> Self {
        let entity = self.entity;
        value.set_or_bind(self.cx, entity, move |cx, entity, value| {
            if let Some(view) = cx.views.get_mut(&entity) {
                if let Some(knob) = view.downcast_mut::<Knob>() {
                    knob.bipolar = value;
                    cx.need_redraw();
                }
            }
        });
        self
    }
}

impl View for Knob {
    fn element(&self) -> Option<&'static str> {
        Some("knob")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let clip_bounds = cx.clip_region();
        canvas.clear_rect(
            clip_bounds.x as _,
            clip_bounds.y as _,
            clip_bounds.w as _,
            clip_bounds.h as _,
            cx.background_color()
                .map(color2color)
                .unwrap_or(vg::Color::black()),
        );

        // Background
        let bounds = cx.bounds();
        let paint = get_fill(cx, canvas).unwrap_or(vg::Paint::color(vg::Color::black()));
        let mut path = vg::Path::new();
        path.rounded_rect_varying(
            bounds.x,
            bounds.y,
            bounds.w,
            bounds.h,
            cx.border_radius_top_left()
                .map(|u| u.value_or(0., 0.))
                .unwrap_or(0.),
            cx.border_radius_top_right()
                .map(|u| u.value_or(0., 0.))
                .unwrap_or(0.),
            cx.border_radius_bottom_left()
                .map(|u| u.value_or(0., 0.))
                .unwrap_or(0.),
            cx.border_radius_bottom_right()
                .map(|u| u.value_or(0., 0.))
                .unwrap_or(0.),
        );

        canvas.fill_path(&mut path, &paint);

        // Value arc + line
        let paint = get_stroke(cx, canvas).unwrap_or(vg::Paint::color(vg::Color::white()));
        let mut path = vg::Path::new();
        let (ctx, cty) = bounds.center();
        let radius = bounds.w.min(bounds.h) - 3.;
        let (start, end) = if self.bipolar {
            let start = 225f32.to_radians();
            let end = (start + 270f32.to_radians() * self.value) % TAU;
            (start, end)
        } else {
            (0., PI * self.value)
        };
        path.arc(ctx, cty, radius, start, end, vg::Solidity::Solid);
        let (s, c) = end.sin_cos();
        path.move_to(ctx, cty);
        path.line_to(radius * s, radius * c);

        canvas.stroke_path(&mut path, &paint);
    }
}

fn get_fill(cx: &mut DrawContext, canvas: &mut Canvas) -> Option<vg::Paint> {
    cx.background_image()
        .cloned()
        .and_then(|imgpath| load_image_file(cx, canvas, &imgpath))
        .or_else(|| {
            cx.background_gradient().map(|g| {
                let bounds = cx.bounds();
                let ((x0, y0), (x1, y1)) = match g.direction {
                    GradientDirection::BottomToTop => (bounds.center_bottom(), bounds.center_top()),
                    GradientDirection::TopToBottom => (bounds.center_top(), bounds.center_bottom()),
                    GradientDirection::LeftToRight => (bounds.center_left(), bounds.center_right()),
                    GradientDirection::RightToLeft => (bounds.center_right(), bounds.center_left()),
                };
                let stops: Vec<_> = g
                    .stops
                    .iter()
                    .map(|stop| (stop.position.value_or(0., 0.), color2color(&stop.color)))
                    .collect();
                vg::Paint::linear_gradient_stops(x0, y0, x1, y1, &stops)
            })
        })
        .or_else(|| cx.background_color().map(color2color).map(vg::Paint::color))
}

fn load_image_file(cx: &mut DrawContext, canvas: &mut Canvas, imgpath: &str) -> Option<vg::Paint> {
    let bounds = cx.bounds();
    let (cx, cy) = bounds.top_left();
    match canvas.load_image_file(imgpath, vg::ImageFlags::empty()) {
        Ok(img) => Some(vg::Paint::image(
            img,
            cx,
            cy,
            bounds.width(),
            bounds.height(),
            0.,
            1.,
        )),
        Err(err) => {
            nih_error!("Cannot get background image: {}", err);
            None
        }
    }
}

fn get_stroke(cx: &mut DrawContext, canvas: &mut Canvas) -> Option<vg::Paint> {
    cx.image()
        .cloned()
        .and_then(|imgpath| load_image_file(cx, canvas, &imgpath))
        .or_else(|| cx.font_color().map(color2color).map(vg::Paint::color))
        .map(|mut paint| {
            paint.set_line_width(
                cx.border_width()
                    .map(|units| units.value_or(1., 1.))
                    .unwrap_or(1.),
            );
            paint
        })
}

fn color2color(inp: &Color) -> vg::Color {
    vg::Color::rgba(inp.r(), inp.g(), inp.b(), inp.a())
}
