use std::{
    f32::consts::{FRAC_PI_2, PI, TAU},
    ops::Neg,
};

use nih_plug::{nih_error, nih_log, prelude::Param};
use nih_plug_vizia::{
    vizia::{prelude::*, vg},
    widgets::param_base::ParamWidgetBase,
};

enum KnobEvents {
    SetValueNormalized(f32),
    EnterValuePlain(String),
}

pub struct Knob {
    widget_base: ParamWidgetBase,
    bipolar: bool,
    drag_start: Option<(f32, f32)>,
}

impl Knob {
    pub fn new<Params: 'static, P: Param>(
        cx: &mut Context,
        bipolar: bool,
        params: impl Lens<Target = Params>,
        get_param: impl 'static + Copy + Fn(&Params) -> &P,
    ) -> Handle<Self> {
        Self {
            widget_base: ParamWidgetBase::new(cx, params, get_param),
            bipolar,
            drag_start: None,
        }
        .build(cx, |_| ())
            .min_width(Pixels(32.0))
            .min_height(Pixels(32.0))
    }
}

impl View for Knob {
    fn element(&self) -> Option<&'static str> {
        Some("knob")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|event, _| {
            match event {
                WindowEvent::MouseDown(MouseButton::Left) => {
                    self.drag_start = Some((
                        cx.mouse().cursory,
                        self.widget_base.unmodulated_normalized_value(),
                    ));
                    self.widget_base.begin_set_parameter(cx);
                    cx.set_active(true);
                }
                WindowEvent::MouseUp(MouseButton::Left) => {
                    self.drag_start = None;
                    self.widget_base.end_set_parameter(cx);
                }
                &WindowEvent::MouseMove(_, y) => {
                    if let Some((start_y, start_normalized_value)) = self.drag_start {
                        let speed = if cx.modifiers().contains(Modifiers::SHIFT) {
                            5e-4
                        } else {
                            5e-3
                        };
                        let normalized_value = start_normalized_value - speed * (y - start_y);
                        nih_log!("Normalized value: {normalized_value}");
                        self.widget_base
                            .set_normalized_value(cx, normalized_value);
                    }
                }
                &WindowEvent::MouseScroll(_, y) => {
                    let step = if cx.modifiers().contains(Modifiers::SHIFT) {
                        0.01
                    } else {
                        0.1
                    };
                    let normalized_value = self.widget_base.unmodulated_normalized_value() +  y*step;
                    nih_log!("Scroll -- normalized value: {normalized_value}");
                    self.widget_base.begin_set_parameter(cx);
                    self.widget_base.set_normalized_value(cx, normalized_value);
                    self.widget_base.end_set_parameter(cx);
                }
                _ => {}
            }
        });
        event.map(|event, _| {
            self.widget_base.begin_set_parameter(cx);
            match event {
                KnobEvents::EnterValuePlain(value) => {
                    if let Some(normalized) = self.widget_base.string_to_normalized_value(value) {
                        self.widget_base.set_normalized_value(cx, normalized);
                    }
                }
                &KnobEvents::SetValueNormalized(value) => {
                    self.widget_base.set_normalized_value(cx, value)
                }
            }
            self.widget_base.end_set_parameter(cx);
        });
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        // Background
        let mut path = vg::Path::new();
        path.rounded_rect(
            bounds.x,
            bounds.y,
            bounds.w,
            bounds.h,
            // cx.border_radius_top_left()
            //     .map(|u| u.value_or(0., 0.))
            //     .unwrap_or(0.),
            // cx.border_radius_top_right()
            //     .map(|u| u.value_or(0., 0.))
            //     .unwrap_or(0.),
            // cx.border_radius_bottom_left()
            //     .map(|u| u.value_or(0., 0.))
            //     .unwrap_or(0.),
            // cx.border_radius_bottom_right()
            //     .map(|u| u.value_or(0., 0.))
            //     .unwrap_or(0.),
            Percentage(100.).to_px(bounds.w, 0.1 * bounds.w),
            // Percentage(100.).value_or(bounds.w, 0.1 * bounds.w),
            // Percentage(100.).value_or(bounds.w, 0.1 * bounds.w),
            // Percentage(100.).value_or(bounds.w, 0.1 * bounds.w),
        );
        cx.draw_shadows(canvas, &mut path);
        cx.draw_background(canvas, &mut path);

        // Value arc + line
        let mut path = vg::Path::new();
        let (ctx, cty) = bounds.center();
        let radius = bounds.w.min(bounds.h) / 2. - 3.;
        let start = if self.bipolar {
            3. * FRAC_PI_2
        } else {
            135f32.to_radians()
        };
        let value = self.widget_base.modulated_normalized_value();
        let end = (135f32.to_radians() + 270f32.to_radians() * value) % TAU;
        let solidity = if self.bipolar && value < 0.5 {
            vg::Solidity::Solid
        } else {
            vg::Solidity::Hole
        };
        path.arc(ctx, cty, radius, start, end, solidity);
        let (s, c) = (-end + FRAC_PI_2).sin_cos();
        path.move_to(ctx, cty);
        path.line_to(ctx + radius * s, cty + radius * c);
        canvas.stroke_path(&mut path, &get_stroke(cx));
    }
}

fn get_stroke(cx: &mut DrawContext) -> vg::Paint {
    vg::Paint::color(cx.outline_color().into()).with_line_width(cx.outline_width())
}
