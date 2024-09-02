use std::f32::consts::{FRAC_PI_2, TAU};

use nih_plug::{nih_error, nih_log, prelude::Param};
use nih_plug_vizia::vizia::vg::{ImageFlags, ImageId};
use nih_plug_vizia::{
    vizia::{prelude::*, vg},
    widgets::param_base::ParamWidgetBase,
};

enum KnobEvents {
    SetValueNormalized(f32),
    //EnterValuePlain(String),
}

struct Arc {
    widget_base: ParamWidgetBase,
    get_normalized_value: Box<dyn Fn(&ParamWidgetBase) -> f32>,
    bipolar: bool,
}

impl Arc {
    pub fn new<Params: 'static, P: Param>(
        cx: &mut Context,
        bipolar: bool,
        params: impl Lens<Target = Params>,
        get_param: impl 'static + Copy + Fn(&Params) -> &P,
        get_normalized_value: impl 'static + Fn(&ParamWidgetBase) -> f32,
    ) -> Handle<Self> {
        Self {
            widget_base: ParamWidgetBase::new(cx, params, get_param),
            bipolar,
            get_normalized_value: Box::new(get_normalized_value),
        }
        .build(cx, |_| ())
    }
}

fn get_element_radius(bounds: BoundingBox) -> f32 {
    bounds.w.min(bounds.h) / 2. - 3.
}

impl View for Arc {
    fn element(&self) -> Option<&'static str> {
        Some("arc")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();

        let mut path = vg::Path::new();
        let (ctx, cty) = bounds.center();
        let radius = get_element_radius(bounds);
        let start = if self.bipolar {
            3. * FRAC_PI_2
        } else {
            135f32.to_radians()
        };
        let value = (self.get_normalized_value)(&self.widget_base);
        let end = get_angle(value);
        let solidity = if self.bipolar && value < 0.5 {
            vg::Solidity::Solid
        } else {
            vg::Solidity::Hole
        };
        path.arc(ctx, cty, radius, start, end, solidity);
        let (s, c) = (-end + FRAC_PI_2).sin_cos();
        path.move_to(ctx, cty);
        path.line_to(ctx + radius * s, cty + radius * c);
        canvas.stroke_path(&path, &get_stroke(cx));
    }
}

fn get_stroke(cx: &mut DrawContext) -> vg::Paint {
    vg::Paint::color(cx.font_color().into()).with_line_width(cx.border_width())
}

fn get_angle(t: f32) -> f32 {
    let deg = 135f32 + t * 270f32;
    deg.to_radians() % TAU
}

struct Ring {
    widget_base: ParamWidgetBase,
}

impl Ring {
    pub fn new<Params: 'static, P: Param>(
        cx: &mut Context,
        params: impl Lens<Target = Params>,
        get_param: impl 'static + Copy + Fn(&Params) -> &P,
    ) -> Handle<Self> {
        Self {
            widget_base: ParamWidgetBase::new(cx, params, get_param),
        }
        .build(cx, |_| ())
    }
}

impl View for Ring {
    fn element(&self) -> Option<&'static str> {
        Some("ring")
    }

    fn draw(&self, ctx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = ctx.bounds();
        let unmodulated = get_angle(self.widget_base.unmodulated_normalized_value());
        let modulated = get_angle(self.widget_base.modulated_normalized_value());
        let mut path = vg::Path::new();
        let (cx, cy) = bounds.center();
        path.arc(
            cx,
            cy,
            get_element_radius(bounds),
            unmodulated.min(modulated),
            unmodulated.max(modulated),
            if (unmodulated - modulated).abs() < 0.5 {
                vg::Solidity::Solid
            } else {
                vg::Solidity::Hole
            },
        );
        canvas.stroke_path(&path, &get_stroke(ctx));
    }
}

#[derive(Debug, Clone, Lens)]
struct KnobModel {
    display_textbox: bool,
    text: String,
}

enum KnobModelEvent {
    SetDisplayTextbox(bool),
}

impl Model for KnobModel {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|ev, _| match ev {
            &KnobModelEvent::SetDisplayTextbox(disp) => {
                self.display_textbox = disp;
                if !disp {
                    self.text.clear();
                }
            }
        });
    }
}

struct Dot {
    widget_base: ParamWidgetBase,
}

impl View for Dot {
    fn element(&self) -> Option<&'static str> {
        Some("dot")
    }
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        let (x, y) = bounds.center();
        let r = bounds.w.min(bounds.h) * 0.35;
        let value = self.widget_base.unmodulated_normalized_value();
        let angle = get_angle(0.7 - value);
        let (x, y) = (x + r * angle.sin(), y + r * angle.cos());

        let color = cx.font_color();
        let paint = vg::Paint::color(color.into());
        let mut path = vg::Path::new();
        path.circle(x, y, 3.3);

        canvas.fill_path(&path, &paint);
    }
}

pub struct Knob {
    widget_base: ParamWidgetBase,
    dragging: bool,
    image_bg: Option<ImageId>,
    image_fg: Option<ImageId>,
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
            dragging: false,
            image_fg: None,
            image_bg: None,
        }
        .build(cx, |cx| {
            KnobModel {
                display_textbox: false,
                text: String::new(),
            }
            .build(cx);
        })
    }
}

impl View for Knob {
    fn element(&self) -> Option<&'static str> {
        Some("knob")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|event, _| match event {
            WindowEvent::MouseDown(MouseButton::Left)
                if !cx.modifiers().contains(Modifiers::CTRL) =>
            {
                self.dragging = true;
                self.widget_base.begin_set_parameter(cx);
                cx.set_active(true);
            }
            WindowEvent::MouseUp(MouseButton::Left) => {
                self.dragging = false;
                self.widget_base.end_set_parameter(cx);
            }
            &WindowEvent::MouseMove(..) => {
                if self.dragging {
                    let speed = if cx.modifiers().contains(Modifiers::SHIFT) {
                        5e-4
                    } else {
                        5e-3
                    };
                    let normalized_value = self.widget_base.unmodulated_normalized_value()
                        - speed * cx.mouse().frame_delta().1;
                    nih_log!(
                        "[{}] Normalized value: {normalized_value}",
                        self.widget_base.name()
                    );
                    self.widget_base.set_normalized_value(cx, normalized_value);
                }
            }
            &WindowEvent::MouseScroll(_, y) => {
                let step = if cx.modifiers().contains(Modifiers::SHIFT) {
                    0.01
                } else {
                    0.1
                };
                let normalized_value = self.widget_base.unmodulated_normalized_value() + y * step;
                self.widget_base.begin_set_parameter(cx);
                self.widget_base.set_normalized_value(cx, normalized_value);
                self.widget_base.end_set_parameter(cx);
            }
            WindowEvent::MouseDown(MouseButton::Left)
                if cx.modifiers().contains(Modifiers::CTRL) =>
            {
                cx.emit(KnobModelEvent::SetDisplayTextbox(true));
            }
            WindowEvent::MouseDoubleClick(MouseButton::Left) => {
                self.widget_base.begin_set_parameter(cx);
                self.widget_base
                    .set_normalized_value(cx, self.widget_base.default_normalized_value());
                self.widget_base.end_set_parameter(cx);
            }
            WindowEvent::KeyUp(Code::Escape, _) => {
                cx.emit(KnobModelEvent::SetDisplayTextbox(false));
                cx.release();
            }
            _ => {}
        });
        event.map(|event, _| {
            self.widget_base.begin_set_parameter(cx);
            match event {
                //KnobEvents::EnterValuePlain(value) => {
                //    if let Some(normalized) = self.widget_base.string_to_normalized_value(value) {
                //        cx.emit(KnobEvents::SetValueNormalized(normalized));
                //        cx.emit(KnobModelEvent::SetDisplayTextbox(false));
                //    }
                //}
                &KnobEvents::SetValueNormalized(value) => {
                    self.widget_base.begin_set_parameter(cx);
                    self.widget_base.set_normalized_value(cx, value);
                    self.widget_base.end_set_parameter(cx);
                }
            }
            self.widget_base.end_set_parameter(cx);
        });
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        const MAX_ANGLE_DEG: f32 = 235f32;
        const EXTEND_X: f32 = 7. / 89.;
        const EXTEND_Y: f32 = 6. / 89.;
        let mut bounds = cx.bounds();
        let width = bounds.width();
        let height = bounds.height();
        let add_width = EXTEND_X * width;
        let add_height = EXTEND_Y * height;
        bounds.x -= add_width;
        bounds.w += 2. * add_width;
        bounds.y -= add_height;
        bounds.h += 2. * add_height;

        let param = self.widget_base.modulated_normalized_value();
        let knob_angle = 2. * (0.5 - param) * MAX_ANGLE_DEG.to_radians();

        let fg = self.image_fg.or_else(|| {
            canvas
                .load_image_file(
                    "data/images/Knob Front@2x.png",
                    ImageFlags::GENERATE_MIPMAPS,
                )
                .inspect_err(|err| nih_error!("Cannot load image: {err}"))
                .ok()
        });
        let bg = self.image_bg.or_else(|| {
            canvas
                .load_image_file("data/images/Knob BG@2x.png", ImageFlags::GENERATE_MIPMAPS)
                .inspect_err(|err| nih_error!("Cannot load bg image: {err}"))
                .ok()
        });

        let bg_paint = if let Some(id) = bg {
            let w = bounds.width();
            let h = bounds.height();
            let (cx, cy) = bounds.center();
            vg::Paint::image(id, cx, cy, w, h, 0., 1.)
        } else {
            vg::Paint::color(Color::transparent().into())
        };
        let fg_paint = if let Some(id) = fg {
            let w = bounds.width();
            let h = bounds.height();
            let (cx, cy) = bounds.center();
            vg::Paint::image(id, cx, cy, w, h, 0., 1.)
        } else {
            vg::Paint::color(Color::transparent().into())
        };

        let mut bounds_path = vg::Path::new();
        bounds_path.rect(-bounds.w / 2., -bounds.h / 2., bounds.w, bounds.h);

        canvas.save();
        let (cx, cy) = bounds.center();
        canvas.translate(cx, cy);
        canvas.rotate(knob_angle);
        canvas.fill_path(&bounds_path, &bg_paint);
        canvas.fill_path(&bounds_path, &fg_paint);
        canvas.restore();
    }
}
