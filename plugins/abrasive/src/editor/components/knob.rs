use std::f32::consts::{FRAC_PI_2, TAU};

use nih_plug::{nih_log, prelude::Param};
use nih_plug_vizia::vizia::vg::Solidity;
use nih_plug_vizia::{
    vizia::{prelude::*, vg},
    widgets::param_base::ParamWidgetBase,
};

enum KnobEvents {
    SetValueNormalized(f32),
    EnterValuePlain(String),
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
                Solidity::Solid
            } else {
                Solidity::Hole
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
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
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

pub struct Knob {
    widget_base: ParamWidgetBase,
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
            drag_start: None,
        }
        .build(cx, |cx| {
            KnobModel {
                display_textbox: false,
                text: String::new(),
            }
            .build(cx);
            ZStack::new(cx, |cx| {
                Arc::new(cx, bipolar, params, get_param, |w| {
                    w.unmodulated_normalized_value()
                })
                .class("unmodulated");
                Arc::new(cx, bipolar, params, get_param, |w| {
                    w.modulated_normalized_value()
                })
                .class("modulated");
                Ring::new(cx, params, get_param).z_index(2);
                Textbox::new(cx, KnobModel::text)
                    .class("textbox")
                    .display(KnobModel::display_textbox.map(|display| {
                        if *display {
                            Display::Flex
                        } else {
                            Display::None
                        }
                    }))
                    .on_submit(|cx, data, _| {
                        cx.emit(KnobModelEvent::SetDisplayTextbox(false));
                        cx.emit(KnobEvents::EnterValuePlain(data));
                    })
                    .on_blur(|cx| cx.emit(KnobModelEvent::SetDisplayTextbox(false)));
            })
            .z_index(1);
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
                    self.widget_base.set_normalized_value(cx, normalized_value);
                }
            }
            &WindowEvent::MouseScroll(_, y) => {
                let step = if cx.modifiers().contains(Modifiers::SHIFT) {
                    0.01
                } else {
                    0.1
                };
                let normalized_value = self.widget_base.unmodulated_normalized_value() - y * step;
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
            _ => {}
        });
        event.map(|event, _| {
            self.widget_base.begin_set_parameter(cx);
            match event {
                KnobEvents::EnterValuePlain(value) => {
                    if let Some(normalized) = self.widget_base.string_to_normalized_value(value) {
                        cx.emit(KnobEvents::SetValueNormalized(normalized));
                    }
                }
                &KnobEvents::SetValueNormalized(value) => {
                    self.widget_base.begin_set_parameter(cx);
                    self.widget_base.set_normalized_value(cx, value);
                    self.widget_base.end_set_parameter(cx);
                }
            }
            self.widget_base.end_set_parameter(cx);
        });
    }
}
