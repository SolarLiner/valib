use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_vizia::{create_vizia_editor, ViziaState, ViziaTheming};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::views::VStack;

use components::{knob::Knob, led::Led};

use crate::params::Ts404Params;

mod components;

pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (200, 350))
}

pub(crate) fn create(
    params: Arc<Ts404Params>,
    drive_led: Arc<AtomicF32>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        params.editor_state.clone(),
        ViziaTheming::Builtin,
        move |cx, _gui_cx| {
            cx.add_stylesheet(include_style!("src/editor/style.css"))
                .expect("Failed to load stylesheet");
            AppData {
                params: params.clone(),
                drive_led: drive_led.clone(),
            }
            .build(cx);
            Binding::new(cx, AppData::params, |cx, params| {
                VStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        labelled_node_float(cx, false, params, |params| &params.dist);
                        labelled_node_float(cx, false, params, |params| &params.tone);
                        labelled_node_float(cx, false, params, |params| &params.out_level);
                    });
                    HStack::new(cx, |cx| {
                        labelled_node_float_generic(
                            cx,
                            params,
                            |params| &params.drive,
                            |cx| {
                                ZStack::new(cx, |cx| {
                                    Knob::new(cx, false, params, |params| &params.drive);
                                    Binding::new(cx, AppData::drive_led, |cx, drive_led| {
                                        Led::new(cx, drive_led.get(cx));
                                    });
                                })
                                .child_space(Pixels(0.));
                            },
                        );
                        labelled_node_float(cx, false, params, |params| &params.component_matching);
                    })
                    .class("small");
                })
                .id("ui");
            })
        },
    )
}

#[derive(Lens)]
struct AppData {
    params: Arc<Ts404Params>,
    drive_led: Arc<AtomicF32>,
}

impl Model for AppData {}

fn labelled_node_float<P: 'static + Param>(
    cx: &mut Context,
    bipolar: bool,
    params: impl Lens<Target = Arc<Ts404Params>>,
    get_param: impl 'static + Copy + Fn(&Arc<Ts404Params>) -> &P,
) -> Handle<'_, impl View>
where
    P::Plain: Data + ToString,
{
    labelled_node_float_generic(cx, params, get_param, move |cx| {
        Knob::new(cx, bipolar, params, get_param);
    })
}

fn labelled_node_float_generic<P: 'static + Param>(
    cx: &mut Context,
    params: impl Lens<Target = Arc<Ts404Params>>,
    get_param: impl 'static + Copy + Fn(&Arc<Ts404Params>) -> &P,
    knob_impl: impl Fn(&mut Context),
) -> Handle<'_, impl View>
where
    P::Plain: Data + ToString,
{
    VStack::new(cx, move |cx| {
        knob_impl(cx);
        Label::new(
            cx,
            params.map(move |params| get_param(params).name().to_string()),
        )
        .text_align(TextAlign::Center)
        .max_width(Pixels(60.));
    })
    .class("param")
    .child_space(Stretch(1.0))
    .col_between(Pixels(8.0))
    .text_align(TextAlign::Center)
    .tooltip(|cx| {
        Tooltip::new(cx, |cx| {
            Label::new(
                cx,
                params.map(move |p| get_param(p).unmodulated_plain_value()),
            );
        });
    })
}
