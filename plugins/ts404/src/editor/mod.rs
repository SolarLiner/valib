use std::sync::Arc;

use components::led::Led;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::views::{Knob, VStack};
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;
use nih_plug_vizia::{create_vizia_editor, ViziaState, ViziaTheming};

use crate::params::Ts404Params;

mod components;

pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (500, 150))
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
            cx.emit(EnvironmentEvent::SetThemeMode(AppTheme::BuiltIn(
                ThemeMode::DarkMode,
            )));
            AppData {
                params: params.clone(),
                drive_led: drive_led.clone(),
            }
            .build(cx);
            Binding::new(cx, AppData::params, |cx, params| {
                VStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        labelled_node_bool(cx, AppData::params, |p| &p.bypass);
                        ZStack::new(cx, |cx| {
                            labelled_node_float(cx, false, params, |p| &p.dist);
                            Binding::new(cx, AppData::drive_led, |cx, drive_led| {
                                Led::new(cx, drive_led.get(cx));
                            });
                        });
                        labelled_node_float(cx, false, params, |params| &params.tone);
                        labelled_node_float(cx, false, params, |params| &params.input_mode);
                        labelled_node_float(cx, false, params, |params| &params.component_matching);
                    })
                    .class("small");
                })
                .id("ui");
            });
        },
    )
}

#[derive(Lens)]
struct AppData {
    params: Arc<Ts404Params>,
    drive_led: Arc<AtomicF32>,
}

impl Model for AppData {}

fn labelled_node_bool(
    cx: &mut Context,
    params: impl Lens<Target = Arc<Ts404Params>>,
    get_param: impl 'static + Copy + Fn(&Arc<Ts404Params>) -> &BoolParam,
) -> Handle<'_, impl View> {
    labelled_node_generic(cx, params, get_param, move |cx| {
        let active = params.map_ref(get_param).map(|p| p.value());
        let param_base = ParamWidgetBase::new(cx, params, get_param);
        Switch::new(cx, active).on_toggle(move |cx| {
            let next = if active.get(cx) { 0.0 } else { 1.0 };
            param_base.begin_set_parameter(cx);
            param_base.set_normalized_value(cx, next);
            param_base.end_set_parameter(cx);
        });
    })
}

fn labelled_node_float<P: 'static + Param>(
    cx: &mut Context,
    bipolar: bool,
    params: impl Lens<Target = Arc<Ts404Params>>,
    get_param: impl 'static + Copy + Fn(&Arc<Ts404Params>) -> &P,
) -> Handle<'_, impl View>
where
    P::Plain: Data + ToString,
{
    labelled_node_generic(cx, params, get_param, move |cx| {
        let default_value = params
            .map_ref(get_param)
            .map(|param| param.default_normalized_value());
        let normalized_value = params
            .map_ref(get_param)
            .map(|param| param.modulated_normalized_value());
        let param_base = ParamWidgetBase::new(cx, params, get_param);
        Knob::new(cx, default_value, normalized_value, bipolar)
            .on_changing(move |cx, value| {
                param_base.begin_set_parameter(cx);
                param_base.set_normalized_value(cx, value);
                param_base.end_set_parameter(cx);
            })
            .width(Pixels(60.))
            .height(Pixels(60.));
    })
}

fn labelled_node_enum<E: 'static + PartialEq + ToString + Data + Enum>(
    cx: &mut Context,
    params: impl Lens<Target = Arc<Ts404Params>>,
    get_param: impl 'static + Copy + Fn(&Arc<Ts404Params>) -> &EnumParam<E>,
) -> Handle<'_, impl View> {
    #[derive(Lens)]
    struct EnumData {
        names: Vec<String>,
    }
    impl Model for EnumData {}

    labelled_node_generic(cx, params, get_param, move |cx| {
        let param = params.map_ref(get_param);
        let name = param.map(|param| param.value().to_string());
        let num_values = E::variants().len();
        EnumData {
            names: E::variants().into_iter().map(|s| s.to_string()).collect(),
        }
        .build(cx);

        Dropdown::new(
            cx,
            move |cx| Label::new(cx, name),
            move |cx| {
                List::new(cx, EnumData::names, move |cx, ix, item| {
                    let param_base = ParamWidgetBase::new(cx, params, get_param);
                    Label::new(cx, item)
                        .cursor(CursorIcon::Hand)
                        .bind(name, move |handle, selected| {
                            if item.get(&handle) == selected.get(&handle) {
                                handle.checked(true);
                            }
                        })
                        .on_press(move |cx| {
                            let value = ix as f32 / num_values as f32;
                            param_base.begin_set_parameter(cx);
                            param_base.set_normalized_value(cx, value);
                            param_base.end_set_parameter(cx);
                            cx.emit(PopupEvent::Close);
                        });
                });
            },
        )
        .width(Pixels(150.))
        .min_space(Pixels(15.))
        .top(Stretch(2.))
        .bottom(Stretch(0.667));
    })
}

fn labelled_node_generic<P: 'static + Param>(
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
