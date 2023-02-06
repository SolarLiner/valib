use std::sync::Arc;

use nih_plug::prelude::{FloatParam, Param};
// use nih_plug::prelude::*;
use super::components::knob::Knob;
use crate::editor::Data;
use crate::{filter::FilterParams, AbrasiveParams};
use nih_plug_vizia::{
    vizia::{modifiers::StyleModifiers, prelude::*},
    widgets::{param_base::ParamWidgetBase, GenericUi},
};

pub fn side_panel(cx: &mut Context) {
    ScrollView::new(cx, 0., 0., false, true, |cx| {
        Binding::new(cx, Data::selected, move |cx, selected| {
            if let Some(selected) = selected.get(cx) {
                band_knobs(cx, selected);
            }
        });
    })
    .id("side-panel")
    .toggle_class("visible", Data::selected.map(|opt| opt.is_some()));
}

pub fn band_knobs(cx: &mut Context, selected: usize) {
    Binding::new(
        cx,
        Data::params.map(move |p: &Arc<AbrasiveParams<{crate::NUM_BANDS}>>| p.params[selected].clone()),
        |cx, fparams| {
            VStack::new(cx, |cx| {
                labelled_node_float(cx, false, fparams.clone(), |params| &params.cutoff);
                HStack::new(cx, move |cx| {
                    labelled_node_float(
                        cx,
                        false,
                        fparams.clone(),
                        |params| &params.q,
                    )
                    .class("small");
                    labelled_node_float(
                        cx,
                        true,
                        fparams,
                        |params| &params.amp,
                    )
                    .class("small");
                });
            });
        },
    );
}

fn labelled_node_float<'cx, P: Param>(
    cx: &'cx mut Context,
    bipolar: bool,
    params: impl Lens<Target = Arc<FilterParams>>,
    get_param: impl 'static + Copy + Fn(&Arc<FilterParams>) -> &P,
) -> Handle<'cx, impl View> {
    VStack::new(cx, move |cx| {
        Knob::new(cx, bipolar, params.clone(), get_param);
        Label::new(
            cx,
            params.map(move |params| get_param(params).name().to_string()),
        );
    })
    .class("param")
}
