use std::sync::Arc;

use nih_plug::{prelude::Param, params::EnumParam};
use nih_plug_vizia::vizia::{modifiers::StyleModifiers, prelude::*};

use crate::{editor::Data, filter::FilterType};
use crate::{filter::FilterParams, AbrasiveParams};

// use nih_plug::prelude::*;
use super::components::knob::Knob;

pub fn band_knobs(cx: &mut Context, selected: usize) {
    Binding::new(
        cx,
        Data::params
            .map(move |p: &Arc<AbrasiveParams<{ crate::NUM_BANDS }>>| p.params[selected].clone()),
        |cx, fparams| {
            VStack::new(cx, |cx| {
                Binding::new(cx, fparams, |cx, params| {
                    let params = params.get(cx);
                    let ftype : &EnumParam<FilterType> = &params.ftype;
                    let current = format!("{:?}", ftype.value());

                    Dropdown::new(cx, move |cx| Label::new(cx, &current), |cx| {
                        let count = ftype.step_count().unwrap();
                        for i in 0..count {
                            let t = i as f32 / count as f32;
                            let plain = ftype.preview_plain(t);
                            Label::new(cx, &format!("{:?}", plain));
                        }
                    });
                });
                labelled_node_float(cx, false, fparams, |params| &params.cutoff);
                HStack::new(cx, move |cx| {
                    labelled_node_float(cx, false, fparams, |params| &params.q)
                        .class("small");
                    labelled_node_float(cx, true, fparams, |params| &params.amp).class("small");
                })
                .child_space(Stretch(1.0))
                .col_between(Pixels(16.0));
            });
        },
    );
}

fn labelled_node_float<P: Param>(
    cx: &mut Context,
    bipolar: bool,
    params: impl Lens<Target = Arc<FilterParams>>,
    get_param: impl 'static + Copy + Fn(&Arc<FilterParams>) -> &P,
) -> Handle<'_, impl View> {
    VStack::new(cx, move |cx| {
        Knob::new(cx, bipolar, params, get_param);
        Label::new(
            cx,
            params.map(move |params| get_param(params).name().to_string()),
        )
        .text_align(TextAlign::Center)
        .width(Percentage(100.0));
    })
    .class("param")
    .child_space(Stretch(1.0))
    .col_between(Pixels(8.0))
    .text_align(TextAlign::Center)
}
