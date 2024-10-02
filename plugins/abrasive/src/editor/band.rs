use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_vizia::{
    vizia::{modifiers::StyleModifiers, prelude::*},
    widgets::param_base::ParamWidgetBase,
};

use crate::editor::components::knob;
use crate::{dsp::filter::FilterParams, editor::Data, AbrasiveParams};
#[derive(Lens)]
struct BandSelector {
    ctrl: ParamWidgetBase,
}

enum BandSelectorEvent {
    Previous,
    Next,
}

impl BandSelector {
    pub fn new<E: Enum + PartialEq<E>>(
        cx: &mut Context,
        selected: usize,
        get_param: impl 'static + Copy + Fn(&FilterParams) -> &EnumParam<E>,
    ) -> Handle<Self> {
        Self {
            ctrl: ParamWidgetBase::new(cx, Data::params, move |p: &Arc<AbrasiveParams>| {
                get_param(&p.dsp_params.filters[selected])
            }),
        }
        .build(cx, |cx| {
            let lens = BandSelector::ctrl.map(|ctrl| {
                ctrl.normalized_value_to_string(ctrl.modulated_normalized_value(), false)
            });
            Spinbox::new(cx, lens, SpinboxKind::Horizontal, SpinboxIcons::Chevrons)
                .on_increment(|cx| cx.emit(BandSelectorEvent::Next))
                .on_decrement(|cx| cx.emit(BandSelectorEvent::Previous));
            /*            Binding::new(
                            cx,
                            BandSelector::ctrl.map(|ctrl| ctrl.modulated_plain_value() as usize),
                            move |cx, sel_lens| {
                                let selected = sel_lens.get(cx);
                                let selected_lens = lens.index(selected);
                                Dropdown::new(
                                    cx,
                                    move |cx| Label::new(cx, selected_lens),
                                    move |cx| {
                                        let dropdown_entity = cx.current();
                                        List::new(cx, lens, move |cx, i, item_lens| {
                                            Label::new(cx, item_lens).on_press(move |cx| {
                                                cx.emit(BandSelectorEvent::SetActive(i));
                                                cx.emit_to(dropdown_entity, Dropdown)
                                            });
                                        });
                                    },
                                );
                            },
                        );
            */
        })
    }
}

impl View for BandSelector {
    fn element(&self) -> Option<&'static str> {
        Some("band-selector")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|ev, _| match ev {
            BandSelectorEvent::Previous => {
                self.ctrl.begin_set_parameter(cx);
                let current = self.ctrl.unmodulated_plain_value();
                let normalized = self.ctrl.preview_normalized(current - 1.0);
                self.ctrl.set_normalized_value(cx, normalized);
                self.ctrl.end_set_parameter(cx);
            }
            BandSelectorEvent::Next => {
                self.ctrl.begin_set_parameter(cx);
                let current = self.ctrl.unmodulated_plain_value();
                let normalized = self.ctrl.preview_normalized(current + 1.0);
                self.ctrl.set_normalized_value(cx, normalized);
                self.ctrl.end_set_parameter(cx);
            }
        })
    }
}

pub fn band_knobs(cx: &mut Context, selected: usize) {
    VStack::new(cx, move |cx| {
        BandSelector::new(cx, selected, |params| &params.ftype);
        Binding::new(
            cx,
            Data::params.map(move |p: &Arc<AbrasiveParams>| p.dsp_params.filters[selected].clone()),
            move |cx, fparams| {
                knob::labelled_node_float(cx, false, fparams, |params| &params.cutoff);
                HStack::new(cx, move |cx| {
                    knob::labelled_node_float(cx, false, fparams, |params| &params.q)
                        .class("small");
                    knob::labelled_node_float(cx, true, fparams, |params| &params.amp)
                        .class("small");
                })
                .child_space(Stretch(1.0))
                .col_between(Pixels(16.0));
            },
        );
    });
}
