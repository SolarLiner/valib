// use nih_plug::prelude::*;
use nih_plug_vizia::{
    vizia::{modifiers::StyleModifiers, prelude::*},
    widgets::GenericUi,
};
use crate::editor::components::knob::HandleKnobExt;
use super::components::knob::Knob;
use crate::editor::Data;

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
        Data::params.map(move |p| p.params[selected].clone()),
        |cx, fparams| {
            // GenericUi::new(cx, fparams);
            GenericUi::new_custom(cx, fparams, |cx, param_ptr| {
                Knob::new(cx).value(unsafe { param_ptr.modulated_normalized_value() });
            });
        },
    );
}
