use std::sync::{Arc, Mutex};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::{
    assets, create_vizia_editor, vizia::prelude::*, widgets::GenericUi, ViziaState, ViziaTheming,
};
use triple_buffer::Output;

use analyzer::SpectrumAnalyzer;

use crate::spectrum::Spectrum;

mod analyzer;
mod background;
mod eq;

pub type SpectrumUI = Arc<Mutex<Output<Spectrum>>>;

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<crate::AbrasiveParams<2>>,
    pub(crate) samplerate: Arc<AtomicF32>,
    pub(crate) spectrum_in: SpectrumUI,
    pub(crate) spectrum_out: SpectrumUI,
    pub(crate) selected: Option<usize>,
}

#[derive(Debug, Copy, Clone)]
enum DataEvent {
    Select(usize),
    Deselect,
}

impl Model for Data {
    fn event(&mut self, _: &mut EventContext, event: &mut Event) {
        event.map(|event, _| match nih_dbg!(event) {
            DataEvent::Select(ix) => self.selected = Some(*ix),
            DataEvent::Deselect => self.selected = None,
        })
    }
}

pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::from_size(850, 300)
}

pub(crate) fn create(data: Data, state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        cx.add_theme(include_str!("./theme.css"));
        data.clone().build(cx);

        HStack::new(cx, |cx| {
            analyzer(cx);
            ScrollView::new(cx, 0., 0., false, true, |cx| {
                Binding::new(cx, Data::selected, move |cx, selected| {
                    if let Some(selected) = selected.get(cx) {
                        GenericUi::new(cx, Data::params.map(move |p| p.params[selected].clone()));
                    }
                });
            })
            .id("side-panel")
            .toggle_class("visible", Data::selected.map(|opt| opt.is_some()));
        })
        .id("ui");
    })
}

fn analyzer(cx: &mut Context) -> Handle<impl View> {
    nih_log!("Creating analyzer");
    ZStack::new(cx, |cx| {
/*        HStack::new(cx, |cx| {
            Button::new(
                cx,
                |cx| {
                    nih_log!("Clicked 'select'");
                    cx.emit(DataEvent::Select(0));
                },
                |cx| Label::new(cx, "Select"),
            );
            Button::new(
                cx,
                |cx| {
                    nih_log!("Clicked 'deselect'");
                    cx.emit(DataEvent::Deselect);
                },
                |cx| Label::new(cx, "Deselect"),
            );
            Label::new(cx, Data::selected.map(|s| format!("{:?}", s)))
                .color("black")
                .background_color("white");
        })
        .col_between(Pixels(10.))
        .width(Pixels(200.))
        .height(Pixels(50.));
*/        SpectrumAnalyzer::new(cx, Data::spectrum_in, Data::samplerate).class("input");
        SpectrumAnalyzer::new(cx, Data::spectrum_out, Data::samplerate).class("output");
        eq::build(cx, Data::samplerate, Data::params);
    })
}
