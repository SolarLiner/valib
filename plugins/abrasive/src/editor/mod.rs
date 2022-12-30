use std::sync::{Arc, Mutex};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::{assets, create_vizia_editor, vizia::prelude::*, ViziaState, ViziaTheming};
use nih_plug_vizia::widgets::ResizeHandle;
use triple_buffer::Output;

use analyzer::SpectrumAnalyzer;

use crate::spectrum::Spectrum;

mod analyzer;
mod background;
mod eq;
mod band;

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

        ResizeHandle::new(cx);
        ZStack::new(cx, |cx| {
            HStack::new(cx, |cx| {
                analyzer(cx);
                band::side_panel(cx);
            })
            .id("ui");
            // FIXME: replace buttons by band handles
            HStack::new(cx, |cx| {
                Button::new(
                    cx,
                    |cx| cx.emit(DataEvent::Deselect),
                    |cx| Label::new(cx, "Deselect"),
                );
                for i in 0..data.params.params.len() {
                    Button::new(
                        cx,
                        move |cx| cx.emit(DataEvent::Select(i)),
                        |cx| Label::new(cx, &format!("Select {}", i + 1)),
                    );
                }
            })
            .col_between(Pixels(3.));
        });
    })
}

fn analyzer(cx: &mut Context) -> Handle<impl View> {
    nih_log!("Creating analyzer");
    ZStack::new(cx, |cx| {
        SpectrumAnalyzer::new(cx, Data::spectrum_in.get(cx), Data::samplerate.get(cx))
            .class("input");
        SpectrumAnalyzer::new(cx, Data::spectrum_out.get(cx), Data::samplerate.get(cx))
            .class("output");
        eq::build(cx, Data::samplerate, Data::params).id("eq");
    })
}
