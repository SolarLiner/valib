use std::sync::{Arc, Mutex};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::widgets::ResizeHandle;
use nih_plug_vizia::{assets, create_vizia_editor, vizia::prelude::*, ViziaState, ViziaTheming};
use triple_buffer::Output;

use analyzer::SpectrumAnalyzer;

use crate::spectrum::Spectrum;

mod analyzer;
mod background;
mod components;
mod band;
mod eq;

pub type SpectrumUI = Arc<Mutex<Output<Spectrum>>>;

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<crate::AbrasiveParams<{super::NUM_BANDS}>>,
    pub(crate) samplerate: Arc<AtomicF32>,
    pub(crate) spectrum_in: SpectrumUI,
    pub(crate) spectrum_out: SpectrumUI,
    pub(crate) selected: Option<usize>,
}

impl Model for Data {}

pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::from_size(683, 500)
}

pub(crate) fn create(data: Data, state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        cx.add_fonts_mem(&[include_bytes!("../assets/Metrophobic-Regular.ttf")]);
        cx.add_theme(include_str!("./theme.css"));
        data.clone().build(cx);

        ResizeHandle::new(cx);
        VStack::new(cx, |cx| {
            analyzer(cx).class("analyzer");
            HStack::new(cx, |cx| {
                for i in 0..super::NUM_BANDS {
                    band::band_knobs(cx, i);
                }
            }).class("panel");
        }).width(Percentage(100.)).height(Percentage(100.)).id("ui");
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
