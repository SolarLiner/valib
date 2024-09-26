use crate::editor::background::Background;
use crate::spectrum::Spectrum;
use analyzer::SpectrumAnalyzer;
use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::widgets::ResizeHandle;
use nih_plug_vizia::{create_vizia_editor, vizia::prelude::*, ViziaState, ViziaTheming};
use resource::resource;
use std::sync::{Arc, Mutex};
use triple_buffer::Output;

mod analyzer;

mod background;

mod band;
pub mod components;

mod eq;

pub type SpectrumUI = Arc<Mutex<Output<Spectrum>>>;

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<crate::AbrasiveParams>,
    pub(crate) samplerate: Arc<AtomicF32>,
    pub(crate) spectrum_in: SpectrumUI,
    pub(crate) spectrum_out: SpectrumUI,
    pub(crate) selected: Option<usize>,
}

impl Model for Data {}

pub(crate) fn default_state() -> Arc<ViziaState> {
    // ViziaState::from_size(683, 500)
    ViziaState::new(|| (683, 500))
}

pub(crate) fn create(data: Data, state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(state, ViziaTheming::Custom, move |cx, _| {
        cx.emit(EnvironmentEvent::SetThemeMode(AppTheme::BuiltIn(
            ThemeMode::DarkMode,
        )));
        cx.add_font_mem(resource!("src/assets/Metrophobic-Regular.ttf"));
        if let Err(err) = cx.add_stylesheet(include_style!("src/editor/theme.css")) {
            nih_error!("Cannot read CSS: {err}");
        }
        data.clone().build(cx);

        ResizeHandle::new(cx);
        VStack::new(cx, |cx| {
            analyzer(cx, data.samplerate.clone()).class("analyzer");
            HStack::new(cx, |cx| {
                for i in 0..super::dsp::NUM_BANDS {
                    band::band_knobs(cx, i);
                }
            })
            .class("panel");
        })
        .width(Percentage(100.))
        .height(Percentage(100.))
        .id("ui");
    })
}

fn analyzer(cx: &mut Context, samplerate: Arc<AtomicF32>) -> Handle<impl View> {
    nih_log!("Creating analyzer");
    ZStack::new(cx, move |cx| {
        Background::new(cx, samplerate).class("bg");
        SpectrumAnalyzer::new(cx, Data::spectrum_in.get(cx), Data::samplerate.get(cx))
            .class("input");
        SpectrumAnalyzer::new(cx, Data::spectrum_out.get(cx), Data::samplerate.get(cx))
            .class("output");
        eq::build(cx, Data::samplerate, Data::params).id("eq");
    })
}
