#[cfg(not(feature = "example"))]
use std::sync::{Arc, Mutex};

#[cfg(not(feature = "example"))]
use atomic_float::AtomicF32;
#[cfg(not(feature = "example"))]
use nih_plug::prelude::*;
#[cfg(not(feature = "example"))]
use nih_plug_vizia::widgets::ResizeHandle;
#[cfg(not(feature = "example"))]
use nih_plug_vizia::{assets, create_vizia_editor, vizia::prelude::*, ViziaState, ViziaTheming};
#[cfg(not(feature = "example"))]
use resource::{resource, resource_str, Resource};
#[cfg(not(feature = "example"))]
use triple_buffer::Output;

#[cfg(not(feature = "example"))]
use analyzer::SpectrumAnalyzer;

use crate::spectrum::Spectrum;

#[cfg(not(feature = "example"))]
mod analyzer;
#[cfg(not(feature = "example"))]
mod background;
#[cfg(not(feature = "example"))]
mod band;
pub mod components;
#[cfg(not(feature = "example"))]
mod eq;

#[cfg(not(feature = "example"))]
pub type SpectrumUI = Arc<Mutex<Output<Spectrum>>>;

#[cfg(not(feature = "example"))]
#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<crate::AbrasiveParams<{ super::NUM_BANDS }>>,
    pub(crate) samplerate: Arc<AtomicF32>,
    pub(crate) spectrum_in: SpectrumUI,
    pub(crate) spectrum_out: SpectrumUI,
    pub(crate) selected: Option<usize>,
}

#[cfg(not(feature = "example"))]
impl Model for Data {}

#[cfg(not(feature = "example"))]
pub(crate) fn default_state() -> Arc<ViziaState> {
    // ViziaState::from_size(683, 500)
    ViziaState::new(|| (683, 500))
}

#[cfg(not(feature = "example"))]
pub(crate) fn create(data: Data, state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        cx.add_font_mem(resource!("src/assets/Metrophobic-Regular.ttf"));
        if let Err(err) = cx.add_stylesheet(include_style!("src/editor/theme.css")) {
            nih_error!("Cannot read CSS: {err}");
        }
        data.clone().build(cx);

        ResizeHandle::new(cx);
        VStack::new(cx, |cx| {
            analyzer(cx).class("analyzer");
            HStack::new(cx, |cx| {
                for i in 0..super::NUM_BANDS {
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

#[cfg(not(feature = "example"))]
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
