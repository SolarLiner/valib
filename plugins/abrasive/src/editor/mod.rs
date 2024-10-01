use crate::editor::background::Background;
use crate::editor::preset_manager::{PresetManager, PresetManagerEvent};
use crate::spectrum::Spectrum;
use crate::AbrasiveParams;
use analyzer::SpectrumAnalyzer;
use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::widgets::ResizeHandle;
use nih_plug_vizia::{create_vizia_editor, vizia::prelude::*, ViziaState, ViziaTheming};
use resource::resource;
use std::fs;
use std::sync::{Arc, Mutex};
use triple_buffer::Output;
use valib::preset_manager::data::PresetMeta;

mod analyzer;

mod background;

mod band;
pub mod components;

mod eq;
mod util;
mod preset_manager;

pub type SpectrumUI = Arc<Mutex<Output<Spectrum>>>;

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<AbrasiveParams>,
    pub(crate) samplerate: Arc<AtomicF32>,
    pub(crate) spectrum_in: SpectrumUI,
    pub(crate) spectrum_out: SpectrumUI,
    pub(crate) selected: Option<usize>,
    pub(crate) show_save_dialog: bool,
}

impl Model for Data {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|event: &AppEvent, _| match event.clone() {
            AppEvent::SaveAsDialog => {
                self.show_save_dialog = true;
            }
            AppEvent::ConfirmSave { bank, preset_meta } => {
                cx.emit(PresetManagerEvent::SaveCurrentPreset { bank, preset_meta });
                self.show_save_dialog = false;
            }
            AppEvent::CancelSave => {
                self.show_save_dialog = false;
            }
        })
    }
}

pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (683, 550))
}

pub(crate) fn create(data: Data, state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(state, ViziaTheming::Custom, move |cx, _| {
        cx.emit(EnvironmentEvent::SetThemeMode(AppTheme::BuiltIn(
            ThemeMode::DarkMode,
        )));
        match fs::read(util::resolve_asset_file(
            "fonts/Metrophobic-Regular.ttf".as_ref(),
        )) {
            Ok(data) => {
                cx.add_font_mem(data);
            }
            Err(err) => {
                nih_error!("Cannot load font: {err}");
            }
        }
        if let Err(err) = cx.add_stylesheet(util::resolve_asset_file("styles/theme.css".as_ref())) {
            nih_error!("Cannot read CSS: {err}");
        }
        data.clone().build(cx);

        ResizeHandle::new(cx);
        VStack::new(cx, |cx| {
            topbar(cx, Data::params);
            analyzer(cx, data.samplerate.clone()).class("analyzer");
            HStack::new(cx, |cx| {
                VStack::new(cx, |cx| {
                    components::knob::labelled_node_float(cx, false, Data::params, |p| {
                        &p.dsp_params.drive
                    });
                })
                .child_space(Stretch(1.0));
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

#[derive(Debug, Clone)]
enum AppEvent {
    SaveAsDialog,
    ConfirmSave {
        bank: String,
        preset_meta: PresetMeta,
    },
    CancelSave,
}

fn topbar(cx: &mut Context, params: impl Lens<Target = Arc<AbrasiveParams>>) -> Handle<impl View> {
    HStack::new(cx, move |cx| {
        Binding::new(cx, params, move |cx, params| {
            let params = params.get(cx);
            PresetManager::new(cx, params).max_width(Pixels(200.0));
            Button::new(
                cx,
                |cx| cx.emit(AppEvent::SaveAsDialog),
                |cx| Label::new(cx, "Save"),
            );
        });
    })
    .class("topbar")
    .width(Percentage(100.))
    .height(Pixels(50.0))
    .child_space(Stretch(1.0))
}

fn analyzer(cx: &mut Context, samplerate: Arc<AtomicF32>) -> Handle<impl View> {
    nih_log!("Creating analyzer");
    ZStack::new(cx, move |cx| {
        Background::new(cx, samplerate).class("bg");
        SpectrumAnalyzer::new(cx, Data::spectrum_in.get(cx)).class("input");
        SpectrumAnalyzer::new(cx, Data::spectrum_out.get(cx)).class("output");
        eq::build(cx, Data::samplerate, Data::params).id("eq");
    })
}
