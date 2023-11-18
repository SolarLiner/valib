use std::fmt::format;
use std::sync::Arc;
use nih_plug::log;
use nih_plug::log::{Level, Log, Record};
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;
use resource::{Resource, resource, resource_str};
use abrasive::editor::components::Knob;
use once_cell::sync::Lazy;

#[derive(Debug)]
struct KnobParams {
    value: FloatParam,
}

impl Default for KnobParams {
    fn default() -> Self {
        Self {
            value: FloatParam::new("value", 0.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2))
        }
    }
}

#[derive(Clone, Lens)]
struct ExampleData {
    bipolar: bool,
    centered: bool,
    knob_params: Arc<KnobParams>,
}

impl Model for ExampleData {}

impl Default for ExampleData {
    fn default() -> Self {
        let knob_params = Arc::default();
        Self {
            bipolar: false,
            centered: true,
            knob_params,
        }
    }
}

// static STYLESHEET: Lazy<Resource<str>> = Lazy::new(|| resource_str!("src/editor/theme.css"));
struct MyLogger;

impl Log for MyLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

fn main() {
    log::set_max_level(log::LevelFilter::Trace);
    log::set_logger(&MyLogger).expect("Cannot set logger");
    nih_log!("Test log");
    Application::new(|cx| {
        cx.add_font_mem(resource!("src/assets/Metrophobic-Regular.ttf"));
        cx.add_stylesheet(include_style!("src/editor/theme.css")).expect("Cannot load stylesheet");
        ExampleData::default().build(cx);

        VStack::new(cx, |cx| {
            Binding::new(cx, ExampleData::bipolar, |cx, lens| {
                let bipolar = lens.get(cx);
                Knob::new(cx, bipolar, ExampleData::knob_params, |params| &params.value);
            });
            VStack::new(cx, |cx| {
                Label::new(cx, ExampleData::knob_params.map(|params| format!("Normalized value: {:2.2}", params.value.unmodulated_normalized_value())));
                Label::new(cx, ExampleData::knob_params.map(|params| format!("Plain      value: {:2.2}", params.value.unmodulated_plain_value())));
            }).font_family([FamilyOwned::Monospace]);
        }).id("ui");
    }).run();
}