use crate::params::PolysynthParams;
use nih_plug::prelude::{Editor, Param};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::Arc;

#[derive(Lens)]
struct Data {
    params: Arc<PolysynthParams>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (1000, 600))
}

pub(crate) fn create(
    params: Arc<PolysynthParams>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);

        Data {
            params: params.clone(),
        }
        .build(cx);

        VStack::new(cx, |cx| {
            HStack::new(cx, move |cx| {
                Label::new(cx, "Polysynth")
                    .font_weight(FontWeightKeyword::Thin)
                    .font_size(30.0)
                    .height(Pixels(50.0))
                    .child_left(Stretch(1.0))
                    .child_right(Stretch(1.0));
                HStack::new(cx, |cx| {
                    Label::new(cx, "Output Level")
                        .child_top(Pixels(5.))
                        .width(Auto)
                        .height(Pixels(30.0));
                    ParamSlider::new(cx, Data::params, |p| &p.output_level).width(Pixels(200.));
                })
                .col_between(Pixels(8.0));
            })
            .col_between(Stretch(1.0))
            .width(Percentage(100.))
            .height(Pixels(30.));
            VStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    for ix in 0..crate::dsp::NUM_OSCILLATORS {
                        let p = Data::params.map(move |p| p.osc_params[ix].clone());
                        VStack::new(cx, |cx| {
                            Label::new(cx, &format!("Oscillator {}", ix + 1))
                                .font_size(22.)
                                .child_bottom(Pixels(8.));
                            GenericUi::new(cx, p);
                        });
                    }
                    VStack::new(cx, |cx| {
                        Label::new(cx, "Filter")
                            .font_size(22.)
                            .child_bottom(Pixels(8.));
                        GenericUi::new(cx, Data::params.map(|p| p.filter_params.clone()));
                    });
                })
                .row_between(Stretch(1.0));
                HStack::new(cx, |cx| {
                    VStack::new(cx, |cx| {
                        Label::new(cx, "Mixer").font_size(22.);
                        GenericUi::new(cx, Data::params.map(|p| p.mixer_params.clone()));
                    });
                    VStack::new(cx, |cx| {
                        Label::new(cx, "Amp Env").font_size(22.);
                        GenericUi::new(cx, Data::params.map(|p| p.vca_env.clone()));
                    });
                    VStack::new(cx, |cx| {
                        Label::new(cx, "Filter Env").font_size(22.);
                        GenericUi::new(cx, Data::params.map(|p| p.vcf_env.clone()));
                    });
                })
                .left(Stretch(1.0))
                .right(Stretch(1.0))
                .width(Pixels(750.));
            })
            .top(Pixels(16.))
            .width(Percentage(100.))
            .height(Percentage(100.))
            .row_between(Pixels(0.0));
        })
        .row_between(Pixels(0.0))
        .width(Percentage(100.))
        .height(Percentage(100.));
        ResizeHandle::new(cx);
    })
}
