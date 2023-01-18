use std::f32::consts::TAU;
use std::sync::{atomic::Ordering, Arc};

use atomic_float::AtomicF32;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::{context::DrawContext, prelude::*, vg};
use realfft::num_complex::Complex;

use valib::{saturators::Linear, svf::Svf};

use crate::filter::FilterParams;
use crate::AbrasiveParams;

#[derive(Debug, Clone)]
struct EqData<const N: usize> {
    samplerate: Arc<AtomicF32>,
    params: Arc<AbrasiveParams<N>>,
    frequency_range: FloatRange,
    // gain_range: FloatRange,
    modulated: bool,
}

impl<const N: usize> View for EqData<N> {
    fn element(&self) -> Option<&'static str> {
        Some("eq")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let samplerate = self.samplerate.load(Ordering::Relaxed);

        let filters = if self.modulated {
            self.params.params.each_ref().map(|p| {
                Svf::<f32, Linear>::new(
                    samplerate,
                    p.cutoff.smoothed.previous_value(),
                    (1. - p.q.smoothed.previous_value()).max(1e-3),
                )
            })
        } else {
            self.params.params.each_ref().map(|p| {
                Svf::<f32, Linear>::new(
                    samplerate,
                    p.cutoff.unmodulated_plain_value(),
                    (1. - p.q.unmodulated_plain_value()).max(1e-3),
                )
            })
        };
        let bounds = cx.bounds();
        let paint = vg::Paint::color(cx.font_color().copied().unwrap_or(Color::white()).into())
            .with_line_width(
                // FIXME: Using border width as stroke-width is not currently accessible
                cx.border_width()
                    .map(|u| u.value_or(bounds.h, 1.))
                    .unwrap_or(1.),
            );
        let mut path = vg::Path::new();

        for j in 0..4 * bounds.w as usize {
            let i = j as f32 / 4.;
            let x = i / bounds.w;
            let freq = self.frequency_range.unnormalize(x);
            let jw = TAU * freq;
            let y = (0..N)
                .map(|i| {
                    let ftype = self.params.params[i].ftype.value();
                    ftype.freq_response(
                        &filters[i],
                        if self.modulated {
                            util::db_to_gain(
                                self.params.scale.smoothed.previous_value()
                                    * util::gain_to_db(
                                        self.params.params[i].amp.smoothed.previous_value(),
                                    ),
                            )
                        } else {
                            util::db_to_gain(
                                self.params.scale.unmodulated_plain_value()
                                    * util::gain_to_db(
                                        self.params.params[i].amp.unmodulated_plain_value(),
                                    ),
                            )
                        },
                        jw,
                    )
                })
                .product::<Complex<f32>>()
                .norm();
            let y = (util::gain_to_db(y) + 24.) / 48.;
            if j == 0 {
                path.move_to(bounds.x + bounds.w * x, bounds.y + bounds.h * (1. - y));
            } else {
                path.line_to(bounds.x + bounds.w * x, bounds.y + bounds.h * (1. - y));
            }
        }

        canvas.stroke_path(&mut path, &paint);
    }
}

impl<const N: usize> EqData<N> {
    pub fn new(
        samplerate: Arc<AtomicF32>,
        params: Arc<AbrasiveParams<N>>,
        modulated: bool,
    ) -> Self {
        Self {
            samplerate,
            params,
            frequency_range: FilterParams::cutoff_range(),
            // gain_range: FloatRange::Skewed {
            //     min: util::db_to_gain(-24.),
            //     max: util::db_to_gain(24.),
            //     factor: FloatRange::gain_skew_factor(-24., 24.),
            // },
            modulated,
        }
    }
}

pub(crate) fn build<const N: usize>(
    cx: &mut Context,
    samplerate: impl Res<Arc<AtomicF32>>,
    params: impl Res<Arc<AbrasiveParams<N>>>,
) -> Handle<impl View> {
    let samplerate = samplerate.get_val(cx);
    let params = params.get_val(cx);

    ZStack::new(cx, |cx| {
        EqData::new(samplerate.clone(), params.clone(), true)
            .build(cx, |_| ())
            .class("modulated");
        EqData::new(samplerate, params, false)
            .build(cx, |_| ())
            .class("unmodulated");
    })
}
