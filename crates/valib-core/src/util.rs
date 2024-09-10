use crate::{Scalar, SimdCast};
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, One, Zero};
use numeric_literals::replace_float_literals;
use simba::simd::{SimdPartialOrd, SimdValue};

pub fn simd_index_scalar<Simd: Zero + SimdValue, Index: SimdValue<Element = usize>>(
    values: &[Simd::Element],
    index: Index,
) -> Simd
where
    Simd::Element: Copy,
{
    let mut ret = Simd::zero();
    for i in 0..Simd::LANES {
        let ix = index.extract(i);
        ret.replace(i, values[ix]);
    }
    ret
}

pub fn simd_index_simd<Simd: Zero + SimdValue, Index: SimdValue>(
    values: &[Simd],
    index: Index,
) -> Simd
where
    <Index as SimdValue>::Element: AsPrimitive<usize>,
{
    let mut ret = Simd::zero();
    for i in 0..Index::LANES {
        let ix = index.extract(i).as_();
        ret.replace(i, values[ix].extract(i));
    }
    ret
}

pub fn simd_is_finite<
    Simd: SimdValue<Element: Float, SimdBool: Default + SimdValue<Element = bool>>,
>(
    value: Simd,
) -> Simd::SimdBool {
    let mut mask = Simd::SimdBool::default();
    for i in 0..Simd::LANES {
        mask.replace(i, value.extract(i).is_finite())
    }
    mask
}

#[replace_float_literals(T::from_f64(literal))]
#[deprecated = "Use math::interpolators"]
pub fn lerp_block<T: Scalar + SimdCast<usize>>(out: &mut [T], inp: &[T])
where
    <T as SimdCast<usize>>::Output: Copy + Num + FromPrimitive + SimdPartialOrd,
{
    let ix_max = <T as SimdCast<usize>>::Output::from_usize(inp.len() - 1).unwrap();
    let rate = T::from_f64(inp.len() as f64) / T::from_f64(out.len() as f64);

    for (i, y) in out.iter_mut().enumerate() {
        let j = T::from_f64(i as f64) * rate;
        let f = j.simd_fract();
        let j = j.simd_floor().cast();
        let jp1 = (j + <T as SimdCast<usize>>::Output::one()).simd_min(ix_max);
        let a = simd_index_simd(inp, j);
        let b = simd_index_simd(inp, jp1);
        *y = lerp(f, a, b);
    }
}

pub fn lerp<T: Scalar>(t: T, a: T, b: T) -> T {
    use crate::math::interpolation::{Interpolate, Linear};
    Linear.interpolate(t, [a, b])
}

#[replace_float_literals(T::from_f64(literal))]
pub fn midi_to_freq<T: Scalar>(midi_note: u8) -> T {
    440.0 * semitone_to_ratio(T::from_f64(midi_note as _) - 69.0)
}

#[replace_float_literals(T::from_f64(literal))]
pub fn semitone_to_ratio<T: Scalar>(semi: T) -> T {
    2.0.simd_powf(semi / 12.0)
}

#[cfg(test)]
pub mod tests {
    use std::{ops::Range, path::Path};

    use plotters::coord::{self, ranged1d::ValueFormatter};
    use plotters::{chart::SeriesAnno, prelude::*};

    use crate::math::interpolation::{Interpolate, Linear};

    fn assert_ok(res: Result<(), impl std::fmt::Display>) {
        match res {
            Ok(()) => {}
            Err(value) => panic!("Not OK: {value}"),
        }
    }

    pub struct Series<'a> {
        pub label: &'a str,
        pub samplerate: f32,
        pub series: &'a [f32],
        pub color: &'a RGBColor,
    }

    impl<'a> Series<'a> {
        pub fn validate(&self) -> Result<(), String> {
            if self.samplerate <= 0. {
                return Err(format!("Series: {:?}: Samplerate is negative", self.label));
            }
            if self.series.is_empty() {
                return Err(format!("Series: {:?}: No data", self.label));
            }

            Ok(())
        }

        pub fn timescale(&self, bode: bool) -> Range<f32> {
            assert_ok(self.validate());
            if bode {
                0.0..self.samplerate / 2.0
            } else {
                let tmax = self.series.len() as f32 / self.samplerate;
                0.0..tmax
            }
        }

        pub fn y_range(&self) -> Range<f32> {
            assert_ok(self.validate());
            let min = self.series.iter().copied().min_by(f32::total_cmp).unwrap();
            let max = self.series.iter().copied().max_by(f32::total_cmp).unwrap();
            min..max
        }

        fn as_series<DB: DrawingBackend>(&self, bode: bool) -> LineSeries<DB, (f32, f32)> {
            LineSeries::new(
                self.series.iter().copied().enumerate().map(|(i, y)| {
                    let x = if bode {
                        i as f32
                    } else {
                        i as f32 / self.samplerate
                    };
                    (x, y)
                }),
                self.color,
            )
        }

        fn apply_legend(&self, ann: &mut SeriesAnno<impl DrawingBackend>) {
            let color = *self.color;
            ann.label(self.label);
            ann.legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color));
        }
    }

    pub struct Plot<'a> {
        pub title: &'a str,
        pub bode: bool,
        pub series: &'a [Series<'a>],
    }

    impl<'a> Plot<'a> {
        pub fn validate(&self) -> Result<(), String> {
            if self.series.is_empty() {
                return Err(format!("Plot {:?}: no series", self.title));
            }
            self.series.iter().try_for_each(|s| s.validate())?;
            Ok(())
        }

        pub fn render_into(&self, output: &DrawingArea<impl DrawingBackend, coord::Shift>) {
            use plotters::prelude::*;
            assert_ok(self.validate());

            let timescale = self
                .series
                .iter()
                .map(|s| s.timescale(self.bode))
                .reduce(|l, r| {
                    let start = l.start.min(r.start);
                    let end = l.end.max(r.end);
                    start..end
                })
                .unwrap();
            let timescale = if self.bode {
                timescale.start * 2.0..timescale.end * 2.0
            } else {
                timescale
            };

            let yrange = self
                .series
                .iter()
                .map(|s| s.y_range())
                .reduce(|l, r| {
                    let start = l.start.min(r.start);
                    let end = l.end.max(r.end);
                    start..end
                })
                .unwrap();

            let mut ctx = ChartBuilder::on(output);
            ctx.set_label_area_size(LabelAreaPosition::Left, 40)
                .set_label_area_size(LabelAreaPosition::Bottom, 40)
                .caption(self.title, ("sans-serif", 40));
            if self.bode {
                let ctx = ctx
                    .build_cartesian_2d(timescale.log_scale(), yrange.log_scale())
                    .unwrap();
                self.render(ctx);
            } else {
                let ctx = ctx.build_cartesian_2d(timescale, yrange).unwrap();
                self.render(ctx);
            }
        }

        fn render<'ctx, T: 'ctx + Ranged<ValueType = f32> + ValueFormatter<f32>>(
            &self,
            mut ctx: ChartContext<'ctx, impl 'ctx + DrawingBackend, Cartesian2d<T, T>>,
        ) {
            ctx.configure_mesh().draw().unwrap();

            for series in self.series {
                let ann = ctx.draw_series(series.as_series(self.bode)).unwrap();
                series.apply_legend(ann);
            }

            ctx.configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .draw()
                .unwrap();
        }

        pub fn create_svg(&self, filename: impl AsRef<Path>) {
            let path = filename.as_ref();
            let _ = std::fs::create_dir_all(path.parent().expect("Filename is empty"));
            let root = SVGBackend::new(path, (600, 400)).into_drawing_area();
            root.fill(&WHITE).unwrap();
            self.render_into(&root);
        }
    }

    #[test]
    fn interp_block() {
        let a = [0., 1., 1.];
        let mut actual = [0.; 12];
        let expected = [0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1.];
        Linear.interpolate_slice(&mut actual, &a);
        assert_eq!(actual, expected);
    }
}
