//! Test utilities. Needs the `test-utils` feature to enable this module.
use std::{ops::Range, path::Path};

use plotters::coord::{self, ranged1d::ValueFormatter};
use plotters::{chart::SeriesAnno, prelude::*};

fn assert_ok(res: Result<(), impl std::fmt::Display>) {
    match res {
        Ok(()) => {}
        Err(value) => panic!("Not OK: {value}"),
    }
}

/// Single time/frequency series
pub struct Series<'a> {
    /// Label of the series
    pub label: &'a str,
    /// Sample rate of the time series
    pub samplerate: f32,
    /// Y-values of the series
    pub series: &'a [f32],
    /// Display color,
    pub color: &'a RGBColor,
}

impl<'a> Series<'a> {
    /// Validate that the series is well-formed
    pub fn validate(&self) -> Result<(), String> {
        if self.samplerate <= 0. {
            return Err(format!("Series: {:?}: Samplerate is negative", self.label));
        }
        if self.series.is_empty() {
            return Err(format!("Series: {:?}: No data", self.label));
        }

        Ok(())
    }

    /// Compute the range of x values spanning this series.
    ///
    /// # Arguments
    ///
    /// * `bode`: The series represent frequency data as opposed to time data.
    ///
    /// returns: Range<f32>
    pub fn timescale(&self, bode: bool) -> Range<f32> {
        assert_ok(self.validate());
        if bode {
            0.0..self.samplerate / 2.0
        } else {
            let tmax = self.series.len() as f32 / self.samplerate;
            0.0..tmax
        }
    }

    /// Return the y-axis range that this series span
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

/// Simple, high-level abstraction to plotting time/frequency series
pub struct Plot<'a> {
    /// Title of the graph
    pub title: &'a str,
    /// Is the graph a Bode plot (plotting frequencies) or a time plot ?
    pub bode: bool,
    /// List of series to plot
    pub series: &'a [Series<'a>],
}

impl<'a> Plot<'a> {
    /// Validate that the plot is well-formed.
    pub fn validate(&self) -> Result<(), String> {
        if self.series.is_empty() {
            return Err(format!("Plot {:?}: no series", self.title));
        }
        self.series.iter().try_for_each(|s| s.validate())?;
        Ok(())
    }

    /// Render the plot into the provided drawing area. This is a lower-level method that allows you
    /// to direct rendering wherever you want.
    ///
    /// # Arguments
    ///
    /// * `output`: Output drawing area, on which the plot will be drawn.
    ///
    /// returns: ()
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
            .background_style(WHITE.mix(0.8))
            .draw()
            .unwrap();
    }

    /// Create an SVG file in which this plot is going to be rendered.
    ///
    /// # Arguments
    ///
    /// * `filename`: Filename pointing to the generated SVG file
    ///
    /// returns: ()
    pub fn create_svg(&self, filename: impl AsRef<Path>) {
        let path = filename.as_ref();
        let _ = std::fs::create_dir_all(path.parent().expect("Filename is empty"));
        let root = SVGBackend::new(path, (600, 400)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        self.render_into(&root);
    }
}
