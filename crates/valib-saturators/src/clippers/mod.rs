//! # Diode clipper
//!
//! Saturators for emulating a diode clipper.

use super::adaa::Antiderivative;
use crate::MultiSaturator;
use crate::Saturator;
use nalgebra as na;
use nalgebra::{Dim, OMatrix, OVector, SVector, VectorView};
use num_traits::Float;
use numeric_literals::replace_float_literals;
use std::num::NonZeroUsize;
use valib_core::dsp::DSPMeta;
use valib_core::dsp::DSPProcess;
use valib_core::math::nr::{NewtonRhapson, RootEq};
use valib_core::simd::SimdBool;
use valib_core::util::vector_view_mut;
use valib_core::Scalar;

mod diode_clipper_model_data;

/// Diode clipper evaluated with the Newton-Rhapson method.
#[derive(Debug, Copy, Clone)]
pub struct DiodeClipper<T> {
    /// Reverse saturation current of the diode
    pub isat: T,
    /// Ideality coefficient of the diode
    pub n: T,
    /// Thermal voltage
    pub vt: T,
    /// Input voltage
    pub vin: T,
    /// Number of diodes in the forward direction
    pub num_diodes_fwd: T,
    /// Number of diodes in the backward direction
    pub num_diodes_bwd: T,
    /// Simulation tolerance
    pub sim_tol: T,
    /// Maximum number of iterations
    pub max_iter: usize,
    last_vout: T,
}

impl<T: Copy> DiodeClipper<T> {
    /// Return the last output of the clipper.
    pub fn last_output(&self) -> T {
        self.last_vout
    }
}

impl<T: Copy> DiodeClipper<T> {
    /// Reset the state of the clipper
    pub fn reset(&mut self) {
        self.last_vout = self.vin;
    }
}

impl<T: Scalar> RootEq for DiodeClipper<T> {
    type Scalar = T;
    type Dim = na::U1;

    #[replace_float_literals(T::from_f64(literal))]
    fn eval(
        &self,
        input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
    ) -> OVector<Self::Scalar, Self::Dim> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        if vout.simd_gt(16.0).any() {
            println!();
        }
        let expn = T::simd_exp(expin / self.num_diodes_fwd).simd_min(1e35);
        let expm = T::simd_exp(-expin / self.num_diodes_bwd).simd_min(1e35);
        let res = self.isat * (expn - expm) + 2. * vout - self.vin;
        [res].into()
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn j_inv(
        &self,
        input: VectorView<Self::Scalar, Self::Dim, impl Dim, impl Dim>,
    ) -> Option<OMatrix<Self::Scalar, Self::Dim, Self::Dim>> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        let expn = T::simd_exp(expin / self.num_diodes_fwd).simd_min(1e35);
        let expm = T::simd_exp(-expin / self.num_diodes_bwd).simd_min(1e35);
        let res = v * self.isat * (expn / self.num_diodes_fwd + expm / self.num_diodes_bwd) + 2.;
        // Biasing to prevent divisions by zero, less accurate around zero
        let ret = 1e-6.select(res.simd_abs().simd_lt(1e-6), res).simd_recip();
        Some([ret].into())
    }
}

impl<T: Scalar> DiodeClipper<T> {
    /// New diode clipper made of silicon.
    ///
    /// # Arguments
    ///
    /// * `fwd`: Number of diodes in the forward direction
    /// * `bwd`: Number of diodes in the backward direction
    /// * `vin`: Input voltage
    ///
    /// returns: DiodeClipper<T>
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new_silicon(fwd: usize, bwd: usize, vin: T) -> Self {
        Self {
            isat: 4.352e-9,
            n: 1.906,
            vt: 23e-3,
            num_diodes_fwd: T::from_f64(fwd as f64),
            num_diodes_bwd: T::from_f64(bwd as f64),
            vin,
            sim_tol: 1e-3,
            max_iter: 50,
            last_vout: vin.simd_tanh(),
        }
    }

    /// New diode clipper made of germanium.
    ///
    /// # Arguments
    ///
    /// * `fwd`: Number of diodes in the forward direction
    /// * `bwd`: Number of diodes in the backward direction
    /// * `vin`: Input voltage
    ///
    /// returns: DiodeClipper<T>
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new_germanium(fwd: usize, bwd: usize, vin: T) -> Self {
        Self {
            isat: 200e-9,
            n: 2.109,
            vt: 23e-3,
            num_diodes_fwd: T::from_f64(fwd as f64),
            num_diodes_bwd: T::from_f64(bwd as f64),
            vin,
            sim_tol: 1e-3,
            max_iter: 50,
            last_vout: vin.simd_tanh(),
        }
    }

    /// New diode clipper made of LEDs.
    ///
    /// # Arguments
    ///
    /// * `fwd`: Number of diodes in the forward direction
    /// * `bwd`: Number of diodes in the backward direction
    /// * `vin`: Input voltage
    ///
    /// returns: DiodeClipper<T>
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new_led(nf: usize, nb: usize, vin: T) -> DiodeClipper<T> {
        Self {
            isat: 2.96406e-12,
            n: 2.475312,
            vt: 23e-3,
            vin,
            num_diodes_fwd: T::from_f64(nf as f64),
            num_diodes_bwd: T::from_f64(nb as f64),
            sim_tol: 1e-4,
            max_iter: 50,
            last_vout: vin.simd_tanh(),
        }
    }
}

impl<T: Scalar> DSPMeta for DiodeClipper<T> {
    type Sample = T;
}

impl<T: Scalar> DSPProcess<1, 1> for DiodeClipper<T>
where
    T::Element: Float,
{
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.vin = x[0];
        let mut value = SVector::<_, 1>::new(
            self.vin
                .simd_clamp(-self.num_diodes_bwd, self.num_diodes_fwd),
        );
        NewtonRhapson::new(&*self, Some(self.sim_tol), NonZeroUsize::new(self.max_iter))
            .run_in_place(vector_view_mut(&mut value));
        self.last_vout = value[0];
        [value[0]]
    }
}

/// Analytical model of the diode clipper, described in the clippers notebook.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DiodeClipperModel<T> {
    /// A parameter
    pub a: T,
    /// B parameter
    pub b: T,
    /// Input scaling
    pub si: T,
    /// Output scaling
    pub so: T,
}

impl<T: Scalar> DSPMeta for DiodeClipperModel<T> {
    type Sample = T;
}

impl<T: Scalar> DiodeClipperModel<T> {
    /// Evaluate the saturator
    ///
    /// # Arguments
    ///
    /// * `x`: Input value
    ///
    /// returns: T
    #[replace_float_literals(T::from_f64(literal))]
    #[inline]
    pub fn eval(&self, x: T) -> T {
        let x = self.si * x;
        let lower = x.simd_lt(-self.a);
        let higher = x.simd_gt(self.b);
        let case1 = -T::simd_ln(1. - x - self.a) - self.a;
        let case2 = T::simd_ln(1. + x - self.b) + self.b;
        case1.select(lower, case2.select(higher, x)) * self.so
    }
}

impl<T: Scalar> Antiderivative<T> for DiodeClipperModel<T> {
    fn evaluate(&self, x: T) -> T {
        self.eval(x) / (self.si * self.so)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn antiderivative(&self, x: T) -> T {
        let cx = self.si * x;
        let lower = cx.simd_lt(-self.a);
        lower.if_else(
            || {
                let x0 = self.a + cx - 1.0;
                let num = (self.a - 1.0) * cx + x0 * T::simd_ln(-x0);
                let den = self.si * self.si;
                -num / den
            },
            || {
                let higher = cx.simd_gt(self.b);
                higher.if_else(
                    || {
                        let x0 = -self.b + cx + 1.0;
                        let num = (self.b - 1.0) * cx + x0 * T::simd_ln(x0);
                        let den = self.si * self.si;
                        num / den
                    },
                    || x * x / 2.0,
                )
            },
        ) / 2.0
    }
}

impl<T: Scalar> Default for DiodeClipperModel<T> {
    fn default() -> Self {
        Self::new_silicon(1, 1)
    }
}

impl<T: Scalar> DSPProcess<1, 1> for DiodeClipperModel<T> {
    #[inline(always)]
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.eval(x)]
    }
}

impl<T: Scalar> Saturator<T> for DiodeClipperModel<T> {
    #[inline(always)]
    fn saturate(&self, x: T) -> T {
        self.eval(x) / (self.si * self.so)
    }
}

impl<T: Scalar, const N: usize> MultiSaturator<T, N> for DiodeClipperModel<T> {
    fn multi_saturate(&self, x: [T; N]) -> [T; N] {
        x.map(|x| self.saturate(x))
    }

    fn update_state_multi(&mut self, _: [T; N], _: [T; N]) {}

    fn sat_jacobian(&self, x: [T; N]) -> [T; N] {
        x.map(|x| self.sat_diff(x))
    }
}

#[cfg(test)]
mod tests {
    use plotters::prelude::*;
    use std::hint;
    use valib_core::simd::SimdValue;

    use super::{DiodeClipper, DiodeClipperModel};
    use valib_core::dsp::DSPProcess;
    use valib_core::util::tests::{Plot, Series};

    fn dc_sweep(name: &str, mut dsp: impl DSPProcess<1, 1, Sample = f32>) {
        let results = Vec::from_iter(
            (-4800..=4800)
                .map(|i| i as f64 / 100.)
                .map(|v| dsp.process([v as f32])[0]),
        );
        let full_name = format!("{name}/dc_sweep");
        let plot_title = format!("DC sweep: {name}");
        Plot {
            title: &plot_title,
            bode: false,
            series: &[Series {
                label: name,
                samplerate: 100.0,
                series: &results,
                color: &Default::default(),
            }],
        }
        .create_svg(format!("plots/saturators/clippers/dc_sweep_{name}.svg"));
        insta::assert_csv_snapshot!(&*full_name, results, { "[]" => insta::rounded_redaction(4) });
    }

    fn drive_test(name: &str, mut dsp: impl DSPProcess<1, 1, Sample = f32>) {
        let sine_it = (0..).map(|i| i as f64 / 10.).map(f64::sin);
        let amp = (0..5000).map(|v| v as f64 / 5000. * 500.);
        let output = sine_it
            .zip(amp)
            .map(|(a, b)| a * b)
            .map(|v| hint::black_box(dsp.process([v as f32])[0]));
        let results = Vec::from_iter(output.map(|v| v.extract(0)));
        let full_name = format!("{name}/drive_test");
        let plot_title = format!("Drive test: {name}");
        Plot {
            title: &plot_title,
            bode: false,
            series: &[Series {
                label: "Output",
                samplerate: 10.,
                series: &results,
                color: &BLUE,
            }],
        }
        .create_svg(format!("plots/saturators/clippers/drive_{name}.svg"));
        insta::assert_csv_snapshot!(&*full_name, results, { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn snapshot_diode_clipper() {
        let clipper = DiodeClipper::new_led(3, 5, 0.0);
        dc_sweep("regressions/clipper_nr", clipper);
        drive_test("regressions/clipper_nr", clipper);
    }

    #[test]
    fn snapshot_diode_clipper_model() {
        let clipper = DiodeClipperModel::<f32>::new_led(3, 5);
        dc_sweep("regressions/clipper_model", clipper);
        drive_test("regressions/clipper_model", clipper);
    }
}
