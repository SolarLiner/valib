mod diode_clipper_model_data;

use std::{fmt, ops::Not};

use nalgebra::{ComplexField, SMatrix, SVector};
use num_traits::Float;
use numeric_literals::replace_float_literals;
use simba::simd::{SimdBool, SimdComplexField, SimdValue};

use crate::{dsp::DSP, math::newton_rhapson_tol_max_iter};
use crate::{math::RootEq, saturators::Saturator, Scalar};

#[derive(Debug, Copy, Clone)]
pub struct DiodeClipper<T> {
    pub isat: T,
    pub n: T,
    pub vt: T,
    pub vin: T,
    pub num_diodes_fwd: T,
    pub num_diodes_bwd: T,
    pub sim_tol: T,
    pub max_iter: usize,
    last_vout: T,
}

impl<T: Copy> DiodeClipper<T> {
    pub fn reset(&mut self) {
        self.last_vout = self.vin;
    }
}

impl<T: Scalar> RootEq<T, 1> for DiodeClipper<T> {
    #[cfg_attr(test, inline(never))]
    #[cfg_attr(not(test), inline)]
    #[replace_float_literals(T::from_f64(literal))]
    fn eval(&self, input: &nalgebra::SVector<T, 1>) -> nalgebra::SVector<T, 1> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        if vout.simd_gt(16.0).any() {
            println!();
        }
        let expn = T::simd_exp(expin / self.num_diodes_fwd).simd_min(1e35);
        let expm = T::simd_exp(-expin / self.num_diodes_bwd).simd_min(1e35);
        let res = self.isat * (expn - expm) + 2. * vout - self.vin;
        SVector::<_, 1>::new(res)
    }

    #[cfg_attr(test, inline(never))]
    #[cfg_attr(not(test), inline)]
    #[replace_float_literals(T::from_f64(literal))]
    fn j_inv(&self, input: &nalgebra::SVector<T, 1>) -> Option<SMatrix<T, 1, 1>> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        if vout.simd_gt(16.0).any() {
            println!();
        }
        let expn = T::simd_exp(expin / self.num_diodes_fwd).simd_min(1e35);
        let expm = T::simd_exp(-expin / self.num_diodes_bwd).simd_min(1e35);
        let res = v * self.isat * (expn / self.num_diodes_fwd + expm / self.num_diodes_bwd) + 2.;
        // Biasing to prevent divisions by zero, less accurate around zero
        let ret = (1e-6).select(res.simd_abs().simd_lt(1e-6), res).simd_recip();
        Some(SMatrix::<_, 1, 1>::new(ret))
    }
}

impl<T: Scalar> DiodeClipper<T> {
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
            last_vout: vin,
        }
    }

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
            last_vout: vin,
        }
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn new_led(nf: usize, nb: usize, vin: T) -> DiodeClipper<T> {
        Self {
            isat: 2.96406e-12,
            n: 2.475312,
            vt: 23e-3,
            vin,
            num_diodes_fwd: T::from_f64(nf as f64),
            num_diodes_bwd: T::from_f64(nb as f64),
            sim_tol: 1e-3,
            max_iter: 50,
            last_vout: vin,
        }
    }
}

impl<T: Scalar + fmt::Display> DSP<1, 1> for DiodeClipper<T>
where
    T::Element: Float,
{
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.vin = x[0];
        let mut vout = SVector::<_, 1>::new(self.last_vout);
        newton_rhapson_tol_max_iter(self, &mut vout, self.sim_tol, self.max_iter);
        self.last_vout = vout[0];
        [vout[0]]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DiodeClipperModel<T> {
    pub a: T,
    pub b: T,
    pub si: T,
    pub so: T,
}

impl<T: Scalar> DiodeClipperModel<T> {
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

impl<T: Scalar> Default for DiodeClipperModel<T> {
    fn default() -> Self {
        Self::new_silicon(1, 1)
    }
}

impl<T: Scalar> DSP<1, 1> for DiodeClipperModel<T> {
    type Sample = T;

    #[inline(always)]
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.eval(x)]
    }
}

impl<T: Scalar> Saturator<T> for DiodeClipperModel<T> {
    #[inline]
    #[replace_float_literals(T::from_f64(literal))]
    fn saturate(&self, x: T) -> T {
        let x = self.si / self.so * x;
        let out = self.eval(x);
        out * self.so / self.si
    }
}

#[cfg(test)]
mod tests {
    use std::hint;

    use crate::dsp::DSP;
    use simba::simd::SimdValue;

    use super::{DiodeClipper, DiodeClipperModel};

    fn dc_sweep(name: &str, mut dsp: impl DSP<1, 1, Sample = f32>) {
        let results = Vec::from_iter(
            (-4800..=4800)
                .map(|i| i as f64 / 100.)
                .map(|v| dsp.process([v as f32])[0]),
        );
        let full_name = format!("{name}/dc_sweep");
        insta::assert_csv_snapshot!(&*full_name, results, { "[]" => insta::rounded_redaction(4) });
    }

    fn drive_test(name: &str, mut dsp: impl DSP<1, 1, Sample = f32>) {
        let sine_it = (0..).map(|i| i as f64 / 10.).map(f64::sin);
        let amp = (0..5000).map(|v| v as f64 / 5000. * 500.);
        let output = sine_it
            .zip(amp)
            .map(|(a, b)| a * b)
            .map(|v| {
                let out = dsp.process([v as f32])[0];
                hint::black_box(out)
            });
        let results = Vec::from_iter(output.map(|v| v.extract(0)));
        let full_name = format!("{name}/drive_test");
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
