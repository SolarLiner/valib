mod diode_clipper_model_data;

use std::{fmt, ops::Not};

use nalgebra::{ComplexField, SMatrix, SVector};
use num_traits::Float;
use numeric_literals::replace_float_literals;

use crate::{dsp::DSP, math::newton_rhapson_tol_max_iter};
use crate::{
    math::{newton_rhapson_steps, RootEq},
    saturators::Saturator,
    Scalar,
};
use crate::dsp::DSP;

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

impl<T: Scalar> RootEq<T, 1> for DiodeClipper<T> {
    #[inline]
    #[replace_float_literals(T::from_f64(literal))]
    fn eval(&self, input: &nalgebra::SVector<T, 1>) -> nalgebra::SVector<T, 1> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        let expn = T::simd_exp(expin / self.num_diodes_fwd);
        let expm = T::simd_exp(-expin / self.num_diodes_bwd);
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
        let expn = T::simd_exp(expin / self.num_diodes_fwd);
        let expm = T::simd_exp(-expin / self.num_diodes_bwd);
        let res = v * self.isat * (expn / self.num_diodes_fwd + expm / self.num_diodes_bwd) + 2.;
        // Biasing to prevent divisions by zero, less accurate around zero
        Some(SMatrix::<_, 1, 1>::new((res + 1e-6).simd_recip()))
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

impl<T: Scalar + fmt::Debug> DSP<1, 1> for DiodeClipper<T>
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
    use std::io::Write;
    use std::{fs::File, result};

    use nalgebra::SVector;

    use crate::{
        dsp::DSP,
        math::{newton_rhapson_tolerance, RootEq},
    };

    use super::{DiodeClipper, DiodeClipperModel};

    fn snapshot(name: &str, mut dsp: impl DSP<1, 1, Sample = f64>) {
        let results = Vec::from_iter(
            (-4800..=4800)
                .map(|i| i as f64 / 100.)
                .map(|v| dsp.process([v])[0]),
        );
        insta::assert_csv_snapshot!(name, results, { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn snapshot_diode_clipper() {
        let clipper = DiodeClipper::new_led(3, 5, 0.0);
        snapshot("regressions/clipper_nr", clipper);
    }

    #[test]
    fn snapshot_diode_clipper_model() {
        let clipper = DiodeClipperModel::new_led(3, 5);
        snapshot("regressions/clipper_model", clipper);
    }
}
