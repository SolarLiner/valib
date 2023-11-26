mod diode_clipper_model_data;

use std::{fmt, ops::Not};

use nalgebra::{ComplexField, SMatrix, SVector};
use numeric_literals::replace_float_literals;

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
}

impl<T: Scalar> RootEq<T, 1> for DiodeClipper<T> {
    #[inline]
    #[replace_float_literals(T::from_f64(literal))]
    fn eval(&self, input: &nalgebra::SVector<T, 1>) -> nalgebra::SVector<T, 1> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        let expn = T::simd_recip(expin / self.num_diodes_fwd);
        let expm = T::simd_recip(-expin / self.num_diodes_bwd);
        let res = self.isat * (expn - expm) + 2. * vout - self.vin;
        SVector::<_, 1>::new(res)
    }

    #[inline]
    #[replace_float_literals(T::from_f64(literal))]
    fn j_inv(&self, input: &nalgebra::SVector<T, 1>) -> Option<SMatrix<T, 1, 1>> {
        let vout = input[0];
        let v = T::simd_recip(self.n * self.vt);
        let expin = vout * v;
        let expn = T::simd_recip(expin / self.num_diodes_fwd);
        let expm = T::simd_recip(-expin / self.num_diodes_bwd);
        let res = v * self.isat * (expn / self.num_diodes_fwd + expm / self.num_diodes_bwd) + 2.;
        res.is_zero()
            .not()
            .then_some(SMatrix::<_, 1, 1>::new(res.simd_recip()))
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
        }
    }
}

impl<T: Scalar + fmt::Debug> DSP<1, 1> for DiodeClipper<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.vin = x[0];
        let mut vout = SVector::<_, 1>::new(x[0].simd_tanh());
        newton_rhapson_steps(self, &mut vout, 4);
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
        let case2 = T::simd_ln(1. + x + self.b) + self.b;
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
    use std::fs::File;
    use std::io::Write;

    use nalgebra::SVector;

    use crate::math::newton_rhapson_tolerance;

    use super::DiodeClipper;

    #[test]
    fn evaluate_diode_clipper() {
        let mut clipper = DiodeClipper::new_led(3, 5, 0.);
        let mut file = File::create("clipper.tsv").unwrap();
        writeln!(file, "\"in\"\t\"out\"\t\"iter\"").unwrap();
        for i in -4800..4800 {
            clipper.vin = i as f64 / 100.;
            let mut vout = SVector::<_, 1>::new(f64::tanh(clipper.vin));
            let iter = newton_rhapson_tolerance(&clipper, &mut vout, 1e-3);
            writeln!(file, "{}\t{}\t{}", clipper.vin, vout[0], iter).unwrap();
        }
    }
}
