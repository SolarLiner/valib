use crate::math::nr::{newton_rhapson_tol_max_iter, RootEq};
use crate::saturators::clippers::{DiodeClipper, DiodeClipperModel};
use crate::wdf::unadapted::WdfDsp;
use crate::wdf::{Wave, Wdf};
use crate::Scalar;
use nalgebra::{SMatrix, SVector, SimdBool};
use num_traits::{Float, Num, Zero};
use numeric_literals::replace_float_literals;

#[inline]
#[replace_float_literals(T::from_f64(literal))]
fn lambdertw<T: Scalar>(x: T) -> T {
    let threshold = x.simd_lt(2.26445);
    let c = (1.546865557).select(threshold, 1.0);
    let d = (2.250366841).select(threshold, 0.0);
    let a = (-0.737769969).select(threshold, 0.0);
    let logterm = T::simd_ln(c * x + d);
    let loglogterm = logterm.simd_ln();

    let minusw = -a - logterm + loglogterm - loglogterm / logterm;
    let expminusw = minusw.simd_exp();
    let xexpminusw = x * expminusw;
    let pexpminusw = xexpminusw - minusw;

    (2.0 * xexpminusw - minusw * (4.0 * xexpminusw - minusw * pexpminusw))
        / (2.0 + pexpminusw * (2.0 - minusw))
}

pub type DiodeModel<T> = WdfDsp<DiodeClipperModel<T>>;

#[derive(Debug, Copy, Clone)]
pub struct DiodeLambert<T> {
    pub isat: T,
    pub nvt: T,
    pub nf: T,
    pub nb: T,
    r: T,
    a: T,
    b: T,
}

impl<T: Scalar> Wdf for DiodeLambert<T> {
    type Scalar = T;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: self.b,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        let mu0 = self.nf.select(self.a.is_simd_positive(), self.nf);
        let mu1 = self.nb.select(self.a.is_simd_positive(), self.nf);
        let ris_vt = self.r * self.isat / self.nvt;
        let lam = self.a.simd_signum();
        let lam_a_vt = self.a * lam / self.nvt;
        let log_ris_vt_mu0 = T::simd_ln(ris_vt / mu0);
        let log_ris_vt_mu1 = T::simd_ln(ris_vt / mu1);
        let e0 = T::simd_exp(log_ris_vt_mu0 + lam_a_vt / mu0);
        let e1 = -T::simd_exp(log_ris_vt_mu1 - lam_a_vt / mu1);
        let w0 = lambdertw(e0);
        let w1 = lambdertw(e1);
        let inner = mu0 * w0 + mu1 * w1;
        self.b = self.a - T::from_f64(2.0) * lam * self.nvt * inner;
        self.b
    }

    fn set_port_resistance(&mut self, resistance: Self::Scalar) {
        self.r = resistance;
    }

    fn reset(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
        self.r.set_zero();
    }
}

impl<T: Num + Zero> DiodeLambert<T> {
    pub fn new(data: DiodeClipper<T>) -> Self {
        Self {
            isat: data.isat,
            nvt: data.n * data.vt,
            nf: data.num_diodes_fwd,
            nb: data.num_diodes_bwd,
            r: T::zero(),
            a: T::zero(),
            b: T::zero(),
        }
    }
}

struct DiodeRootEq<T: Scalar> {
    pub isat: T,
    pub n: T,
    pub vt: T,
    pub nf: T,
    pub nb: T,
    r: T,
    a: T,
}

impl<T: Scalar> RootEq<T, 1> for DiodeRootEq<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn eval(&self, input: &SVector<T, 1>) -> SVector<T, 1> {
        let b = input[0];
        let r2 = 2. * self.r;
        let log_r2isat = r2.simd_ln() + self.isat.simd_ln();
        let exp_op = (self.a + b) / (2.0 * self.n * self.vt);
        let e0 = log_r2isat + exp_op / self.nf;
        let e1 = log_r2isat - exp_op / self.nb;
        let x0 = e0.simd_exp() - e1.simd_exp();
        let inner = (x0 + -self.a + b) / r2;
        SVector::<_, 1>::new(inner)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn j_inv(&self, input: &SVector<T, 1>) -> Option<SMatrix<T, 1, 1>> {
        let b = input[0];
        let log_risat = self.r.simd_ln() + self.isat.simd_ln();
        let log_m = self.nf.simd_ln();
        let log_n = self.nb.simd_ln();
        let exp_op = (self.a + b) / (2.0 * self.n * self.vt);
        let e0 = log_m + log_risat - exp_op / self.nf;
        let e1 = log_n + log_risat + exp_op / self.nb;
        let mnnvt = self.nf * self.nb * self.n * self.vt;
        let inner = 2.0 * self.r * mnnvt / (mnnvt + e0.simd_exp() + e1.simd_exp());
        Some(SMatrix::<_, 1, 1>::new(inner))
    }
}

pub struct DiodeNR<T: Scalar> {
    pub root_eq: DiodeRootEq<T>,
    pub max_tolerance: T,
    pub max_iter: usize,
    b: T,
}

impl<T: Scalar> DiodeNR<T> {
    pub fn from_data(data: DiodeClipper<T>) -> Self {
        Self {
            root_eq: DiodeRootEq {
                isat: data.isat,
                n: data.n,
                vt: data.vt,
                nf: data.num_diodes_fwd,
                nb: data.num_diodes_bwd,
                a: T::zero(),
                r: T::zero(),
            },
            max_tolerance: T::from_f64(1e-4),
            max_iter: 50,
            b: T::zero(),
        }
    }
}

impl<T: Scalar<Element: Float>> Wdf for DiodeNR<T> {
    type Scalar = T;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.root_eq.a,
            b: self.b,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        self.root_eq.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        let mut value = SVector::<_, 1>::new(-self.root_eq.a);
        newton_rhapson_tol_max_iter(&self.root_eq, &mut value, T::from_f64(1e-4), 1000);
        self.b = value[0];
        self.b
    }

    fn set_port_resistance(&mut self, resistance: Self::Scalar) {
        self.root_eq.r = resistance;
    }

    fn reset(&mut self) {
        self.root_eq.a.set_zero();
        self.root_eq.r.set_zero();
        self.b.set_zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::saturators::clippers::DiodeClipper;
    use crate::util::tests::{Plot, Series};
    use crate::wdf::adapters::Parallel;
    use crate::wdf::dsl::*;
    use crate::wdf::leaves::{Capacitor, ResistiveVoltageSource};
    use crate::wdf::module::WdfModule;
    use plotters::style::*;
    use std::f32::consts::TAU;

    #[test]
    fn test_diode_clipper_lambertw() {
        const C: f32 = 33e-9;
        const CUTOFF: f32 = 256.0;
        const FS: f32 = 4096.0;
        let r = f32::recip(TAU * C * CUTOFF);
        let c = capacitor(FS, C);
        let rvs = rvsource(r, 0.);
        let diode = diode_lambert(DiodeClipper::new_germanium(1, 1, 0.));
        let mut module = module(diode, parallel(rvs.clone(), c.clone()));

        let input = (0..256)
            .map(|i| f32::fract(50.0 * i as f32 / FS))
            .map(|x| 2.0 * x - 1.)
            .map(|x| 1.0 * x)
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(input.len());

        for x in input.iter().copied() {
            node_mut(&rvs).vs = x;
            module.process_sample();
            output.push(voltage(&module.root));
        }

        Plot {
            title: "Diode Clipper",
            bode: false,
            series: &[
                Series {
                    label: "Input",
                    samplerate: FS,
                    series: &input,
                    color: &BLUE,
                },
                Series {
                    label: "Output",
                    samplerate: FS as _,
                    series: &output,
                    color: &RED,
                },
            ],
        }
        .create_svg("plots/wdf/diode_clipper_lambertw.svg");
        insta::assert_csv_snapshot!(&output, { "[]" => insta::rounded_redaction(4) })
    }

    #[test]
    fn test_diode_clipper_model() {
        const C: f32 = 33e-9;
        const CUTOFF: f32 = 256.0;
        const FS: f32 = 4096.0;
        let r = f32::recip(TAU * C * CUTOFF);
        let c = capacitor(FS, C);
        let rvs = rvsource(r, 0.);
        let mut module = module(
            diode_model(DiodeClipperModel::new_germanium(1, 1)),
            parallel(rvs.clone(), c.clone()),
        );

        let input = (0..256)
            .map(|i| f32::fract(50.0 * i as f32 / FS))
            .map(|x| 2.0 * x - 1.)
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(input.len());

        for x in input.iter().copied() {
            node_mut(&rvs).vs = 10. * x;
            module.process_sample();
            output.push(voltage(&module.root));
        }

        Plot {
            title: "Diode Clipper",
            bode: false,
            series: &[
                Series {
                    label: "Input",
                    samplerate: FS,
                    series: &input,
                    color: &BLUE,
                },
                Series {
                    label: "Output",
                    samplerate: FS as _,
                    series: &output,
                    color: &RED,
                },
            ],
        }
        .create_svg("plots/wdf/diode_clipper_model.svg");
        insta::assert_csv_snapshot!(&output, { "[]" => insta::rounded_redaction(4) })
    }

    #[test]
    fn test_diode_clipper_nr() {
        const C: f32 = 33e-9;
        const CUTOFF: f32 = 256.0;
        const FS: f32 = 4096.0;
        let r = f32::recip(TAU * C * CUTOFF);
        let c = capacitor(FS, C);
        let rvs = rvsource(r, 0.);
        let diode = {
            let data = DiodeClipper::new_germanium(1, 1, 0.);
            diode_nr(data)
        };
        let mut module = module(diode, parallel(rvs.clone(), c.clone()));

        let input = (0..256)
            .map(|i| f32::fract(50.0 * i as f32 / FS))
            .map(|x| 2.0 * x - 1.)
            .map(|x| 1.0 * x)
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(input.len());

        for x in input.iter().copied() {
            node_mut(&rvs).vs = x;
            module.process_sample();
            output.push(voltage(&module.root));
        }

        Plot {
            title: "Diode Clipper",
            bode: false,
            series: &[
                Series {
                    label: "Input",
                    samplerate: FS,
                    series: &input,
                    color: &BLUE,
                },
                Series {
                    label: "Output",
                    samplerate: FS as _,
                    series: &output,
                    color: &RED,
                },
            ],
        }
        .create_svg("plots/wdf/diode_clipper_nr.svg");
        insta::assert_csv_snapshot!(&output, { "[]" => insta::rounded_redaction(4) })
    }
}
