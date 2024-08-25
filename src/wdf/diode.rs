use crate::math::{newton_rhapson_tol_max_iter, RootEq};
use crate::saturators::clippers::{DiodeClipper, DiodeClipperModel};
use crate::wdf::unadapted::WdfDsp;
use crate::wdf::{Wave, Wdf};
use crate::Scalar;
use nalgebra::{SMatrix, SVector};
use num_traits::{Float, Zero};
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
    pub vt: T,
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
        let ris_vt = self.r * self.isat / self.vt;
        let lam = self.a.simd_signum();
        let lam_a_vt = self.a * lam / self.vt;
        let inner = lambdertw(ris_vt * lam_a_vt.simd_exp()) + (-ris_vt * (-lam_a_vt).simd_exp());
        self.b = self.a - T::from_f64(2.0) * lam * self.vt * inner;
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

impl<T: Zero> DiodeLambert<T> {
    pub fn new(isat: T, vt: T) -> Self {
        Self {
            isat,
            vt,
            r: T::zero(),
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> DiodeLambert<T> {
    #[replace_float_literals(T::from_f64(literal))]
    pub fn germanium(num_diodes: usize) -> Self {
        Self::new(200e-9, T::from_f64(num_diodes as _) * 23e-3)
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
        let x0 = 0.5 * (self.a - b) / (self.r * self.n * self.vt);
        SVector::<_, 1>::new(
            -0.5 * (self.a + b)
                + self.isat
                    * ((1.0 - (-x0 / self.nf).simd_exp()) + (x0 / self.nb).simd_exp() - 1.0),
        )
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn j_inv(&self, input: &SVector<T, 1>) -> Option<SMatrix<T, 1, 1>> {
        let b = input[0];
        let x0 = self.nf * self.nb * self.n * self.vt;
        let x1 = 0.5 * (self.a - b) / (self.r * self.n * self.vt);
        let j_inv = SVector::<_, 1>::new(
            -2.0 * x0
                / (self.nf * self.isat * (-x1 / self.nb).simd_exp()
                    + self.nb * self.isat * (x1 / self.nf).simd_exp()
                    + x0),
        );
        Some(j_inv)
    }
}

pub struct DiodeNR<T: Scalar> {
    pub root_eq: DiodeRootEq<T>,
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
        newton_rhapson_tol_max_iter(&self.root_eq, &mut value, T::from_f64(1e-6), 100);
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
        let diode = {
            let data = DiodeClipper::new_germanium(1, 1, 0.);
            diode_lambert(data.isat, data.vt)
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
