use crate::saturators::clippers::DiodeClipperModel;
use crate::wdf::{Wave, Wdf, WdfDsp};
use crate::Scalar;
use num_traits::Zero;
use numeric_literals::replace_float_literals;
use simba::simd::SimdValue;

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

#[derive(Debug, Copy, Clone)]
pub struct RModel<T> {
    pub q: T,
    pub y0: T,
    pub b: T,
    pub m: T,
    pub n: T,
}

impl<T: Scalar> RModel<T>
where
    T::SimdRealField: Scalar,
{
    pub fn eval(&self, r: T) -> T {
        -T::simd_ln(self.q * r)
            + self.y0
            + self.b * r.simd_powf(T::SimdRealField::from_f64(0.25))
            + self.m * r.simd_sqrt()
            + self.n * r
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::tests::{Plot, Series};
    use crate::wdf::{node, voltage, Capacitor, Parallel, ResistiveVoltageSource, WdfModule};
    use plotters::style::*;
    use std::f32::consts::TAU;

    #[test]
    fn test_diode_clipper_lambertw() {
        const C: f32 = 33e-9;
        const CUTOFF: f32 = 256.0;
        const FS: f32 = 4096.0;
        let r = f32::recip(TAU * C * CUTOFF);
        let c = node(Capacitor::new(FS, C));
        let rvs = node(ResistiveVoltageSource::new(r, 0.));
        let mut tree = WdfModule::new(
            node(DiodeLambert::germanium(1)),
            node(Parallel::new(rvs.clone(), c.clone())),
        );

        let input = (0..256)
            .map(|i| f32::fract(50.0 * i as f32 / FS))
            .map(|x| 2.0 * x - 1.)
            .map(|x| 1.0 * x)
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(input.len());

        for x in input.iter().copied() {
            rvs.borrow_mut().vs = x;
            tree.next_sample();
            output.push(voltage(&tree.root));
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
        let c = node(Capacitor::new(FS, C));
        let rvs = node(ResistiveVoltageSource::new(r, 0.));
        let mut tree = WdfModule::new(
            node(DiodeModel::new(DiodeClipperModel::new_germanium(1, 1))),
            node(Parallel::new(rvs.clone(), c.clone())),
        );

        let input = (0..256)
            .map(|i| f32::fract(50.0 * i as f32 / FS))
            .map(|x| 2.0 * x - 1.)
            .collect::<Vec<_>>();
        let mut output = Vec::with_capacity(input.len());

        for x in input.iter().copied() {
            rvs.borrow_mut().vs = 10. * x;
            tree.next_sample();
            output.push(voltage(&tree.root));
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
}
