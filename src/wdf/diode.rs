use crate::saturators::clippers::DiodeClipperModel;
use crate::wdf::{Wave, Wdf};
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
pub struct Diode<T> {
    pub model: DiodeClipperModel<T>,
    r: T,
    a: T,
    b: T,
}

impl<T: Scalar> Wdf for Diode<T> {
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
        self.b = self.model.eval(self.a) * T::simd_exp(-self.r) - self.a;
        self.b
    }

    fn set_port_resistance(&mut self, resistance: Self::Scalar) {
        self.r = resistance;
    }
}

impl<T: Zero> Diode<T> {
    pub fn new(model: DiodeClipperModel<T>) -> Self {
        Self {
            model,
            r: T::zero(),
            a: T::zero(),
            b: T::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::tests::{Plot, Series};
    use crate::wdf::{node, voltage, Capacitor, Parallel, ResistiveVoltageSource, WdfModule};
    use plotters::style::{BLUE, RED};
    use std::f32::consts::TAU;

    #[test]
    fn test_diode_clipper() {
        let rvs = node(ResistiveVoltageSource::new(1000f32, 0.));
        let fs = 4096.0;
        let mut tree = WdfModule::new(
            node(Diode::new(DiodeClipperModel {
                si: 10.,
                so: 10.,
                a: 38.,
                b: 0.05,
            })),
            node(Parallel::new(rvs.clone(), node(Capacitor::new(fs, 33e-9)))),
        );

        let input = (0..4096)
            .map(|i| TAU * i as f32 / fs)
            .map(f32::sin)
            //.map(|x| 10.0 * x)
            .collect::<Vec<_>>();
        let output = input
            .iter()
            .copied()
            .map(|x| {
                rvs.borrow_mut().vs = x;
                tree.next_sample();
                voltage(&tree.root)
            })
            .collect::<Vec<_>>();

        Plot {
            title: "Diode Clipper",
            bode: false,
            series: &[
                Series {
                    label: "Input",
                    samplerate: fs,
                    series: &input,
                    color: &BLUE,
                },
                Series {
                    label: "Output",
                    samplerate: fs as _,
                    series: &output,
                    color: &RED,
                },
            ],
        }
        .create_svg("plots/wdf/diode_clipper.svg");
        insta::assert_csv_snapshot!(&output, { "[]" => insta::rounded_redaction(4) })
    }
}
