use crate::dsp::{DSPMeta, DSPProcess};
use crate::Scalar;
pub use adapters::*;
use atomic_refcell::{AtomicRef, AtomicRefCell, AtomicRefMut};
pub use diode::*;
pub use leaves::*;
pub use module::*;
use num_traits::Zero;
use simba::simd::SimdComplexField;
use std::any::Any;
use std::sync::Arc;
pub use unadapted::*;

pub mod adapters;
pub mod diode;
pub mod dsl;
pub mod leaves;
pub mod module;
pub mod unadapted;

pub type Node<T> = Arc<AtomicRefCell<T>>;

pub type NodeRef<'a, T> = AtomicRef<'a, T>;

pub type NodeMut<'a, T> = AtomicRefMut<'a, T>;

#[derive(Debug, Copy, Clone)]
pub struct Wave<T> {
    pub a: T,
    pub b: T,
}

impl<T: Scalar> Wave<T> {
    pub fn voltage(&self) -> T {
        (self.a + self.b) / T::from_f64(2.0)
    }

    pub fn current(&self, resistance: T) -> T {
        (self.a - self.b) / (T::from_f64(2.0) * resistance)
    }
}

#[allow(unused)]
pub trait Wdf {
    type Scalar: Scalar;
    fn wave(&self) -> Wave<Self::Scalar>;
    fn incident(&mut self, x: Self::Scalar);
    fn reflected(&mut self) -> Self::Scalar;
    fn set_samplerate(&mut self, samplerate: f64) {}
    fn set_port_resistance(&mut self, resistance: Self::Scalar) {}
    fn reset(&mut self);
}

impl<'a, T: Wdf> Wdf for &'a mut T {
    type Scalar = T::Scalar;

    fn wave(&self) -> Wave<Self::Scalar> {
        T::wave(self)
    }

    fn incident(&mut self, x: Self::Scalar) {
        T::incident(self, x)
    }

    fn reflected(&mut self) -> Self::Scalar {
        T::reflected(self)
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        T::set_samplerate(self, samplerate)
    }

    fn reset(&mut self) {
        T::reset(self)
    }
}

pub trait AdaptedWdf: Wdf {
    fn impedance(&self) -> Self::Scalar {
        self.admittance().simd_recip()
    }
    fn admittance(&self) -> Self::Scalar {
        self.impedance().simd_recip()
    }
}

impl<'a, T: AdaptedWdf> AdaptedWdf for &'a mut T {
    fn impedance(&self) -> Self::Scalar {
        T::impedance(self)
    }

    fn admittance(&self) -> Self::Scalar {
        T::admittance(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::tests::Plot;
    use crate::wdf::adapters::{Inverter, Parallel, Series};
    use crate::wdf::dsl::*;
    use crate::wdf::leaves::{Capacitor, ResistiveVoltageSource, Resistor};
    use crate::wdf::module::WdfModule;
    use crate::wdf::unadapted::{IdealVoltageSource, OpenCircuit};
    use crate::Scalar;
    use plotters::prelude::{BLUE, GREEN, RED};
    use std::f32::consts::TAU;

    #[test]
    fn test_voltage_divider() {
        let inp = ivsource(12.);
        let out = resistor(100.0);
        let mut module = module(inp, inverter(series(resistor(100.0), out.clone())));
        module.process_sample();
        assert_eq!(6.0, voltage(&out));
    }

    #[test]
    fn test_lowpass_filter() {
        const C: f32 = 33e-9;
        const CUTOFF: f32 = 256.0;
        const FS: f32 = 4096.0;
        let r = f32::recip(TAU * C * CUTOFF);
        let rvs = rvsource(r, 0.);
        let mut module = module(open_circuit(), parallel(rvs.clone(), capacitor(FS, C)));

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
                crate::util::tests::Series {
                    label: "Input",
                    samplerate: FS,
                    series: &input,
                    color: &BLUE,
                },
                crate::util::tests::Series {
                    label: "Output",
                    samplerate: FS,
                    series: &output,
                    color: &RED,
                },
            ],
        }
        .create_svg("plots/wdf/low_pass.svg");
        insta::assert_csv_snapshot!(&output, { "[]" => insta::rounded_redaction(4) })
    }
}
