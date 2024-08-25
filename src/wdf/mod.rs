use crate::dsp::{DSPMeta, DSPProcess};
use crate::Scalar;
use num_traits::Zero;
use simba::simd::SimdComplexField;
use std::any::Any;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

pub use adapters::*;
pub use diode::*;
pub use leaves::*;
pub use module::*;
pub use unadapted::*;

pub mod adapters;
pub mod diode;
pub mod leaves;
pub mod module;
pub mod unadapted;

pub type Node<T> = Rc<RefCell<T>>;

pub type NodeRef<'a, T> = Ref<'a, T>;

pub type NodeMut<'a, T> = RefMut<'a, T>;

#[inline]
pub fn node<T>(value: T) -> Node<T> {
    Rc::new(RefCell::new(value))
}

#[inline]
pub fn voltage<T: Scalar>(node: &impl Wdf<Scalar = T>) -> T {
    node.wave().voltage()
}

#[inline]
pub fn current<T: Scalar>(node: &impl AdaptedWdf<Scalar = T>) -> T {
    node.wave().current(node.impedance())
}

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
pub trait Wdf: Any {
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
    use crate::wdf::leaves::{Capacitor, ResistiveVoltageSource, Resistor};
    use crate::wdf::module::WdfModule;
    use crate::wdf::unadapted::{IdealVoltageSource, OpenCircuit};
    use crate::Scalar;
    use plotters::prelude::{BLUE, GREEN, RED};
    use std::f32::consts::TAU;

    #[test]
    fn test_voltage_divider() {
        let inp = node(IdealVoltageSource::new(12.));
        let out = node(Resistor::new(100.0));
        let mut module = WdfModule::new(
            inp,
            node(Inverter::new(node(Series::new(
                node(Resistor::new(100.0)),
                out.clone(),
            )))),
        );
        module.next_sample();
        assert_eq!(6.0, voltage(&out));
    }

    #[test]
    fn test_lowpass_filter() {
        const C: f32 = 33e-9;
        const CUTOFF: f32 = 256.0;
        const FS: f32 = 4096.0;
        let r = f32::recip(TAU * C * CUTOFF);
        let rvs = node(ResistiveVoltageSource::new(r, 0.));
        let mut tree = WdfModule::new(
            node(OpenCircuit::default()),
            node(Parallel::new(rvs.clone(), node(Capacitor::new(FS, C)))),
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
