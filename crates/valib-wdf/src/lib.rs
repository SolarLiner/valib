#![warn(missing_docs)]
//! # Wave Digital Filters
//!
//! This crate provides an implementation of wave digital filters as a tree of adapted nodes, with a
//! module type for assembling an adapted tree with an unadaptable root.
///
/// Waves in this WDF implementation are voltage wave, which is the most common kind. Wave variables
/// are defined  with the following equation:
///
/// $$ a = v + R_P * i \\ b = v - R_P * i $$
///
/// Conversely, Kirchhoff variables are defined in terns of wave variables as follows:
///
/// $$ v = \frac{a + b}{2} || i = \frac{a - b}{2 R_P} $$
///
/// WDF Trees are defined with a single upward facing port, which is the one that gets adapted. Trees
/// are combined by plugging their upward facing ports into any of the downward facing ports of
/// adapters. One-port are put at the leaves of the tree, and represent 2-port components like
/// resistors or capacitors.
pub use adapters::*;
use atomic_refcell::{AtomicRef, AtomicRefCell, AtomicRefMut};
pub use diode::*;
pub use leaves::*;
pub use module::*;
use num_traits::Zero;
use std::sync::Arc;
pub use unadapted::*;
use valib_core::dsp::DSPMeta;
use valib_core::simd::SimdComplexField;
use valib_core::Scalar;

pub mod adapters;
pub mod diode;
pub mod dsl;
pub mod leaves;
pub mod module;
pub mod unadapted;
//mod v2;

/// Type definition of nodes of the WDF tree.
pub type Node<T> = Arc<AtomicRefCell<T>>;

/// Type definition of node references.
pub type NodeRef<'a, T> = AtomicRef<'a, T>;

/// Type definition of mutable node references.
pub type NodeMut<'a, T> = AtomicRefMut<'a, T>;

/// Electrical value in the wave domain.
#[derive(Debug, Copy, Clone)]
pub struct Wave<T> {
    /// A value, corresponding to V + I * R in the Kirchhoff domain.
    pub a: T,
    /// B value, corresponding to V - I * R in the Kirchhoff domain.
    pub b: T,
}

impl<T: Scalar> Wave<T> {
    /// Compute the voltage value of the wave.
    pub fn voltage(&self) -> T {
        (self.a + self.b) / T::from_f64(2.0)
    }

    /// Compute the current value of the wave, given a port resistance.
    ///
    /// # Arguments
    ///
    /// * `resistance`: Port resistance to use in computing the current.
    ///
    /// returns: T
    pub fn current(&self, resistance: T) -> T {
        (self.a - self.b) / (T::from_f64(2.0) * resistance)
    }
}

/// Wave Digital Filter type trait.
///
/// All WDF nodes must implement this trait. There is no restriction on the adaptability of the node
/// here, only that it must receive an incident wave (variable $a$), and reflect it back (variable
/// $b$).
#[allow(unused)]
pub trait Wdf {
    /// Scalar type used within this node.
    type Scalar: Scalar;
    /// Wave variables at the upward facing port of this node.
    fn wave(&self) -> Wave<Self::Scalar>;
    /// Update the internal state of this node given the incident wave (variable $a$).
    ///
    /// # Arguments
    ///
    /// * `a`: Incident wave
    ///
    /// returns: ()
    fn incident(&mut self, a: Self::Scalar);
    /// Update the internal state of this node and output the reflected wave (variable $b$).
    fn reflected(&mut self) -> Self::Scalar;
    /// Set the sample rate of this node
    ///
    /// # Arguments
    ///
    /// * `samplerate`: New sample rate
    ///
    /// returns: ()
    fn set_samplerate(&mut self, samplerate: f64) {}
    /// Update the port resistance of other port that is plugged into this node's upward facing port.
    ///
    /// # Arguments
    ///
    /// * `resistance`: Port resistance
    ///
    /// returns: ()
    fn set_port_resistance(&mut self, resistance: Self::Scalar) {}
    /// Reset the internal state of this node.
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

/// Adapted WDF node. Nodes which can set a specific port resistance to prevent delay-free loops
/// (where $b$ immediately depends on $a$ without unit delays) can be adapted, making them composable.
///
/// **Implementation note**: One of [`Self::impedance`] or [`Self::admittance`] (or both) must be
/// implemented. By default the methods call each other (to allow the user to choose which one to
/// implement), however, failure to do so will result in stack overflow from infinite recursion.
pub trait AdaptedWdf: Wdf {
    /// Return the impedance of the upward facing port.
    fn impedance(&self) -> Self::Scalar {
        self.admittance().simd_recip()
    }
    /// Return the admittance of the upward facing port.
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
    use crate::dsl::*;
    use plotters::prelude::{BLUE, RED};
    use std::f32::consts::TAU;
    use valib_core::util::tests::Plot;

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
                valib_core::util::tests::Series {
                    label: "Input",
                    samplerate: FS,
                    series: &input,
                    color: &BLUE,
                },
                valib_core::util::tests::Series {
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
