//! # WDF leaves
//!
//! Provides nodes which only have one port.
use num_traits::Zero;

use crate::{AdaptedWdf, Wave, Wdf};
use valib_core::Scalar;

/// Resistive voltage source leaf.
///
/// This node can be adapter thanks to the resistance running in series to the voltage source.
#[derive(Debug, Copy, Clone)]
pub struct ResistiveVoltageSource<T> {
    /// Voltage source value (V)
    pub vs: T,
    /// Series resistance value (Ohm)
    pub r: T,
    a: T,
    b: T,
}

impl<T: Scalar> Wdf for ResistiveVoltageSource<T> {
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
        self.b = self.vs;
        self.b
    }

    fn reset(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<T: Scalar> AdaptedWdf for ResistiveVoltageSource<T> {
    fn impedance(&self) -> Self::Scalar {
        self.r
    }
}

impl<T: Scalar> ResistiveVoltageSource<T> {
    /// Create a new resistive voltage source node.
    ///
    /// # Arguments
    ///
    /// * `r`: Resistance value (Ohm)
    /// * `vs`: Voltage source value (V)
    ///
    /// returns: ResistiveVoltageSource<T>
    pub fn new(r: T, vs: T) -> Self {
        Self {
            vs,
            r,
            a: T::zero(),
            b: T::zero(),
        }
    }
}

/// Resistive current source leaf.
///
/// This node can be adapter thanks to the resistance running in parallel to the voltage source.
#[derive(Debug, Copy, Clone)]
pub struct ResistiveCurrentSource<T> {
    /// Current source value (A)
    pub j: T,
    /// Resistance value (Ohm)
    pub r: T,
    a: T,
    b: T,
}

impl<T: Zero> ResistiveCurrentSource<T> {
    /// Create a new resistive current source node.
    ///
    /// # Arguments
    ///
    /// * `r`: Resistance value (Ohm)
    /// * `j`: Current source value (A)
    ///
    /// returns: ResistiveCurrentSource<T>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn new(r: T, j: T) -> Self {
        Self {
            j,
            r,
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> Wdf for ResistiveCurrentSource<T> {
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
        self.b = T::from_f64(2.) * self.r * self.j;
        self.b
    }

    fn reset(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<T: Scalar> AdaptedWdf for ResistiveCurrentSource<T> {
    fn impedance(&self) -> Self::Scalar {
        self.r
    }
}

/// Resistor node.
#[derive(Debug, Copy, Clone)]
pub struct Resistor<T> {
    /// Resistance value (Ohm)
    pub r: T,
    a: T,
}

impl<T: Scalar> Wdf for Resistor<T> {
    type Scalar = T;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: T::zero(),
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        T::zero()
    }

    fn reset(&mut self) {
        self.a.set_zero();
    }
}

impl<T: Scalar> AdaptedWdf for Resistor<T> {
    fn impedance(&self) -> Self::Scalar {
        self.r
    }
}

impl<T: Scalar> Resistor<T> {
    /// Create a new resistor node.
    ///
    /// # Arguments
    ///
    /// * `r`: Resistance value
    ///
    /// returns: Resistor<T>
    pub fn new(r: T) -> Self {
        Self { r, a: T::zero() }
    }
}

/// Capacitor leaf node.
#[derive(Debug, Copy, Clone)]
pub struct Capacitor<T> {
    /// Sample rate (Hz)
    pub fs: T,
    /// Capacitance (F)
    pub c: T,
    a: T,
    b: T,
}

impl<T: Scalar> Capacitor<T> {
    /// Create a new capacitor leaf node.
    ///
    /// # Arguments
    ///
    /// * `fs`: Sample rate (Hz)
    /// * `c`: Capacitance (F)
    ///
    /// returns: Capacitor<T>
    pub fn new(fs: T, c: T) -> Self {
        Self {
            fs,
            c,
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> Wdf for Capacitor<T> {
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
        self.b = self.a;
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        self.fs = T::from_f64(samplerate);
    }

    fn reset(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<T: Scalar> AdaptedWdf for Capacitor<T> {
    fn admittance(&self) -> Self::Scalar {
        self.c * self.fs * T::from_f64(2.0)
    }
}
