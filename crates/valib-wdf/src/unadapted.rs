//! # WDF unadapted nodes
//!
//! Provides nodes which cannot be adapted anywhere in the tree, and have to sit at the root.
use crate::{Wave, Wdf};
use num_traits::Zero;
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::simd::SimdBool;
use valib_core::Scalar;

/// Ideal voltage source WDF node.
#[derive(Debug, Copy, Clone)]
pub struct IdealVoltageSource<T> {
    /// Voltage source value (V)
    pub vs: T,
    a: T,
    b: T,
}

impl<T: Zero> IdealVoltageSource<T> {
    /// Create a new ideal voltage source node.
    ///
    /// # Arguments
    ///
    /// * `vs`: Voltage source value (V)
    ///
    /// returns: IdealVoltageSource<T>
    pub fn new(vs: T) -> Self {
        Self {
            vs,
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> Wdf for IdealVoltageSource<T> {
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
        self.b = -self.a + self.vs * T::from_f64(2.0);
        self.b
    }

    fn reset(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
    }
}

/// Ideal current source WDF node.
#[derive(Debug, Copy, Clone)]
pub struct IdealCurrentSource<T> {
    /// Current source value (A)
    pub j: T,
    r: T,
    a: T,
    b: T,
}

impl<T: Zero> IdealCurrentSource<T> {
    /// Create a new ideal current source node.
    ///
    /// # Arguments
    ///
    /// * `j`: Current source value (A)
    ///
    /// returns: IdealCurrentSource<T>
    pub fn new(j: T) -> Self {
        Self {
            j,
            r: T::zero(),
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Zero> Default for IdealCurrentSource<T> {
    fn default() -> Self {
        Self::new(T::zero())
    }
}

impl<T: Scalar> Wdf for IdealCurrentSource<T> {
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
        self.b = T::from_f64(2.) * self.r * self.j + self.a;
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

/// Short circuit WDF node.
#[derive(Debug, Copy, Clone)]
pub struct ShortCircuit<T> {
    a: T,
}

impl<T: Zero> Default for ShortCircuit<T> {
    fn default() -> Self {
        Self { a: T::zero() }
    }
}

impl<T: Scalar> Wdf for ShortCircuit<T> {
    type Scalar = T;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: -self.a,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        -self.a
    }

    fn reset(&mut self) {
        self.a.set_zero();
    }
}

/// Open circuit WDF node.
#[derive(Debug, Copy, Clone)]
pub struct OpenCircuit<T> {
    a: T,
}

impl<T: Zero> Default for OpenCircuit<T> {
    fn default() -> Self {
        Self { a: T::zero() }
    }
}

impl<T: Scalar> Wdf for OpenCircuit<T> {
    type Scalar = T;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: self.a,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        self.a
    }

    fn reset(&mut self) {
        self.a.set_zero();
    }
}

/// Switch WDF node.
pub struct Switch<T: Scalar> {
    /// State of the switch
    pub closed: T::SimdBool,
    a: T,
    b: T,
}

impl<T: Scalar> Wdf for Switch<T> {
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
        let k = self.closed.if_else(|| -T::one(), || T::one());
        self.b = k * self.a;
        self.b
    }

    fn reset(&mut self) {
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<T: Scalar> Switch<T> {
    /// Create a new switch node.
    ///
    /// # Arguments
    ///
    /// * `closed`: When true, the switch starts closed.
    ///
    /// returns: Switch<T>
    pub fn new(closed: T::SimdBool) -> Self {
        Self {
            closed,
            a: T::zero(),
            b: T::zero(),
        }
    }
}

/// Arbitrary DSP WDF node.
///
/// Arbitrary filters can be run as an unadapted WDF node with the following equation:
///
/// $$ b = 2*f(a) - a $$
///
/// Where $f$ is the function which is run on the incident wave.
#[derive(Debug, Copy, Clone)]
pub struct WdfDsp<P: DSPMeta> {
    /// Inner DSP process
    pub dsp: P,
    a: P::Sample,
    b: P::Sample,
}

impl<P: DSPMeta<Sample: Zero>> WdfDsp<P> {
    /// Create a new DSP WDF node.
    ///
    /// # Arguments
    ///
    /// * `dsp`: Inner DSP process to run. Must be a `DSPProcess<1, 1>`.
    ///
    /// returns: WdfDsp<P>
    pub fn new(dsp: P) -> Self {
        Self {
            dsp,
            a: P::Sample::zero(),
            b: P::Sample::zero(),
        }
    }
}

impl<P: DSPProcess<1, 1>> Wdf for WdfDsp<P> {
    type Scalar = P::Sample;

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
        let [y] = self.dsp.process([self.a]);
        self.b = P::Sample::from_f64(2.0) * y - self.a;
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        self.dsp.set_samplerate(samplerate as _);
    }

    fn reset(&mut self) {
        self.dsp.reset();
        self.a.set_zero();
        self.b.set_zero();
    }
}
