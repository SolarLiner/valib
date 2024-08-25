use crate::wdf::{AdaptedWdf, Wave, Wdf};
use crate::Scalar;
use num_traits::Zero;

#[derive(Debug, Copy, Clone)]
pub struct ResistiveVoltageSource<T> {
    pub vs: T,
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
    pub fn new(r: T, vs: T) -> Self {
        Self {
            vs,
            r,
            a: T::zero(),
            b: T::zero(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Resistor<T> {
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
    pub fn new(r: T) -> Self {
        Self { r, a: T::zero() }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Capacitor<T> {
    pub fs: T,
    pub c: T,
    a: T,
    b: T,
}

impl<T: Scalar> Capacitor<T> {
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
