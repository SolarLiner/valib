use crate::dsp::{DSPMeta, DSPProcess};
use crate::wdf::{Wave, Wdf};
use crate::Scalar;
use num_traits::Zero;

#[derive(Debug, Copy, Clone)]
pub struct IdealVoltageSource<T> {
    pub vs: T,
    a: T,
    b: T,
}

impl<T: Zero> IdealVoltageSource<T> {
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

#[derive(Debug, Copy, Clone)]
pub struct WdfDsp<P: DSPMeta> {
    pub dsp: P,
    a: P::Sample,
    b: P::Sample,
}

impl<P: DSPMeta<Sample: Zero>> WdfDsp<P> {
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
