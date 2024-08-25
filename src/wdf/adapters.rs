use crate::wdf::dsl::{node_mut, node_ref};
use crate::wdf::{AdaptedWdf, Node, Wave, Wdf};
use num_traits::Zero;
use simba::simd::SimdComplexField;

#[derive(Debug, Clone)]
pub struct Series<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> {
    pub left: Node<A>,
    pub right: Node<B>,
    a: A::Scalar,
    b: A::Scalar,
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> Series<A, B> {
    pub fn new(left: Node<A>, right: Node<B>) -> Self {
        Self {
            left,
            right,
            a: A::Scalar::zero(),
            b: A::Scalar::zero(),
        }
    }
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> Series<A, B> {}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> Wdf for Series<A, B> {
    type Scalar = A::Scalar;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: self.b,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        let z = self.impedance(); // Borrows self.left & self.right

        let mut left = node_mut(&self.left);
        let mut right = node_mut(&self.right);
        let p1z = left.impedance() / z;
        let w1 = left.wave();
        let w2 = right.wave();
        let b1 = w1.b - p1z * (x + w1.b + w2.b);
        left.incident(b1);
        right.incident(-x - b1);
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        let mut left = node_mut(&self.left);
        let mut right = node_mut(&self.right);
        self.b = -left.reflected() - right.reflected();
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        node_mut(&self.left).set_samplerate(samplerate);
        node_mut(&self.right).set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        node_mut(&self.left).reset();
        node_mut(&self.right).reset();
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> AdaptedWdf for Series<A, B> {
    fn impedance(&self) -> Self::Scalar {
        let left = node_ref(&self.left);
        let right = node_ref(&self.right);
        left.impedance() + right.impedance()
    }
}

pub struct Parallel<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> {
    pub left: Node<A>,
    pub right: Node<B>,
    a: A::Scalar,
    b: A::Scalar,
    bdiff: A::Scalar,
    btemp: A::Scalar,
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> Parallel<A, B> {
    pub fn new(left: Node<A>, right: Node<B>) -> Self {
        Self {
            left,
            right,
            a: A::Scalar::zero(),
            b: A::Scalar::zero(),
            bdiff: A::Scalar::zero(),
            btemp: A::Scalar::zero(),
        }
    }
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> Wdf for Parallel<A, B> {
    type Scalar = A::Scalar;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: self.b,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        let mut left = node_mut(&self.left);
        let mut right = node_mut(&self.right);
        let b2 = x + self.btemp;
        left.incident(self.bdiff + b2);
        right.incident(b2);
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        let z = self.impedance(); // Borrows self.left & self.right

        let mut left = node_mut(&self.left);
        let mut right = node_mut(&self.right);
        let p1z = (left.impedance() / z).simd_recip();
        let b1 = left.reflected();
        let b2 = right.reflected();
        self.bdiff = b2 - b1;
        self.btemp = -p1z * self.bdiff;
        self.b = b2 + self.btemp;
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        node_mut(&self.left).set_samplerate(samplerate);
        node_mut(&self.right).set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        node_mut(&self.left).reset();
        node_mut(&self.right).reset();
        self.a.set_zero();
        self.b.set_zero();
        self.btemp.set_zero();
        self.bdiff.set_zero();
    }
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> AdaptedWdf for Parallel<A, B> {
    fn admittance(&self) -> Self::Scalar {
        let left = node_ref(&self.left);
        let right = node_ref(&self.right);
        left.admittance() + right.admittance()
    }
}

pub struct Inverter<A: AdaptedWdf> {
    pub inner: Node<A>,
    a: A::Scalar,
    b: A::Scalar,
}

impl<A: AdaptedWdf> Inverter<A> {
    pub fn new(inner: Node<A>) -> Self {
        Self {
            inner,
            a: A::Scalar::zero(),
            b: A::Scalar::zero(),
        }
    }
}

impl<A: AdaptedWdf> Wdf for Inverter<A> {
    type Scalar = A::Scalar;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.a,
            b: self.b,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        node_mut(&self.inner).incident(-x);
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        self.b = -node_mut(&self.inner).reflected();
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        node_mut(&self.inner).set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        node_mut(&self.inner).reset();
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<A: AdaptedWdf> AdaptedWdf for Inverter<A> {
    fn impedance(&self) -> Self::Scalar {
        node_ref(&self.inner).impedance()
    }

    fn admittance(&self) -> Self::Scalar {
        node_ref(&self.inner).admittance()
    }
}
