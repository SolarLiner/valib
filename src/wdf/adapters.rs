use crate::wdf::{AdaptedWdf, Node, Wave, Wdf};
use num_traits::Zero;

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

        let mut left = self.left.borrow_mut();
        let mut right = self.right.borrow_mut();
        let p1z = left.impedance() / z;
        let w1 = left.wave();
        let w2 = right.wave();
        let b1 = w1.b - p1z * (x + w1.b + w2.b);
        left.incident(b1);
        right.incident(-x - b1);
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        let mut left = self.left.borrow_mut();
        let mut right = self.right.borrow_mut();
        self.b = -left.reflected() - right.reflected();
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        self.left.borrow_mut().set_samplerate(samplerate);
        self.right.borrow_mut().set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        self.left.borrow_mut().reset();
        self.right.borrow_mut().reset();
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> AdaptedWdf for Series<A, B> {
    fn impedance(&self) -> Self::Scalar {
        let left = self.left.borrow();
        let right = self.right.borrow();
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
        let mut left = self.left.borrow_mut();
        let mut right = self.right.borrow_mut();
        let b2 = x + self.btemp;
        left.incident(self.bdiff + b2);
        right.incident(b2);
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        let z = self.impedance(); // Borrows self.left & self.right

        let mut left = self.left.borrow_mut();
        let mut right = self.right.borrow_mut();
        let p1z = (left.impedance() / z).simd_recip();
        let b1 = left.reflected();
        let b2 = right.reflected();
        self.bdiff = b2 - b1;
        self.btemp = -p1z * self.bdiff;
        self.b = b2 + self.btemp;
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        self.left.borrow_mut().set_samplerate(samplerate);
        self.right.borrow_mut().set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        self.left.borrow_mut().reset();
        self.right.borrow_mut().reset();
        self.a.set_zero();
        self.b.set_zero();
        self.btemp.set_zero();
        self.bdiff.set_zero();
    }
}

impl<A: AdaptedWdf, B: AdaptedWdf<Scalar = A::Scalar>> AdaptedWdf for Parallel<A, B> {
    fn admittance(&self) -> Self::Scalar {
        let left = self.left.borrow();
        let right = self.right.borrow();
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
        self.inner.borrow_mut().incident(-x);
        self.a = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        self.b = -self.inner.borrow_mut().reflected();
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        self.inner.borrow_mut().set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        self.inner.borrow_mut().reset();
        self.a.set_zero();
        self.b.set_zero();
    }
}

impl<A: AdaptedWdf> AdaptedWdf for Inverter<A> {
    fn impedance(&self) -> Self::Scalar {
        self.inner.borrow().impedance()
    }

    fn admittance(&self) -> Self::Scalar {
        self.inner.borrow().admittance()
    }
}
