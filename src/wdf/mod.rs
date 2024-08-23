use crate::Scalar;
use num_traits::Zero;
use simba::simd::SimdComplexField;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

pub mod diode;

pub type Node<T> = Rc<RefCell<T>>;

pub type NodeRef<'a, T> = Ref<'a, T>;

pub type NodeMut<'a, T> = RefMut<'a, T>;

#[inline]
pub fn node<T>(value: T) -> Node<T> {
    Rc::new(RefCell::new(value))
}

#[inline]
pub fn voltage<W: Wdf>(node: &Node<W>) -> W::Scalar {
    node.borrow().wave().voltage()
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
}

#[allow(unused)]
pub trait Wdf {
    type Scalar: Scalar;
    fn wave(&self) -> Wave<Self::Scalar>;
    fn incident(&mut self, x: Self::Scalar);
    fn reflected(&mut self) -> Self::Scalar;
    fn set_samplerate(&mut self, samplerate: f64) {}
    fn set_port_resistance(&mut self, resistance: Self::Scalar) {}
    fn reset(&mut self) {}
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

    fn set_samplerate(&mut self, _: f64) {}
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

pub struct WdfModule<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> {
    pub root: Node<Root>,
    pub leaf: Node<Leaf>,
}

impl<Root: Wdf, Leaf: AdaptedWdf<Scalar = Root::Scalar>> WdfModule<Root, Leaf> {
    pub fn new(root: Node<Root>, leaf: Node<Leaf>) -> Self {
        Self { root, leaf }
    }

    pub fn set_samplerate(&mut self, samplerate: f64) {
        let mut root = self.root.borrow_mut();
        let mut leaf = self.leaf.borrow_mut();
        root.set_samplerate(samplerate);
        leaf.set_samplerate(samplerate);
    }

    pub fn next_sample(&mut self) {
        let mut root = self.root.borrow_mut();
        let mut leaf = self.leaf.borrow_mut();
        root.set_port_resistance(leaf.impedance());
        root.incident(leaf.reflected());
        leaf.incident(root.reflected());
    }
}

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

    fn set_samplerate(&mut self, _: f64) {}
}

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

    fn set_samplerate(&mut self, _: f64) {}
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

    fn set_samplerate(&mut self, _: f64) {}
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
    z: T,
    b: T,
}

impl<T: Scalar> Capacitor<T> {
    fn new(fs: T, c: T) -> Self {
        Self {
            fs,
            c,
            z: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> Wdf for Capacitor<T> {
    type Scalar = T;

    fn wave(&self) -> Wave<Self::Scalar> {
        Wave {
            a: self.z,
            b: self.b,
        }
    }

    fn incident(&mut self, x: Self::Scalar) {
        self.z = x;
    }

    fn reflected(&mut self) -> Self::Scalar {
        self.b = self.z;
        self.b
    }

    fn set_samplerate(&mut self, samplerate: f64) {
        self.fs = T::from_f64(samplerate);
    }

    fn reset(&mut self) {
        self.z.set_zero();
    }
}

impl<T: Scalar> AdaptedWdf for Capacitor<T> {
    fn admittance(&self) -> Self::Scalar {
        self.c * self.fs * T::from_f64(2.0)
    }
}

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
}

impl<A: AdaptedWdf> AdaptedWdf for Inverter<A> {
    fn impedance(&self) -> Self::Scalar {
        self.inner.borrow().impedance()
    }

    fn admittance(&self) -> Self::Scalar {
        self.inner.borrow().admittance()
    }
}

#[cfg(test)]
mod tests {
    use crate::wdf::{node, voltage, IdealVoltageSource, Inverter, Resistor, Series, WdfModule};
    use crate::Scalar;

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
}
