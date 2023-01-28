use crate::wdf::{Wave, Wdf};
use crate::Scalar;

use super::{Impedance, IntoNode, Node};

#[derive(Debug, Clone)]
pub struct Parallel<T, L, R> {
    pub left: Node<T, L>,
    pub right: Node<T, R>,
    btemp: T,
    bdiff: T,
}

impl<T: 'static + Scalar + 'static, L: Wdf<T>, R: Wdf<T>> Wdf<T> for Parallel<T, L, R> {
    fn impedance(&self) -> super::Impedance<T> {
        self.left.impedance().parallel_with(self.right.impedance())
    }

    fn reflected(&mut self, p1_reflect: Impedance<T>, wave: &mut Wave<T>) -> T {
        let rl = self.left.reflected();
        let rr = self.right.reflected();

        self.bdiff = rr - rl;
        self.btemp = -p1_reflect.resistance() * self.bdiff;
        wave.b = rr + self.btemp;
        wave.b
    }

    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        let b2 = a + self.btemp;
        self.left.incident(self.bdiff + b2);
        self.right.incident(b2);
        wave.a = a;
    }
}

impl<T: 'static + Scalar, L: Wdf<T>, R: Wdf<T>> Parallel<T, L, R> {
    pub fn new(left: impl Into<Node<T, L>>, right: impl Into<Node<T, R>>) -> Node<T, Self> {
        let mut left = left.into();
        let mut right = right.into();
        let this = Self {
            left: left.clone(),
            right: right.clone(),
            btemp: T::zero(),
            bdiff: T::zero(),
        }
        .into_node();
        left.set_parent(&this);
        right.set_parent(&this);
        this
    }
}

#[derive(Debug, Clone)]
pub struct Series<T, L, R> {
    pub left: Node<T, L>,
    pub right: Node<T, R>,
}

impl<T: 'static + Scalar, L: Wdf<T>, R: Wdf<T>> Wdf<T> for Series<T, L, R> {
    fn impedance(&self) -> Impedance<T> {
        self.left.impedance().series_with(self.right.impedance())
    }

    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        let rl = self.left.reflected();
        let rr = self.right.reflected();

        wave.b = -rl - rr;
        wave.b
    }

    fn incident(&mut self, p1_reflect: Impedance<T>, wave: &mut Wave<T>, a: T) {
        let lb = self.left.wave().b;
        let rb = self.right.wave().b;
        let b1 = lb - p1_reflect.resistance() * (a + lb + rb);
        self.left.incident(b1);
        self.right.incident(-a - b1);
        wave.a = a;
    }
}

impl<T: 'static + Scalar, L: Wdf<T>, R: Wdf<T>> Series<T, L, R> {
    pub fn new(left: impl Into<Node<T, L>>, right: impl Into<Node<T, R>>) -> Node<T, Self> {
        let mut left = left.into();
        let mut right = right.into();
        let this = Self {
            left: left.clone(),
            right: right.clone(),
        }
        .into_node();
        left.set_parent(&this);
        right.set_parent(&this);
        this
    }
}

#[derive(Debug, Clone)]
pub struct PolarityInvert<T, W> {
    pub child: Node<T, W>,
}

impl<T: 'static + Scalar, W: Wdf<T>> Wdf<T> for PolarityInvert<T, W> {
    fn impedance(&self) -> Impedance<T> {
        self.child.impedance()
    }

    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b = -self.child.reflected();
        wave.b
    }

    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
        self.child.incident(-a);
    }
}

impl<T: 'static + Scalar, W: Wdf<T>> PolarityInvert<T, W> {
    pub fn new(child: impl Into<Node<T, W>>) -> Node<T, Self> {
        let mut child = child.into();
        let this = Self { child: child.clone() }.into_node();
        child.set_parent(&this);
        this
    }
}