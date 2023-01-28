use num_traits::Zero;
use numeric_literals::replace_float_literals;

use crate::clippers::DiodeClipperModel;
use crate::wdf::{Wave, Wdf};
use crate::Scalar;

use super::{Impedance, IntoNode, Node};

#[derive(Debug, Copy, Clone)]
pub struct DiodePair<T> {
    pub model: DiodeClipperModel<T>,
}

impl<T: Scalar> From<DiodeClipperModel<T>> for DiodePair<T> {
    fn from(model: DiodeClipperModel<T>) -> Self {
        Self { model }
    }
}

impl<T: 'static + Scalar> Wdf<T> for DiodePair<T> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b
    }

    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
        wave.b = a - self.model.eval(wave.voltage());
    }
}

#[derive(Debug, Clone)]
pub struct IdealVs<T, W> {
    pub vs: T,
    child: Node<T, W>,
}

impl<T: 'static + Scalar, W: Wdf<T>> Wdf<T> for IdealVs<T, W> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    #[inline]
    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b
    }

    #[inline]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
        wave.b = 2. * self.vs - a;
    }
}

impl<T: 'static + Scalar, W: Wdf<T>> IdealVs<T, W> {
    pub fn new(child: impl Into<Node<T, W>>) -> Node<T, Self> {
        let mut child = child.into();
        let this = Self {
            vs: T::zero(),
            child: child.clone(),
        }
        .into_node();
        child.set_parent(&this);
        this
    }
}
