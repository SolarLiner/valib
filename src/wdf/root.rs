use num_traits::Zero;
use numeric_literals::replace_float_literals;

use crate::{
    clippers::DiodeClipperModel,
    wdf::{Wave, Wdf},
    Scalar,
};

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
pub struct IdealVs<T>(pub T);

impl<T: 'static + Scalar> Wdf<T> for IdealVs<T> {
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
        wave.b = 2. * self.0 - a;
    }
}

impl<T: 'static + Scalar> IdealVs<T> {
    pub fn new<W>(child: &mut Node<T, W>) -> Node<T, Self> {
        let this = Self(T::zero()).into_node();
        child.set_parent(&this);
        this
    }
}
