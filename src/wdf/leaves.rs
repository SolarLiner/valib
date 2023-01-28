use crate::wdf::{Wave, Wdf};
use crate::Scalar;
use num_traits::Zero;
use numeric_literals::replace_float_literals;

use super::Impedance;

#[derive(Debug, Copy, Clone)]
pub struct Resistor<T>(pub Impedance<T>);

impl<T: 'static + Scalar> Wdf<T> for Resistor<T> {
    #[inline]
    fn impedance(&self) -> Impedance<T> {
        self.0
    }

    #[inline]
    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b.set_zero();
        wave.b
    }

    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Capacitor<T> {
    pub fs: T,
    pub c: T,
}

impl<T: 'static + Scalar> Wdf<T> for Capacitor<T> {
    #[replace_float_literals(T::from(literal).unwrap())]
    #[inline]
    fn impedance(&self) -> Impedance<T> {
        Impedance::from_admittance(2. * self.fs * self.c)
    }

    #[inline]
    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b = std::mem::replace(&mut wave.a, T::zero());
        wave.b
    }

    #[inline]
    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct ShortCircuit;

impl<T: Scalar> Wdf<T> for ShortCircuit {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    #[inline]
    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b
    }

    #[inline]
    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
        wave.b = -a;
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct OpenCircuit;

impl<T: Scalar> Wdf<T> for OpenCircuit {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    #[inline]
    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.a
    }

    #[inline]
    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
        wave.b = a;
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Switch(pub bool);

impl<T: Scalar> Wdf<T> for Switch {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b
    }

    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
        wave.b = if self.0 { -a } else { a };
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ResistiveVs<T> {
    pub r: Impedance<T>,
    pub vs: T,
}

impl<T: 'static + Scalar> Wdf<T> for ResistiveVs<T> {
    fn impedance(&self) -> Impedance<T> {
        self.r
    }

    #[inline]
    fn reflected(&mut self, _: Impedance<T>, wave: &mut Wave<T>) -> T {
        wave.b = self.vs;
        wave.b
    }

    fn incident(&mut self, _: Impedance<T>, wave: &mut Wave<T>, a: T) {
        wave.a = a;
    }
}
