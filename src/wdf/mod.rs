#![cfg(feature = "unstable-wdf")]
use std::{
    any::Any,
    cell::{RefCell, RefMut},
    fmt,
    ops::{self, Neg},
    rc::Rc,
};

use num_traits::{One, Zero};
use numeric_literals::replace_float_literals;

use crate::Scalar;

#[derive(Debug, Copy, Clone)]
pub struct Wave<T> {
    pub a: T,
    pub b: T,
}

impl<T: Zero> Default for Wave<T> {
    fn default() -> Self {
        Self {
            a: T::zero(),
            b: T::zero(),
        }
    }
}

impl<T: Scalar> Wave<T> {
    pub fn from_kirchhoff(rp: T, v: T, i: T) -> Self {
        Self {
            a: v + rp * i,
            b: v - rp * i,
        }
    }
    pub fn a(a: T) -> Self {
        Self { a, b: T::zero() }
    }
    pub fn b(b: T) -> Self {
        Self { a: T::zero(), b }
    }

    #[replace_float_literals(T::from_f64(literal))]
    #[inline]
    pub fn voltage(&self) -> T {
        0.5 * (self.a + self.b)
    }

    #[replace_float_literals(T::from_f64(literal))]
    #[inline]
    pub fn current(&self, rp: T) -> T {
        (self.a - self.b) * (0.5 / rp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Impedance<T> {
    r: T,
    g: T,
}

impl<T: Scalar> Impedance<T> {
    pub fn nonadaptable() -> Self {
        Self {
            r: T::zero(),
            g: T::infinity(),
        }
    }

    pub fn from_resistance(r: T) -> Self {
        Self {
            r,
            g: r.simd_recip(),
        }
    }

    pub fn from_admittance(g: T) -> Self {
        Self {
            r: g.simd_recip(),
            g,
        }
    }

    pub fn set_resistance(&mut self, r: T) {
        self.r = r;
        self.g = r.simd_recip();
    }

    pub fn set_admittance(&mut self, g: T) {
        self.g = g;
        self.r = g.simd_recip();
    }

    pub fn is_adaptable(&self) -> bool {
        self.r.is_zero() || self.g.is_infinite()
    }

    #[inline]
    pub fn resistance(&self) -> T {
        self.r
    }

    #[inline]
    pub fn admittance(&self) -> T {
        self.g
    }

    pub fn parallel_with(self, other: Self) -> Self {
        Self::from_admittance(self.g + other.g)
        // Self {
        //     r: self.r * other.r / (self.r + other.r),
        //     g: self.g + other.g,
        // }
    }

    pub fn series_with(self, other: Self) -> Self {
        Self::from_resistance(self.r + other.r)
        // Self {
        //     r: self.r + other.r,
        //     g: self.g * other.g / (self.g + other.g),
        // }
    }
}

pub trait Wdf<T>: Any {
    /// Returns the matched impedance of the port. If it is unadaptable, the impedence should indicate 0 resistance (or +oo admittance).
    fn impedance(&self) -> Impedance<T>;

    /// Evaluate a wave through this port. The `p` parameter indicates the type of wave (1 for voltage waves, 1/2 for power waves and 0 for current waves).
    /// The `a`-value of the wave is passed in, and the `b` parameter should be returned.
    fn eval_wave(&mut self, rp: Impedance<T>, p: T, a: T) -> T;
}

pub struct Resistor<T>(pub T);

impl<T: 'static + Scalar> Wdf<T> for Resistor<T> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::from_resistance(self.0)
    }

    fn eval_wave(&mut self, _: Impedance<T>, _: T, _: T) -> T {
        T::zero()
    }
}

pub struct IdealVs<T>(pub T);

impl<T: 'static + Scalar> Wdf<T> for IdealVs<T> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn eval_wave(&mut self, rp: Impedance<T>, p: T, a: T) -> T {
        2. * rp.resistance().simd_powf(p - 1.) * self.0 - a
    }
}

pub struct IdealIs<T>(pub T);

impl<T: 'static + Scalar> Wdf<T> for IdealIs<T> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn eval_wave(&mut self, rp: Impedance<T>, p: T, a: T) -> T {
        2. * rp.resistance().simd_powf(p) + a
    }
}

pub struct ResistiveVs<T> {
    pub r: T,
    pub vs: T,
}

impl<T: 'static + Scalar> Wdf<T> for ResistiveVs<T> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::from_resistance(self.r)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn eval_wave(&mut self, port_impedance: Impedance<T>, p: T, _: T) -> T {
        self.vs * port_impedance.resistance().simd_powf(p - 1.)
    }
}

pub struct ResistiveIs<T> {
    pub r: T,
    pub is: T,
}

impl<T: 'static + Scalar> Wdf<T> for ResistiveIs<T> {
    fn impedance(&self) -> Impedance<T> {
        Impedance::from_resistance(self.r)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn eval_wave(&mut self, port_impedance: Impedance<T>, p: T, _: T) -> T {
        self.is * port_impedance.resistance().simd_powf(p)
    }
}

pub struct ShortCircuit;

impl<T: 'static + Scalar> Wdf<T> for ShortCircuit {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    fn eval_wave(&mut self, _: Impedance<T>, p: T, a: T) -> T {
        a.neg()
    }
}

pub struct OpenCircuit;

impl<T: 'static + Scalar> Wdf<T> for OpenCircuit {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    fn eval_wave(&mut self, _: Impedance<T>, p: T, a: T) -> T {
        a
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchState {
    Closed,
    Open,
}

impl SwitchState {
    #[inline(always)]
    pub fn lambda<T: One + Neg<Output = T>>(&self) -> T {
        match self {
            Self::Open => T::one(),
            Self::Closed => T::one().neg(),
        }
    }
}

pub struct Switch(pub SwitchState);

impl<T: 'static + Scalar> Wdf<T> for Switch {
    fn impedance(&self) -> Impedance<T> {
        Impedance::nonadaptable()
    }

    fn eval_wave(&mut self, _: Impedance<T>, p: T, a: T) -> T {
        a * self.0.lambda()
    }
}

pub struct Capacitor<T> {
    pub fs: T,
    pub c: T,
    state: T,
}

impl<T: Scalar> Default for Capacitor<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn default() -> Self {
        Self {
            fs: 44.1e3,
            c: 0.,
            state: 0.,
        }
    }
}

impl<T: 'static + Scalar> Wdf<T> for Capacitor<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn impedance(&self) -> Impedance<T> {
        Impedance::from_admittance(2. * self.fs * self.c)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn eval_wave(&mut self, port_impedance: Impedance<T>, p: T, a: T) -> T {
        std::mem::replace(&mut self.state, a)
    }
}

pub struct Inductor<T> {
    pub fs: T,
    pub l: T,
    state: T,
}

impl<T: Scalar> Default for Inductor<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn default() -> Self {
        Self {
            fs: 44.1e3,
            l: 0.,
            state: 0.,
        }
    }
}

impl<T: 'static + Scalar> Wdf<T> for Inductor<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn impedance(&self) -> Impedance<T> {
        Impedance::from_resistance(2. * self.fs * self.l)
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn eval_wave(&mut self, _: Impedance<T>, p: T, a: T) -> T {
        std::mem::replace(&mut self.state, -a)
    }
}

// Two-ports
pub struct Parallel<L, R>(pub L, pub R);

impl<T: 'static + Scalar, L: Wdf<T>, R: Wdf<R>> Wdf<T> for Parallel<L, R> {
    fn impedance(&self) -> Impedance<T> {
        todo!()
    }

    fn eval_wave(&mut self, rp: Impedance<T>, p: T, a: T) -> T {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::wdf::{Resistor, Wave};

    use super::{Capacitor, ResistiveVs, Wdf};

    #[test]
    fn lpf_1pole() {
        let mut rs = ResistiveVs { r: 330., vs: 1. };
        let mut r = Resistor(330.);
        let b_rs = rs.eval_wave(r.impedance(), 1., 0.);
        let b_c = r.eval_wave(rs.impedance(), 1., b_rs);
        let out_wave = Wave { a: b_rs, b: b_c };
        println!(
            "b_rs: {b_rs}\tb_c: {b_c}\tout voltage: {}",
            out_wave.voltage()
        );
    }
}
