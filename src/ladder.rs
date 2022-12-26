//! Implementation of various blocks of DSP code from the VA Filter Design book.
//! Downloaded from https://www.discodsp.net/VAFilterDesign_2.1.2.pdf
//! All references in this module, unless specified otherwise, are taken from this book.

use crate::{saturators::Saturator, Scalar, DSP};
use nalgebra::SVector;
use numeric_literals::replace_float_literals;
use std::{fmt, marker::PhantomData};

#[derive(Debug, Copy, Clone)]
pub struct Ladder<T, S, const N: usize> {
    g: T,
    s: SVector<T, N>,
    k: T,
    __saturator: PhantomData<S>,
}

impl<T: Scalar + fmt::Debug, S: Saturator<T>, const N: usize> DSP<1, 1> for Ladder<T, S, N> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let x = x[0];
        let y0 = x - self.k * self.s[3];
        let yd = SVector::from_column_slice(&[
            y0 - self.s[0],
            self.s[0] - self.s[1],
            self.s[1] - self.s[2],
            self.s[2] - self.s[3],
        ]);
        let y = yd.map(S::saturate) * self.g + self.s;
        self.s = y;
        [y[3]]
    }
}
