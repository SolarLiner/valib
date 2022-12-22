use crate::{saturators::Saturator, Scalar, DSP};
use numeric_literals::replace_float_literals;
use std::marker::PhantomData;

#[derive(Debug, Copy, Clone)]
pub struct Ladder<T, S, const N: usize> {
    g: T,
    s: [T; N],
    k: T,
    __saturator: PhantomData<S>,
}

impl<T: Scalar, S: Saturator<T>, const N: usize> DSP<1, 1> for Ladder<T, S, N> {
    type Sample = T;

    #[inline(always)]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        todo!()
    }
}
