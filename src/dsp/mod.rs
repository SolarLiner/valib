use crate::Scalar;

pub mod analysis;
pub mod analog;
pub mod blocks;

pub trait DSP<const I: usize, const O: usize> {
    type Sample: Scalar;

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O];
}

impl<P: DSP<N, N>, const A: usize, const N: usize> DSP<N, N> for [P; A] {
    type Sample = P::Sample;

    #[inline(always)]
    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        self.iter_mut().fold(x, |x, f| f.process(x))
    }
}
