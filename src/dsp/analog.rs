use nalgebra::Complex;
use crate::dsp::DSP;

pub trait DspAnalog<const I: usize, const O: usize>: DSP<I, O> {
    fn h_s(&self, s: [Complex<Self::Sample>; I]) -> [Complex<Self::Sample>; O];
}
