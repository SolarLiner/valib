use nalgebra::{Complex, zero};
use simba::simd::SimdComplexField;
use numeric_literals::replace_float_literals;
use simba::scalar::RealField;
use crate::dsp::DSP;
use crate::Scalar;
use crate::dsp::analog::DspAnalog;

pub trait DspAnalysis<const I: usize, const O: usize>: DSP<I, O> {
    fn h_z(&self, z: [Complex<Self::Sample>; I]) -> [Complex<Self::Sample>; O];
    fn freq_response(&self, jw: [Self::Sample; I]) -> [Complex<Self::Sample>; O]
        where
            Self::Sample: RealField,
            Complex<Self::Sample>: SimdComplexField<SimdRealField=Self::Sample>
    {
        let z = jw.map(Complex::from_simd_real).map(|jw| <Complex<Self::Sample> as SimdComplexField>::simd_exp(Complex::i() * jw));
        self.h_z(z)
    }
}

impl<const I: usize, const O: usize, D: DspAnalog<I, O>> DspAnalysis<I, O> for D where Complex<D::Sample>: SimdComplexField {
    #[replace_float_literals(Complex::from(D::Sample::from_f64(literal)))]
    #[inline(always)]
    fn h_z(&self, z: [Complex<D::Sample>; I]) -> [Complex<D::Sample>; O] {
        self.h_s(z.map(|z| 2. * (z - 1.) / (z + 1.)))
    }

    #[inline(always)]
    fn freq_response(&self, jw: [D::Sample; I]) -> [Complex<D::Sample>; O]
        where
            Self::Sample: RealField,
    {
        self.h_s(jw.map(|jw| Complex::new(D::Sample::from_f64(0.0), jw)))
    }
}
