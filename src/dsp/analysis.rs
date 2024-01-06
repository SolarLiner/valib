use nalgebra::{Complex, ComplexField};
use simba::simd::SimdComplexField;
use simba::scalar::RealField;
use crate::dsp::DSP;

/// Trait for DSP structs that have a z-domain transfer function available.
/// For processes with nonlinear methods, the transfer function can still be defined by
/// linearizing the process, e.g. in filters, nonlinearities can be removed.
///
/// The goal of this trait is to provide an easy way to compute frequency responses of
/// filters for end-user visual feedback and not to be scientifically accurate.
pub trait DspAnalysis<const I: usize, const O: usize>: DSP<I, O> {
    /// Discrete transfer function in the z-domain.
    fn h_z(&self, samplerate: Self::Sample, z: Complex<Self::Sample>) -> [Complex<Self::Sample>; O];

    /// Frequency response of the filter, using the complex exponential transformation to
    /// translate the input angular velocity into its z-domain position to pass into `h_z`.
    ///
    /// This is provided in the trait to allow overrides where the frequency representation
    /// is faster to compute than the full z-domain transfer function.
    fn freq_response(&self, samplerate: Self::Sample, jw: Self::Sample) -> [Complex<Self::Sample>; O]
        where
            Self::Sample: RealField,
    {
        let jw = Complex::from_real(jw);
        let z = Complex::simd_exp(Complex::<Self::Sample>::i() * jw * Self::Sample::pi() / samplerate);
        self.h_z(samplerate, z)
    }
}
