use nalgebra::Complex;
use simba::simd::SimdComplexField;

use crate::{dsp::DSP, math::freq_to_z};

/// Trait for DSP structs that have a z-domain transfer function available.
/// For processes with nonlinear methods, the transfer function can still be defined by
/// linearizing the process, e.g. in filters, nonlinearities can be removed.
///
/// The goal of this trait is to provide an easy way to compute frequency responses of
/// filters for end-user visual feedback and not to be scientifically accurate.
pub trait DspAnalysis<const I: usize, const O: usize>: DSP<I, O> {
    /// Discrete transfer function in the z-domain.
    fn h_z(
        &self,
        samplerate: Self::Sample,
        z: Complex<Self::Sample>,
    ) -> [[Complex<Self::Sample>; O]; I];

    /// Frequency response of the filter, using the complex exponential transformation to
    /// translate the input angular velocity into its z-domain position to pass into `h_z`.
    ///
    /// This is provided in the trait to allow overrides where the frequency representation
    /// is faster to compute than the full z-domain transfer function.
    fn freq_response(
        &self,
        samplerate: Self::Sample,
        f: Self::Sample,
    ) -> [[Complex<Self::Sample>; O]; I]
    where
        Complex<Self::Sample>: SimdComplexField,
    {
        self.h_z(samplerate, freq_to_z(samplerate, f))
    }
}
