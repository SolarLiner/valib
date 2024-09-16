#![warn(missing_docs)]
//! # Oscillators
//!
//! This module provides oscillators for `valib`.
use numeric_literals::replace_float_literals;
use valib_core::dsp::DSPMeta;
use valib_core::dsp::DSPProcess;
use valib_core::Scalar;

pub mod blit;
pub mod polyblep;
pub mod wavetable;

/// Tracks normalized phase for a given frequency. Phase is smooth even when frequency changes, so
/// it is suitable for driving oscillators.
#[derive(Debug, Clone, Copy)]
pub struct Phasor<T> {
    samplerate: T,
    frequency: T,
    phase: T,
    step: T,
}

impl<T: Scalar> DSPMeta for Phasor<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = T::from_f64(samplerate as _);
    }

    fn reset(&mut self) {
        self.phase = T::zero();
    }
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<0, 1> for Phasor<T> {
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let p = self.phase;
        let new_phase = self.phase + self.step;
        let gt = new_phase.simd_ge(T::one());
        self.phase = (new_phase - T::one()).select(gt, new_phase);
        [p]
    }
}

impl<T: Scalar> Phasor<T> {
    /// Create a new phasor.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Sample rate the phasor will run at
    /// * `freq`: Frequency of the phasor.
    ///
    /// returns: Phasor<T>
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: T, frequency: T) -> Self {
        Self {
            samplerate,
            frequency,
            phase: 0.0,
            step: frequency / samplerate,
        }
    }

    pub fn phase(&self) -> T {
        self.phase
    }

    pub fn set_phase(&mut self, phase: T) {
        self.phase = phase.simd_fract();
    }

    pub fn with_phase(mut self, phase: T) -> Self {
        self.set_phase(phase);
        self
    }

    pub fn next_sample_resets(&self) -> T::SimdBool {
        (self.phase + self.step).simd_ge(T::one())
    }

    /// Sets the frequency of this phasor. Phase is not reset, which means the phase remains
    /// continuous.
    /// # Arguments
    ///
    /// * `samplerate`: New sample rate
    /// * `freq`: New frequency
    ///
    /// returns: ()
    pub fn set_frequency(&mut self, freq: T) {
        self.step = freq / self.samplerate;
    }
}
