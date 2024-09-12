use numeric_literals::replace_float_literals;
use valib_core::dsp::DSPMeta;
use valib_core::dsp::DSPProcess;
use valib_core::Scalar;

pub mod blit;
pub mod wavetable;

#[derive(Debug, Clone, Copy)]
pub struct Phasor<T> {
    phase: T,
    step: T,
}

impl<T: Scalar> DSPMeta for Phasor<T> {
    type Sample = T;
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<0, 1> for Phasor<T> {
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let p = self.phase;
        let new_phase = self.phase + self.step;
        let gt = new_phase.simd_gt(T::one());
        self.phase = (new_phase - T::one()).select(gt, new_phase);
        [p]
    }
}

impl<T: Scalar> Phasor<T> {
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: T, freq: T) -> Self {
        Self {
            phase: 0.0,
            step: freq / samplerate,
        }
    }

    pub fn set_frequency(&mut self, samplerate: T, freq: T) {
        self.step = freq / samplerate;
    }
}
