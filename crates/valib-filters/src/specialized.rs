use crate::biquad::Biquad;
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::Scalar;
use valib_saturators::Linear;

pub struct DcBlocker<T>(Biquad<T, Linear>);

impl<T> DcBlocker<T> {
    const CUTOFF_HZ: f32 = 5.0;
    const Q: f32 = 0.707;
    pub fn new(samplerate: f32) -> Self
    where
        T: Scalar,
    {
        Self(Biquad::highpass(
            T::from_f64((Self::CUTOFF_HZ / samplerate) as f64),
            T::from_f64(Self::Q as f64),
        ))
    }
}

impl<T: Scalar> DSPMeta for DcBlocker<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.0.set_samplerate(samplerate);
        self.0.update_coefficients(&Biquad::highpass(
            T::from_f64((Self::CUTOFF_HZ / samplerate) as f64),
            T::from_f64(Self::Q as f64),
        ));
    }

    fn latency(&self) -> usize {
        self.0.latency()
    }

    fn reset(&mut self) {
        self.0.reset()
    }
}

impl<T: Scalar> DSPProcess<1, 1> for DcBlocker<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.0.process(x)
    }
}