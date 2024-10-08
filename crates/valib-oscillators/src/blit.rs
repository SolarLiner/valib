//! # Band-limited Impulse Train oscillators
//!
//! Provides oscillators which are generated from integrating BLITs, or Band-Limited Impulse Trains.
use numeric_literals::replace_float_literals;
use valib_core::dsp::DSPMeta;
use valib_core::dsp::DSPProcess;
use valib_core::Scalar;

/// Raw Band-Limited Impulse Train output. To be fed to leaky integrators to reconstruct the oscillator shape.
#[derive(Debug, Clone, Copy)]
pub struct Blit<T> {
    /// Current phase. Can be changed to perform phase modulation, at the cost of aliasing.
    pub p: T,
    dp: T,
    pmax: T,
    x: T,
    fc: T,
    samplerate: f32,
}

impl<T: Scalar> DSPMeta for Blit<T> {
    type Sample = T;

    #[replace_float_literals(T::from_f64(literal))]
    fn reset(&mut self) {
        self.p = 0.0;
        self.dp = 1.0;
        self.x = 0.0;
    }
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<0, 1> for Blit<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        self.p += self.dp;

        let neg_p = self.p.is_simd_negative();
        self.p = self.p.neg().select(neg_p, self.p);
        self.dp = self.dp.neg().select(neg_p, self.dp);

        let p_max = self.p.simd_gt(self.pmax);
        self.p = (self.pmax + self.pmax - self.p).select(p_max, self.p);
        self.dp = self.dp.neg().select(p_max, self.dp);

        self.x = T::simd_pi() * self.p;
        self.x = self.x.simd_max(1e-5);
        [self.x.simd_sin() / self.x]
    }
}

impl<T: Copy> Blit<T> {
    /// Maximum phase value
    #[inline(always)]
    pub fn pmax(&self) -> T {
        self.pmax
    }
}

impl<T: Scalar> Blit<T> {
    /// Construct a new BLIT with the given samplerate and oscillation frequency (in Hz).
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: f32, freq: T) -> Self {
        let mut this = Self {
            p: 0.0,
            dp: 1.0,
            pmax: 0.0,
            x: 0.0,
            fc: freq,
            samplerate,
        };
        this.update_coefficients();
        this
    }

    /// Set the samplerate and frequency (in Hz) of this instance.
    pub fn set_frequency(&mut self, freq: T) {
        self.fc = freq;
        self.update_coefficients();
    }

    /// Set the current position of the oscillator.
    ///
    /// # Arguments
    ///
    /// * `pos`: New position of the oscillator, normalized
    ///
    /// returns: ()
    pub fn set_position(&mut self, pos: T) {
        let delta = pos - self.p;
        self.p += delta * self.pmax;
    }

    /// Return a modified oscillator with the position set to the given value.
    ///
    /// # Arguments
    ///
    /// * `pos`: New position of the oscillator, normalized
    ///
    /// returns: Blit<T>
    pub fn with_position(mut self, pos: T) -> Self {
        self.set_position(pos);
        self
    }

    fn update_coefficients(&mut self) {
        self.pmax = T::from_f64(0.5 * self.samplerate as f64) / self.fc;
    }
}

/// BLIT sawtooth oscillator.
#[derive(Debug, Clone, Copy)]
pub struct Sawtooth<T> {
    blit: Blit<T>,
    integrator_state: T,
    dc: T,
}

impl<T: Scalar> DSPMeta for Sawtooth<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.blit.set_samplerate(samplerate);
    }
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<0, 1> for Sawtooth<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let [x] = self.blit.process(x);
        self.integrator_state = self.dc + x + 0.995 * self.integrator_state;
        [self.integrator_state]
    }
}

impl<T: Scalar> Sawtooth<T> {
    /// Create a new BLIT sawtooth wave oscillator, at the given samplerate with the given frequency (in Hz).
    pub fn new(samplerate: f32, freq: T) -> Self {
        let blit = Blit::new(samplerate, freq);
        Self {
            blit,
            integrator_state: T::from_f64(0.0),
            dc: Self::get_dc(blit.pmax),
        }
    }

    /// Set the samplerate and frequency (in Hz) of this instance.
    pub fn set_frequency(&mut self, freq: T) {
        self.dc = Self::get_dc(self.blit.pmax);
        self.blit.set_frequency(freq);
    }

    #[inline(always)]
    #[replace_float_literals(T::from_f64(literal))]
    fn get_dc(pmax: T) -> T {
        -0.498 / pmax
    }
}

/// BLIT pulse wave oscillator with variable pulse width modulation.
#[derive(Debug, Clone, Copy)]
pub struct Square<T> {
    blit_pos: Blit<T>,
    blit_neg: Blit<T>,
    pw: T,
    integrator_state: T,
}

impl<T: Scalar> DSPMeta for Square<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.blit_pos.set_samplerate(samplerate);
        self.blit_neg.set_samplerate(samplerate);
    }
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<0, 1> for Square<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let [xpos] = self.blit_pos.process(x);
        let [xneg] = self.blit_neg.process(x);
        let summed = xpos - xneg;
        self.integrator_state = summed + 0.9995 * self.integrator_state;
        // self.integrator_state = summed;
        [self.integrator_state]
    }
}

impl<T: Scalar> Square<T> {
    /// Create a new BLIT pulse wave, at the given samplerate with the given frequency (in Hz) and pulse width (in 0..1)
    pub fn new(samplerate: f32, freq: T, pw: T) -> Self {
        let blit = Blit::new(samplerate, freq);
        let mut this = Self {
            blit_pos: blit,
            blit_neg: blit,
            pw: T::from_f64(0.0),
            integrator_state: T::from_f64(0.0),
        };
        this.set_pulse_width(pw);
        this
    }

    /// Sets the pulse width of this instance, in (0..1)
    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_pulse_width(&mut self, pw: T) {
        let delta = pw - self.pw;
        self.blit_neg.p += 2.0 * delta * self.blit_neg.pmax;

        let state_delta = Self::get_offset(pw) - Self::get_offset(self.pw);
        self.integrator_state += state_delta;
    }

    /// Set the samplerate and frequency (in Hz) of this instance.
    pub fn set_frequency(&mut self, freq: T) {
        self.blit_pos.set_frequency(freq);
        self.blit_neg.set_frequency(freq);
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn get_offset(pw: T) -> T {
        pw
    }
}

#[cfg(test)]
mod tests {
    use valib_core::dsp::buffer::AudioBuffer;
    use valib_core::dsp::{BlockAdapter, DSPProcessBlock};

    use super::*;

    #[test]
    fn test_blit() {
        let mut blit = BlockAdapter(Blit::new(8192.0, 10.0));
        insta::assert_debug_snapshot!(&blit);
        let input = AudioBuffer::zeroed(8192);
        let mut actual = AudioBuffer::zeroed(8192);
        blit.process_block(input.as_ref(), actual.as_mut());
        insta::assert_csv_snapshot!(actual.get_channel(0), { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn test_sawtooth() {
        let mut saw = BlockAdapter(Sawtooth::new(8192.0, 10.0));
        insta::assert_debug_snapshot!(&saw);
        let input = AudioBuffer::zeroed(8192);
        let mut actual = AudioBuffer::zeroed(8192);
        saw.process_block(input.as_ref(), actual.as_mut());
        insta::assert_csv_snapshot!(actual.get_channel(0), { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn test_square() {
        let mut square = BlockAdapter(Square::new(8192.0, 10.0, 0.5));
        insta::assert_debug_snapshot!(&square);
        let input = AudioBuffer::zeroed(8192);
        let mut actual = AudioBuffer::zeroed(8192);
        square.process_block(input.as_ref(), actual.as_mut());
        insta::assert_csv_snapshot!(actual.get_channel(0), { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn test_square_pw() {
        let mut square = BlockAdapter(Square::new(8192.0, 10.0, 0.1));
        insta::assert_debug_snapshot!(&square);
        let input = AudioBuffer::zeroed(8192);
        let mut actual = AudioBuffer::zeroed(8192);
        square.process_block(input.as_ref(), actual.as_mut());
        insta::assert_csv_snapshot!(actual.get_channel(0), { "[]" => insta::rounded_redaction(4) });
    }
}
