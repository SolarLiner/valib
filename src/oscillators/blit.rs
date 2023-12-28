use numeric_literals::replace_float_literals;

use crate::{dsp::DSP, Scalar};

/// Raw BLIT output, to be passed to a leaky integrator or a full lowpass filter
#[derive(Debug, Clone, Copy)]
pub struct Blit<T> {
    p: T,
    dp: T,
    pmax: T,
    x: T,
}

impl<T: Scalar> DSP<0, 1> for Blit<T> {
    type Sample = T;

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

    #[replace_float_literals(T::from_f64(literal))]
    fn reset(&mut self) {
        self.p = 0.0;
        self.dp = 1.0;
        self.x = 0.0;
    }
}

impl<T: Scalar> Blit<T> {
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: T, freq: T) -> Self {
        let mut this = Self {
            p: 0.0,
            dp: 1.0,
            pmax: 0.0,
            x: 0.0,
        };
        this.set_frequency(samplerate, freq);
        this
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_frequency(&mut self, samplerate: T, freq: T) {
        self.pmax = 0.5 * samplerate / freq;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sawtooth<T> { blit: Blit<T>, integrator_state: T, dc: T }

impl<T: Scalar> DSP<0, 1> for Sawtooth<T> {
    type Sample = T;

    #[replace_float_literals(T::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let [x] = self.blit.process(x);
        self.integrator_state = self.dc + x + 0.995 * self.integrator_state;
        [self.integrator_state]
    }
}

impl<T: Scalar> Sawtooth<T> {
    pub fn new(samplerate: T, freq: T) -> Self {
        let blit = Blit::new(samplerate, freq);
        Self {
            blit,
            integrator_state: T::from_f64(0.0),
            dc: Self::get_dc(blit.pmax),
        }
    }

    pub fn set_frequency(&mut self, samplerate: T, freq: T) {
        self.dc = Self::get_dc(self.blit.pmax);
        self.blit.set_frequency(samplerate, freq);
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn get_dc(pmax: T) -> T {
        -0.498 / pmax
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Square<T> {
    blit_pos: Blit<T>,
    blit_neg: Blit<T>,
    pw: T,
    integrator_state: T,
}

impl<T: Scalar> DSP<0, 1> for Square<T> {
    type Sample = T;

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
    pub fn new(samplerate: T, freq: T, pw: T) -> Self {
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

    #[replace_float_literals(T::from_f64(literal))]
    pub fn set_pulse_width(&mut self, pw: T) {
        let delta = pw - self.pw;
        self.blit_neg.p += 2.0 * delta * self.blit_neg.pmax;

        let state_delta = Self::get_offset(pw) - Self::get_offset(self.pw);
        self.integrator_state += state_delta;
    }

    pub fn set_frequency(&mut self, samplerate: T, freq: T) {
        self.blit_pos.set_frequency(samplerate, freq);
        self.blit_neg.set_frequency(samplerate, freq);
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn get_offset(pw: T) -> T {
        pw
    }
}

#[cfg(test)]
mod tests {
    use crate::dsp::{utils::slice_to_mono_block_mut, DSPBlock};

    use super::*;

    #[test]
    fn test_blit() {
        let mut blit = Blit::new(8192.0, 10.0);
        insta::assert_debug_snapshot!(&blit);
        let mut actual = [0.0; 8192];
        blit.process_block(&[[]; 8192], slice_to_mono_block_mut(&mut actual));
        insta::assert_csv_snapshot!(&actual as &[_], { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn test_sawtooth() {
        let mut saw = Sawtooth::new(8192.0, 10.0);
        insta::assert_debug_snapshot!(&saw);
        let mut actual = [0.0; 8192];
        saw.process_block(&[[]; 8192], slice_to_mono_block_mut(&mut actual));
        insta::assert_csv_snapshot!(&actual as &[_], { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn test_square() {
        let mut square = Square::new(8192.0, 10.0, 0.5);
        insta::assert_debug_snapshot!(&square);
        let mut actual = [0.0; 8192];
        square.process_block(&[[]; 8192], slice_to_mono_block_mut(&mut actual));
        insta::assert_csv_snapshot!(&actual as &[_], { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn test_square_pw() {
        let mut square = Square::new(8192.0, 10.0, 0.1);
        insta::assert_debug_snapshot!(&square);
        let mut actual = [0.0; 8192];
        square.process_block(&[[]; 8192], slice_to_mono_block_mut(&mut actual));
        insta::assert_csv_snapshot!(&actual as &[_], { "[]" => insta::rounded_redaction(4) });
    }
}
