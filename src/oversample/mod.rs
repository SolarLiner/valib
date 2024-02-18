use std::ops::{Deref, DerefMut};

use crate::dsp::parameter::{HasParameters, Parameter};
use crate::dsp::{
    utils::{mono_block_to_slice, mono_block_to_slice_mut, slice_to_mono_block_mut},
    DSP,
};
use crate::saturators::Linear;
use crate::Scalar;
use crate::{
    dsp::{blocks::Series, DSPBlock},
    filters::biquad::Biquad,
};

const CASCADE: usize = 5;

#[derive(Debug, Clone)]
pub struct Oversample<T> {
    os_factor: usize,
    os_buffer: Box<[T]>,
    pre_filter: Series<[Biquad<T, Linear>; CASCADE]>,
    post_filter: Series<[Biquad<T, Linear>; CASCADE]>,
}

impl<T: Scalar> Oversample<T> {
    pub fn new(os_factor: usize, max_block_size: usize) -> Self {
        assert!(os_factor > 1);
        let os_buffer = vec![T::zero(); max_block_size * os_factor].into_boxed_slice();
        let fc_raw = f64::recip(2.0 * os_factor as f64);
        let cascade_adjustment = f64::sqrt(2f64.powf(1.0 / CASCADE as f64) - 1.0).recip();
        let fc_corr = fc_raw * cascade_adjustment;
        println!("cascade adjustment {cascade_adjustment}: {fc_raw} -> {fc_corr}");
        let filters =
            std::array::from_fn(|_| Biquad::lowpass(T::from_f64(fc_raw), T::from_f64(0.707)));
        Self {
            os_factor,
            os_buffer,
            pre_filter: Series(filters),
            post_filter: Series(filters),
        }
    }

    pub fn latency(&self) -> usize {
        2 * self.os_factor + DSP::latency(&self.pre_filter) + DSP::latency(&self.post_filter)
    }

    pub fn max_block_size(&self) -> usize {
        self.os_buffer.len() / self.os_factor
    }

    pub fn oversample(&mut self, buffer: &[T]) -> OversampleBlock<T> {
        let os_len = self.zero_stuff(buffer);
        for s in &mut self.os_buffer[..os_len] {
            *s = self.pre_filter.process([*s])[0];
        }
        OversampleBlock {
            filter: self,
            os_len,
        }
    }

    pub fn reset(&mut self) {
        self.os_buffer.fill(T::zero());
        DSP::reset(&mut self.pre_filter);
        DSP::reset(&mut self.post_filter);
    }

    pub fn with_dsp<P: DSPBlock<1, 1>>(self, samplerate: f32, mut dsp: P) -> Oversampled<T, P> {
        let max_block_size = dsp.max_block_size().unwrap_or(self.os_buffer.len());
        // Verify that we satisfy the inner DSPBlock instance's requirement on maximum block size
        assert!(self.os_buffer.len() <= max_block_size);
        let staging_buffer = vec![[T::zero(); 1]; max_block_size].into_boxed_slice();
        dsp.set_samplerate(samplerate * self.os_factor as f32);
        Oversampled {
            oversampling: self,
            staging_buffer,
            inner: dsp,
            samplerate,
        }
    }

    fn zero_stuff(&mut self, inp: &[T]) -> usize {
        let os_len = inp.len() * self.os_factor;
        assert!(self.os_buffer.len() >= os_len);

        self.os_buffer[..os_len].fill(T::zero());
        for (i, s) in inp.iter().copied().enumerate() {
            self.os_buffer[self.os_factor * i] = s * T::from_f64(self.os_factor as f64);
        }
        os_len
    }

    fn decimate(&mut self, out: &mut [T]) {
        let os_len = out.len() * self.os_factor;
        assert!(os_len <= self.os_buffer.len());

        for (i, s) in self
            .os_buffer
            .iter()
            .step_by(self.os_factor)
            .copied()
            .enumerate()
            .take(out.len())
        {
            out[i] = s;
        }
    }
}

pub struct OversampleBlock<'a, T> {
    filter: &'a mut Oversample<T>,
    os_len: usize,
}

impl<'a, T> Deref for OversampleBlock<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.filter.os_buffer[..self.os_len]
    }
}

impl<'a, T> DerefMut for OversampleBlock<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.filter.os_buffer[..self.os_len]
    }
}

impl<'a, T: Scalar> OversampleBlock<'a, T> {
    pub fn finish(self, out: &mut [T]) {
        let filter = self.filter;
        for s in &mut filter.os_buffer[..self.os_len] {
            *s = filter.post_filter.process([*s])[0];
        }
        filter.decimate(out);
    }
}

pub struct Oversampled<T, P> {
    oversampling: Oversample<T>,
    staging_buffer: Box<[[T; 1]]>,
    pub inner: P,
    samplerate: f32,
}

impl<T, P> Oversampled<T, P> {
    pub fn os_factor(&self) -> usize {
        self.oversampling.os_factor
    }
}

impl<T, P> Oversampled<T, P>
where
    T: Scalar,
{
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<T, P> Oversampled<T, P>
where
    T: Scalar,
    P: DSPBlock<1, 1, Sample = T>,
{
    pub fn set_oversampling_amount(&mut self, amt: usize) {
        assert!(amt > 1);
        self.oversampling.os_factor = amt;
        self.set_samplerate(self.samplerate);
    }
}

impl<T, P> DSPBlock<1, 1> for Oversampled<T, P>
where
    T: Scalar,
    P: DSPBlock<1, 1, Sample = T>,
{
    type Sample = T;

    fn process_block(&mut self, inputs: &[[Self::Sample; 1]], outputs: &mut [[Self::Sample; 1]]) {
        let inputs = mono_block_to_slice(inputs);
        let mut os_block = self.oversampling.oversample(inputs);
        let inner_outputs = slice_to_mono_block_mut(&mut os_block);
        self.staging_buffer[..inner_outputs.len()].copy_from_slice(inner_outputs);
        self.inner
            .process_block(&self.staging_buffer[..inner_outputs.len()], inner_outputs);
        os_block.finish(mono_block_to_slice_mut(outputs));
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner
            .set_samplerate(self.oversampling.os_factor as f32 * samplerate);
    }

    fn max_block_size(&self) -> Option<usize> {
        Some(match self.inner.max_block_size() {
            Some(size) => size.min(self.oversampling.max_block_size() / self.os_factor()),
            None => self.oversampling.max_block_size() / self.os_factor(),
        })
    }

    fn latency(&self) -> usize {
        self.oversampling.latency() + self.inner.latency()
    }

    fn reset(&mut self) {
        self.oversampling.reset();
        self.inner.reset();
    }
}

impl<S, P: HasParameters> HasParameters for Oversampled<S, P> {
    type Enum = P::Enum;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        self.inner.get_parameter(param)
    }
}

#[cfg(test)]
mod tests {
    use std::{f32::consts::TAU, hint::black_box};

    use numeric_literals::replace_float_literals;

    use crate::{dsp::DSPBlock as _, Scalar};

    use super::Oversample;

    #[test]
    fn oversample_no_dc_offset() {
        let inp: [f32; 512] = std::array::from_fn(|i| (TAU * i as f32 / 64.).sin());
        let mut out = [0.0; 512];
        let mut os = Oversample::new(4, 512);

        let osblock = black_box(os.oversample(&inp));
        insta::assert_csv_snapshot!("os block", &*osblock, { "[]" => insta::rounded_redaction(3) });

        osblock.finish(&mut out);
        insta::assert_csv_snapshot!("post os", &out as &[_], { "[]" => insta::rounded_redaction(3) });
    }

    #[test]
    fn oversampled_dsp_block() {
        struct NaiveSquare<T> {
            samplerate: T,
            frequency: T,
            phase: T,
        }

        impl<T: Scalar> crate::dsp::DSP<1, 1> for NaiveSquare<T> {
            type Sample = T;

            #[replace_float_literals(T::from_f64(literal))]
            fn process(&mut self, _: [Self::Sample; 1]) -> [Self::Sample; 1] {
                let step = self.frequency / self.samplerate;
                let out = (1.0).select(self.phase.simd_gt(0.5), -1.0);
                self.phase += step;
                self.phase = (self.phase - 1.0).select(self.phase.simd_gt(1.0), self.phase);
                [out]
            }
        }

        let samplerate = 1000f32;
        let freq = 10f32;
        let dsp = NaiveSquare {
            samplerate,
            frequency: freq,
            phase: 0.0,
        };
        let mut os = Oversample::<f32>::new(4, 64).with_dsp(samplerate, dsp);

        let input = [[0.0]; 64];
        let mut output = [[0.0]; 64];
        os.process_block(&input, &mut output);
        insta::assert_csv_snapshot!(&output as &[_], { "[][]" => insta::rounded_redaction(3) });
    }
}
