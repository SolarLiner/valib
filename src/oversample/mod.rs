use std::ops::{Deref, DerefMut};

use crate::dsp::DSP;
use crate::dsp::{
    buffer::{AudioBufferMut, AudioBufferRef},
    parameter::{HasParameters, Parameter},
};
use crate::saturators::Linear;
use crate::Scalar;
use crate::{
    dsp::{blocks::Series, DSPBlock},
    filters::biquad::Biquad,
};

#[derive(Debug, Clone)]
pub struct Oversample<T> {
    os_factor: usize,
    os_buffer: Box<[T]>,
    pre_filter: Series<[Biquad<T, Linear>; 8]>,
    post_filter: Series<[Biquad<T, Linear>; 8]>,
}

impl<T: Scalar> Oversample<T> {
    pub fn new(os_factor: usize, max_block_size: usize) -> Self {
        assert!(os_factor > 1);
        let os_buffer = vec![T::zero(); max_block_size * os_factor].into_boxed_slice();
        let filters = std::array::from_fn(|_| {
            Biquad::lowpass(
                T::from_f64(2.0 * os_factor as f64).simd_recip(),
                T::from_f64(0.707),
            )
        });
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

    pub fn with_dsp<P: DSPBlock<1, 1>>(self, dsp: P) -> Oversampled<T, P> {
        let max_block_size = dsp.max_block_size().unwrap_or(self.os_buffer.len());
        // Verify that we satisfy the inner DSPBlock instance's requirement on maximum block size
        assert!(self.os_buffer.len() <= max_block_size);
        let staging_buffer = vec![T::zero(); max_block_size].into_boxed_slice();
        Oversampled {
            oversampling: self,
            staging_buffer,
            inner: dsp,
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
    staging_buffer: Box<[T]>,
    pub inner: P,
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
    #[deprecated = "Use Oversample::with_dsp"]
    pub fn new(oversampling: Oversample<T>, inner: P) -> Self
    where
        P: DSP<1, 1, Sample = T>,
    {
        oversampling.with_dsp(inner)
    }
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<T, P> DSPBlock<1, 1> for Oversampled<T, P>
where
    T: Scalar,
    P: DSPBlock<1, 1, Sample = T>,
{
    type Sample = T;

    fn latency(&self) -> usize {
        self.oversampling.latency() + self.inner.latency()
    }

    fn max_block_size(&self) -> Option<usize> {
        Some(self.oversampling.max_block_size())
    }

    fn reset(&mut self) {
        self.oversampling.reset();
        self.inner.reset();
    }

    fn process_block(&mut self, inputs: AudioBufferRef<T, 1>, mut outputs: AudioBufferMut<T, 1>) {
        let mut os_block = self.oversampling.oversample(inputs.get_channel(0));
        let input = &mut self.staging_buffer[..os_block.len()];
        input.copy_from_slice(&os_block);
        self.inner.process_block(
            AudioBufferRef::from(&*input),
            AudioBufferMut::from(&mut *os_block),
        );
        os_block.finish(outputs.get_channel_mut(0));
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner
            .set_samplerate(self.oversampling.os_factor as f32 * samplerate);
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

    use crate::{
        dsp::{buffer::AudioBufferBox, DSPBlock as _},
        Scalar,
    };

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
        let mut os = Oversample::<f32>::new(4, 64).with_dsp(dsp);

        let input = AudioBufferBox::zeroed(64);
        let mut output = AudioBufferBox::zeroed(64);
        os.process_block(input.as_ref(), output.as_mut());
        insta::assert_csv_snapshot!(output.get_channel(0), { "[][]" => insta::rounded_redaction(3) });
    }
}
