use num_traits::Num;
use std::ops::{Deref, DerefMut};

use crate::dsp::parameter::{HasParameters, Parameter};
use crate::dsp::DSPBlock;
use crate::dsp::{
    utils::{mono_block_to_slice, mono_block_to_slice_mut, slice_to_mono_block_mut},
    DSP,
};
use crate::math::interpolation::{Cubic, Interpolate, SimdIndex, SimdInterpolatable};
use crate::{Scalar, SimdCast};

const CASCADE: usize = 4;

#[derive(Debug, Clone)]
pub struct Oversample<T> {
    os_factor: usize,
    os_buffer: Box<[T]>,
}

impl<T: Scalar> Oversample<T> {
    pub fn new(os_factor: usize, max_block_size: usize) -> Self {
        assert!(os_factor > 1);
        let os_buffer = vec![T::zero(); max_block_size * os_factor].into_boxed_slice();
        Self {
            os_factor,
            os_buffer,
        }
    }

    pub fn latency(&self) -> usize {
        2 * self.os_factor
    }

    pub fn max_block_size(&self) -> usize {
        self.os_buffer.len() / self.os_factor
    }

    pub fn oversample(&mut self, buffer: &[T]) -> OversampleBlock<T>
    where
        Cubic: Interpolate<T, 4>,
        T: SimdInterpolatable,
        <T as SimdCast<usize>>::Output: SimdIndex,
    {
        let os_len = buffer.len() * self.os_factor;
        let output = &mut self.os_buffer[..os_len];
        Cubic.interpolate_slice(output, buffer);

        OversampleBlock {
            filter: self,
            os_len,
        }
    }

    pub fn reset(&mut self) {
        self.os_buffer.fill(T::zero());
    }

    pub fn with_dsp<P: DSPBlock<1, 1>>(self, dsp: P) -> Oversampled<T, P> {
        let max_block_size = dsp.max_block_size().unwrap_or(self.os_buffer.len());
        // Verify that we satisfy the inner DSPBlock instance's requirement on maximum block size
        assert!(self.os_buffer.len() <= max_block_size);
        let staging_buffer = vec![[T::zero(); 1]; max_block_size].into_boxed_slice();
        Oversampled {
            oversampling: self,
            staging_buffer,
            inner: dsp,
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

impl<'a, T: Scalar + SimdInterpolatable> OversampleBlock<'a, T>
where
    Cubic: Interpolate<T, 4>,
    <T as SimdCast<usize>>::Output: SimdIndex,
{
    pub fn finish(self, out: &mut [T]) {
        let inlen = out.len() * self.filter.os_factor;
        let input = &self.filter.os_buffer[..inlen];
        Cubic.interpolate_slice(out, input);
    }
}

pub struct Oversampled<T, P> {
    oversampling: Oversample<T>,
    staging_buffer: Box<[[T; 1]]>,
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
    T: Scalar + SimdInterpolatable,
    <T as SimdCast<usize>>::Output: SimdIndex,
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
            None => self.oversampling.max_block_size(),
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
        let mut os = Oversample::<f32>::new(4, 64).with_dsp(dsp);

        let input = [[0.0]; 64];
        let mut output = [[0.0]; 64];
        os.process_block(&input, &mut output);
        insta::assert_csv_snapshot!(&output as &[_], { "[][]" => insta::rounded_redaction(3) });
    }
}
