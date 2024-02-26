use crate::dsp::buffer::{AudioBufferBox, AudioBufferMut, AudioBufferRef};
use nalgebra::Complex;
use num_traits::Zero;

use crate::Scalar;

use self::analysis::DspAnalysis;

pub mod analysis;
pub mod blocks;
pub mod buffer;
pub mod parameter;
pub mod utils;

/// DSP trait. This is the main abstraction of the whole library.
///
/// Implementors of this trait are processes that work on a per-sample basis.
///
/// Multichannel I/O is supported, and is determined by the `I` and `O` const generics.
/// It's up to each implementor to document what the inputs and outputs mean.
/// There are no restrictions on the number of input or output channels in and of itself.
/// Implementors can support multiple configurations of I/O for different setups.
#[allow(unused_variables)]
pub trait DSP<const I: usize, const O: usize> {
    /// Type of the audio sample used by this DSP instance.
    type Sample: Scalar;

    /// Process a single sample of audio. No assumptions are made on the contents of said sample,
    /// so it can both work for working with audio data, but also control signals like frequency in Hertz
    /// for oscillators, or gate signals that are actually either 0 or 1.
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O];

    /// Sets the processing samplerate for this [`DSP`] instance.
    fn set_samplerate(&mut self, samplerate: f32) {}

    /// Report the latency of this DSP instance, that is the time, in samples, it takes for an input sample to be
    /// output back.
    fn latency(&self) -> usize {
        0
    }

    /// Reset this instance. Parameters should be kept, but any memory and derived state should be put back to a
    /// well-known default value.
    fn reset(&mut self) {}
}

/// Trait for DSP processes that take in buffers of audio instead of single-samples.
/// Documentation of [`DSP`] still applies in here; only the process method changes.
#[allow(unused_variables)]
pub trait DSPBlock<const I: usize, const O: usize> {
    type Sample: Scalar;

    /// Process a block of audio. Implementors should take the values of inputs and produce a stream into outputs,
    /// *as if* it was processed sample-by-sample. Ideally there should be no difference between a [`DSP`] and [`DSPBlock`]
    /// implementation, when both can exist. Note that [`DSPBlock`] is blanket-implemented for all [`DSP`] implementors,
    /// and as such, a simultaneous implementation of both onto one struct is impossible.
    ///
    /// Implementors should assume inputs and outputs are of the same length, as it is the caller's responsibility to make sure of that.
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, I>,
        outputs: AudioBufferMut<Self::Sample, O>,
    );

    /// Sets the processing samplerate for this [`DSP`] instance.
    fn set_samplerate(&mut self, samplerate: f32) {}

    /// Define an optional maximum buffer size alloed by this [`DSPBlock`] instance. Callers into this instance must
    /// then only provide buffers that are up to this size in samples.
    #[inline(always)]
    fn max_block_size(&self) -> Option<usize> {
        None
    }

    /// Report the latency of this DSP instance, that is the time, in samples, it takes for an input sample to be
    /// output back.
    #[inline(always)]
    fn latency(&self) -> usize {
        0
    }

    /// Reset this instance. Parameters should be kept, but any memory and derived state should be put back to a
    /// well-known default value.
    #[inline(always)]
    fn reset(&mut self) {}
}

impl<P, const I: usize, const O: usize> DSPBlock<I, O> for P
where
    P: DSP<I, O>,
{
    type Sample = <Self as DSP<I, O>>::Sample;

    #[inline(never)]
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<P::Sample, I>,
        mut outputs: AudioBufferMut<P::Sample, O>,
    ) {
        if I == 0 || O == 0 {
            return;
        }
        for i in 0..inputs.samples() {
            outputs.set_frame(i, self.process(inputs.get_frame(i)))
        }
    }

    fn set_samplerate(&mut self, samplerate: f32) {
        DSP::set_samplerate(self, samplerate)
    }

    #[inline(always)]
    fn latency(&self) -> usize {
        DSP::latency(self)
    }

    #[inline(always)]
    fn reset(&mut self) {
        DSP::reset(self)
    }
}

/// Adapt a [`DSPBlock`] instance to be able to used as a [`DSP`].
///
/// This introduces as much latency as the internal buffer size is.
/// The internal buffer size is determined by either the max accepted buffer size of the inner instance, or is set
/// to 64 samples by default.
pub struct PerSampleBlockAdapter<P, const I: usize, const O: usize>
where
    P: DSPBlock<I, O>,
{
    input_buffer: AudioBufferBox<P::Sample, I>,
    input_filled: usize,
    output_buffer: AudioBufferBox<P::Sample, O>,
    output_filled: usize,
    inner: P,
    pub buffer_size: usize,
}

impl<P, const I: usize, const O: usize> std::ops::Deref for PerSampleBlockAdapter<P, I, O>
where
    P: DSPBlock<I, O>,
{
    type Target = P;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P, const I: usize, const O: usize> std::ops::DerefMut for PerSampleBlockAdapter<P, I, O>
where
    P: DSPBlock<I, O>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<P, const I: usize, const O: usize> PerSampleBlockAdapter<P, I, O>
where
    P: DSPBlock<I, O>,
{
    pub const DEFAULT_BUFFER_SIZE: usize = 64;
    pub fn new(dsp_block: P) -> Self {
        Self::new_with_max_buffer_size(dsp_block, Self::DEFAULT_BUFFER_SIZE)
    }

    pub fn new_with_max_buffer_size(dsp_block: P, max_buffer_size: usize) -> Self {
        let buffer_size = dsp_block
            .max_block_size()
            .map(|mbs| mbs.min(max_buffer_size))
            .unwrap_or(max_buffer_size);
        Self {
            input_buffer: AudioBufferBox::zeroed(buffer_size),
            input_filled: 0,
            output_buffer: AudioBufferBox::zeroed(buffer_size),
            output_filled: buffer_size,
            buffer_size,
            inner: dsp_block,
        }
    }

    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P, const I: usize, const O: usize> DSP<I, O> for PerSampleBlockAdapter<P, I, O>
where
    P: DSPBlock<I, O>,
{
    type Sample = P::Sample;

    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        self.input_buffer.set_frame(self.input_filled, x);
        self.input_filled += 1;
        if self.input_buffer.samples() == self.input_filled {
            self.inner
                .process_block(self.input_buffer.as_ref(), self.output_buffer.as_mut());
            self.input_filled = 0;
            self.output_filled = 0;
        }

        if self.output_filled < self.buffer_size {
            let ret = self.output_buffer.get_frame(self.output_filled);
            self.output_filled += 1;
            ret
        } else {
            [Self::Sample::zero(); O]
        }
    }

    fn latency(&self) -> usize {
        (self.inner.latency() + self.input_buffer.samples()).saturating_sub(1)
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.input_filled = 0;
        self.output_filled = self.output_buffer.samples();
    }
}

impl<P, const I: usize, const O: usize> DspAnalysis<I, O> for PerSampleBlockAdapter<P, I, O>
where
    P: DspAnalysis<I, O>,
{
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; O]; I] {
        self.inner.h_z(z)
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use super::*;

    #[test]
    fn test_per_sample_block_adapter() {
        struct Counter<T>(PhantomData<T>);

        impl<T: Scalar> Counter<T> {
            pub fn new() -> Self {
                Self(PhantomData)
            }
        }

        impl<T: Scalar> DSPBlock<0, 1> for Counter<T> {
            type Sample = T;

            fn process_block(
                &mut self,
                inputs: AudioBufferRef<T, 0>,
                mut outputs: AudioBufferMut<T, 1>,
            ) {
                let len = inputs.samples();
                assert_eq!(len, outputs.samples());

                for (i, out) in outputs.get_channel_mut(0).iter_mut().enumerate() {
                    *out = T::from_f64(i as _);
                }
            }
        }

        let mut adaptor = PerSampleBlockAdapter::new_with_max_buffer_size(Counter::<f32>::new(), 4);
        assert_eq!(3, DSP::latency(&adaptor));

        let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
        let mut actual = [0.0; 8];

        // Calling `process_block` but it's actually calling the impl for all `DSP` passing each sample through.
        adaptor.process_block(
            AudioBufferRef::new([]).unwrap(),
            AudioBufferMut::new([&mut actual]).unwrap(),
        );

        assert_eq!(expected, actual);
    }
}
