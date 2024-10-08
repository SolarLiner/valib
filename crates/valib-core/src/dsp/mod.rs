#![doc = include_str!("README.md")]

use nalgebra::Complex;
use num_traits::Zero;

use crate::dsp::buffer::{AudioBufferBox, AudioBufferMut, AudioBufferRef};
use crate::dsp::parameter::HasParameters;
use crate::Scalar;

use self::analysis::DspAnalysis;

pub mod analysis;
pub mod blocks;
pub mod buffer;
pub mod parameter;

/// Trait for interacting with a DSP algorithm, outside of processing. Shared by processors of both
/// per-sample algorithms and block-based algorithms.
#[allow(unused_variables)]
pub trait DSPMeta {
    /// Type of the audio sample used by this DSP instance.
    type Sample: Scalar;

    /// Sets the processing samplerate for this [`DSPProcess`] instance.
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

/// DSP trait. This is the main abstraction of the whole library.
///
/// Implementors of this trait are processes that work on a per-sample basis.
///
/// Multichannel I/O is supported, and is determined by the `I` and `O` const generics.
/// It's up to each implementor to document what the inputs and outputs mean.
/// There are no restrictions on the number of input or output channels in and of itself.
/// Implementors can support multiple configurations of I/O for different setups.
pub trait DSPProcess<const I: usize, const O: usize>: DSPMeta {
    /// Process a single sample of audio. No assumptions are made on the contents of said sample,
    /// so it can both work for working with audio data, but also control signals like frequency in Hertz
    /// for oscillators, or gate signals that are actually either 0 or 1.
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O];
}

/// Trait for DSP processes that take in buffers of audio instead of single-samples.
/// Documentation of [`DSPProcess`] still applies in here; only the process method changes.
pub trait DSPProcessBlock<const I: usize, const O: usize>: DSPMeta {
    /// Process a block of audio. Implementors should take the values of inputs and produce a stream into outputs,
    /// *as if* it was processed sample-by-sample. Ideally there should be no difference between a [`DSPProcess`] and [`DSPProcessBlock`]
    /// implementation, when both can exist. Note that [`DSPProcessBlock`] is blanket-implemented for all [`DSPProcess`] implementors,
    /// and as such, a simultaneous implementation of both onto one struct is impossible.
    ///
    /// Implementors should assume inputs and outputs are of the same length, as it is the caller's responsibility to make sure of that.
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, I>,
        outputs: AudioBufferMut<Self::Sample, O>,
    );

    /// Define an optional maximum buffer size alloed by this [`DSPProcessBlock`] instance. Callers into this instance must
    /// then only provide buffers that are up to this size in samples.
    #[inline(always)]
    fn max_block_size(&self) -> Option<usize> {
        None
    }
}

/// Adapter for per-sample processes implementing [`DSPProcess`], so that they work as a [`DSPProcessBlock`].
#[derive(Debug, Copy, Clone)]
pub struct BlockAdapter<P>(pub P);

impl<P: HasParameters> HasParameters for BlockAdapter<P> {
    type Name = P::Name;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        self.0.set_parameter(param, value)
    }
}

impl<P: DSPMeta> DSPMeta for BlockAdapter<P> {
    type Sample = P::Sample;
}

impl<P: DSPProcess<I, O>, const I: usize, const O: usize> DSPProcess<I, O> for BlockAdapter<P> {
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        self.0.process(x)
    }
}

#[profiling::all_functions]
impl<P, const I: usize, const O: usize> DSPProcessBlock<I, O> for BlockAdapter<P>
where
    P: DSPProcess<I, O>,
{
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<P::Sample, I>,
        mut outputs: AudioBufferMut<P::Sample, O>,
    ) {
        if I == 0 && O == 0 {
            return;
        }
        for i in 0..inputs.samples() {
            outputs.set_frame(i, self.0.process(inputs.get_frame(i)))
        }
    }
}

/// Adapt a [`DSPProcessBlock`] instance to be able to used as a [`DSPProcess`].
///
/// This introduces as much latency as the internal buffer size is.
/// The internal buffer size is determined by either the max accepted buffer size of the inner instance, or is set
/// to 64 samples by default.
pub struct SampleAdapter<P, const I: usize, const O: usize>
where
    P: DSPProcessBlock<I, O>,
{
    /// Size of the buffers passed into the inner block processor.
    pub buffer_size: usize,
    input_buffer: AudioBufferBox<P::Sample, I>,
    input_filled: usize,
    output_buffer: AudioBufferBox<P::Sample, O>,
    output_filled: usize,
    inner: P,
}

impl<P, const I: usize, const O: usize> std::ops::Deref for SampleAdapter<P, I, O>
where
    P: DSPProcessBlock<I, O>,
{
    type Target = P;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P, const I: usize, const O: usize> std::ops::DerefMut for SampleAdapter<P, I, O>
where
    P: DSPProcessBlock<I, O>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<P, const I: usize, const O: usize> SampleAdapter<P, I, O>
where
    P: DSPProcessBlock<I, O>,
{
    /// Default buffer size used without constraints from the inner processor or explicit size given
    /// by the user.
    pub const DEFAULT_BUFFER_SIZE: usize = 64;

    /// Create a new adapter for block processes to be used per-sample.
    ///
    /// # Arguments
    ///
    /// * `dsp_block`: Block process to adapt
    ///
    /// returns: SampleAdapter<P, { I }, { O }>
    pub fn new(dsp_block: P) -> Self {
        Self::new_with_max_buffer_size(dsp_block, Self::DEFAULT_BUFFER_SIZE)
    }

    /// Create a new per-sample adaptor with the given buffer size for the inner block processor
    ///
    /// # Arguments
    ///
    /// * `dsp_block`: Block process to adapt
    /// * `max_buffer_size`: Maximum buffer size to be passed to the buffer.
    ///
    /// returns: SampleAdapter<P, { I }, { O }>
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

    /// Drop this per-sample adapter, and return the inner block process
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P, const I: usize, const O: usize> DSPMeta for SampleAdapter<P, I, O>
where
    P: DSPProcessBlock<I, O>,
{
    type Sample = P::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        (self.inner.latency() + self.input_buffer.samples()).saturating_sub(1)
    }

    fn reset(&mut self) {
        self.input_filled = 0;
        self.output_filled = self.output_buffer.samples();
        self.input_buffer.fill(P::Sample::zero());
        self.output_buffer.fill(P::Sample::zero());
    }
}

#[profiling::all_functions]
impl<P, const I: usize, const O: usize> DSPProcess<I, O> for SampleAdapter<P, I, O>
where
    P: DSPProcessBlock<I, O>,
{
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
}

impl<P, const I: usize, const O: usize> DspAnalysis<I, O> for SampleAdapter<P, I, O>
where
    P: DSPProcessBlock<I, O> + DspAnalysis<I, O>,
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

        impl<T: Scalar> DSPMeta for Counter<T> {
            type Sample = T;
        }

        impl<T: Scalar> DSPProcessBlock<0, 1> for Counter<T> {
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

        let adaptor = SampleAdapter::new_with_max_buffer_size(Counter::<f32>::new(), 4);
        assert_eq!(3, adaptor.latency());

        let expected = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
        let mut actual = [0.0; 8];

        BlockAdapter(adaptor).process_block(
            AudioBufferRef::empty(8),
            AudioBufferMut::new([&mut actual]).unwrap(),
        );

        assert_eq!(expected, actual);
    }
}
