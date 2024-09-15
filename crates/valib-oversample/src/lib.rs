#![warn(missing_docs)]
//! # Oversampling
//!
//! This crate provides oversampling capabilities, allowing you to oversample any block processor.
//!
//! Available here is a polyphase-based oversampling method, with more to come in the future.

use num_complex::Complex;

use valib_core::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib_core::dsp::parameter::HasParameters;
use valib_core::dsp::DSPProcessBlock;
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::simd::SimdComplexField;
use valib_core::Scalar;
use valib_filters::halfband;
use valib_filters::halfband::HalfbandFilter;

/// Ping-pong buffer. Allows processing of effect chains operating on buffers, by allowing the input
/// and output buffers be swapped after each effect.
#[derive(Debug, Clone)]
pub struct PingPongBuffer<T> {
    left: Box<[T]>,
    right: Box<[T]>,
    input_is_left: bool,
}

impl<T> PingPongBuffer<T> {
    /// Create a new ping-pong buffer
    ///
    /// # Arguments
    ///
    /// * `contents`: Initial contents of the buffers
    ///
    /// returns: PingPongBuffer<T>
    pub fn new<I: IntoIterator<Item = T>>(contents: I) -> Self
    where
        I::IntoIter: Clone,
    {
        let it = contents.into_iter();
        Self {
            left: it.clone().collect(),
            right: it.collect(),
            input_is_left: true,
        }
    }

    /// Fill the buffers
    ///
    /// # Arguments
    ///
    /// * `value`:
    ///
    /// returns: ()
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        self.left.fill(value);
        self.right.fill(value);
    }

    /// Get the input and output buffers.
    ///
    /// # Arguments
    ///
    /// * `index`: Value to slice the indices with, or .. to get the entire buffer at once.
    ///
    /// returns: (&[T], &mut [T])
    pub fn get_io_buffers<I: Clone>(&mut self, index: I) -> (&[T], &mut [T])
    where
        [T]: std::ops::IndexMut<I, Output = [T]>,
    {
        if self.input_is_left {
            let input = &self.left[index.clone()];
            let output = &mut self.right[index];
            (input, output)
        } else {
            let input = &self.right[index.clone()];
            let output = &mut self.left[index];
            (input, output)
        }
    }

    /// Get an immutable reference to the output buffer
    ///
    /// # Arguments
    ///
    /// * `index`: Index value to slice the buffer with, or .. to get the entire buffer at once
    ///
    /// returns: &[T]
    pub fn get_output_ref<I>(&self, index: I) -> &[T]
    where
        [T]: std::ops::Index<I, Output = [T]>,
    {
        if self.input_is_left {
            &self.right[index]
        } else {
            &self.left[index]
        }
    }

    /// Copy the output buffer of this ping-pong buffer into the provided output buffer.
    ///
    /// # Arguments
    ///
    /// * `output`: Output buffer into which the inner output buffer will be copied into.
    ///
    /// returns: ()
    pub fn copy_into(&self, output: &mut [T])
    where
        T: Copy,
    {
        let slice = if self.input_is_left {
            &self.right[..output.len()]
        } else {
            &self.left[..output.len()]
        };
        output.copy_from_slice(slice);
    }

    /// Switch the buffers around.
    pub fn switch(&mut self) {
        self.input_is_left = !self.input_is_left;
    }

    /// Returns true if the ping-pong buffers are empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    /// Returns the ping-pong buffer length.
    pub fn len(&self) -> usize {
        self.left.len()
    }
}

/// Single resample stage.
#[derive(Debug, Clone, Copy)]
pub struct ResampleStage<T, const UPSAMPLE: bool> {
    filter: HalfbandFilter<T, 6>,
}

impl<T: Scalar, const UPSAMPLE: bool> Default for ResampleStage<T, UPSAMPLE> {
    fn default() -> Self {
        Self {
            filter: halfband::steep_order12(),
        }
    }
}

impl<T: Scalar, const UPSAMPLE: bool> ResampleStage<T, UPSAMPLE> {
    /// Latency of the resample stage
    pub fn latency(&self) -> usize {
        self.filter.latency()
    }

    /// Reset the resample stage
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

impl<T: Scalar> ResampleStage<T, true> {
    /// Upsample a single sample of audio
    ///
    /// # Arguments
    ///
    /// * `s`: Input sample
    ///
    /// returns: [T; 2]
    pub fn process(&mut self, s: T) -> [T; 2] {
        let [x0] = self.filter.process([s + s]);
        let [x1] = self.filter.process([T::zero()]);
        [x0, x1]
    }

    /// Upsample the input buffer by a factor of 2.
    ///
    /// The output slice should be twice the length of the input slice.
    #[allow(clippy::identity_op)]
    pub fn process_block(&mut self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len() * 2, output.len());
        for (i, s) in input.iter().copied().enumerate() {
            let [x0, x1] = self.process(s);
            output[2 * i + 0] = x0;
            output[2 * i + 1] = x1;
        }
    }
}

impl<T: Scalar> ResampleStage<T, false> {
    /// Downsample 2 samples of input audio, and output a single sample of audio.
    ///
    /// # Arguments
    ///
    /// * `[x0, x1]`: Inputs samples
    ///
    /// returns: T
    pub fn process(&mut self, [x0, x1]: [T; 2]) -> T {
        let [y] = self.filter.process([x0]);
        let _ = self.filter.process([x1]);
        y
    }

    /// Downsample the input buffer by a factor of 2.
    ///
    /// The output slice should be twice the length of the input slice.
    #[allow(clippy::identity_op)]
    pub fn process_block(&mut self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), 2 * output.len());
        for i in 0..output.len() {
            let [y] = self.filter.process([input[2 * i + 0]]);
            let [_] = self.filter.process([input[2 * i + 1]]);
            output[i] = y;
        }
    }
}

/// Raw oversampling type. Works by taking a block of audio, processing it and returning a slice to
/// an internal buffer containing the upsampled audio data you should process in place. Once done,
/// call `.finish(output)` on the slice to downsample the internal buffer again, and output it to
/// `output`.
#[derive(Debug, Clone)]
pub struct Oversample<T> {
    max_factor: usize,
    num_stages_active: usize,
    os_buffer: PingPongBuffer<T>,
    upsample: Box<[ResampleStage<T, true>]>,
    downsample: Box<[ResampleStage<T, false>]>,
}

impl<T> Oversample<T> {
    /// Returns the current oversampling amount.
    pub fn oversampling_amount(&self) -> usize {
        usize::pow(2, self.num_stages_active as _)
    }

    /// Sets the oversampling amount.
    ///
    /// Only square numbers are supported; otherwise the next power of two from the given factor
    /// will be used.
    ///
    /// # Arguments
    ///
    /// `amt`: Oversampling amount. Needs to be less than or equal to the maximum oversampling rate
    ///     configured when constructed with [`Oversample::new`].
    pub fn set_oversampling_amount(&mut self, amt: usize) {
        assert!(amt <= self.max_factor);
        self.num_stages_active = amt.next_power_of_two().ilog2() as _;
    }

    /// Maximum block size supported at the current oversampling factor.
    pub fn max_block_size(&self) -> usize {
        self.os_buffer.len() / usize::pow(2, self.num_stages_active as _)
    }

    /// Return the length of the oversampled buffer.
    pub fn get_os_len(&self, input_len: usize) -> usize {
        input_len * usize::pow(2, self.num_stages_active as _)
    }
}

impl<T: Scalar> Oversample<T> {
    /// Create a new oversampling filter.
    ///
    /// # Arguments
    ///
    /// * `max_os_factor`: Maximum oversampling factor supported by this instance. The actual
    ///     oversampling can be changed after creation, but will need to always be less than or equal
    ///     to this factor.
    /// * `max_block_size`: Maximum block size that will be expected to be processed.
    ///
    /// returns: Oversample<T>
    pub fn new(max_os_factor: usize, max_block_size: usize) -> Self
    where
        Complex<T>: SimdComplexField,
    {
        assert!(max_os_factor >= 1);
        let max_os_factor = max_os_factor.next_power_of_two();
        let num_stages = max_os_factor.ilog2() as usize;
        let os_buffer = vec![T::zero(); max_block_size * max_os_factor];
        let os_buffer = PingPongBuffer::new(os_buffer);
        let upsample = (0..num_stages).map(|_| ResampleStage::default()).collect();
        let downsample = (0..num_stages).map(|_| ResampleStage::default()).collect();
        Self {
            max_factor: max_os_factor,
            num_stages_active: num_stages,
            os_buffer,
            upsample,
            downsample,
        }
    }

    /// Returns the latency of the filter. This includes both upsampling and downsampling.
    pub fn latency(&self) -> usize {
        let upsample_latency = self.upsample.iter().map(|p| p.latency()).sum::<usize>();
        let downsample_latency = self.downsample.iter().map(|p| p.latency()).sum::<usize>();
        2 * self.num_stages_active + upsample_latency + downsample_latency
    }

    /// Reset the state of this oversampling filter.
    pub fn reset(&mut self) {
        self.os_buffer.fill(T::zero());
        for stage in &mut self.upsample {
            stage.reset();
        }
        for stage in &mut self.downsample {
            stage.reset();
        }
    }

    /// Construct an [`Oversampled`] given this oversample instance and a block processor to wrap.
    pub fn with_dsp<P: DSPProcessBlock<1, 1>>(
        self,
        samplerate: f32,
        mut dsp: P,
    ) -> Oversampled<T, P> {
        let max_block_size = dsp.max_block_size().unwrap_or(self.os_buffer.len());
        // Verify that we satisfy the inner DSPBlock instance's requirement on maximum block size
        assert!(self.os_buffer.len() <= max_block_size);
        let staging_buffer = vec![T::zero(); max_block_size].into_boxed_slice();
        dsp.set_samplerate(samplerate * self.num_stages_active as f32);
        Oversampled {
            oversampling: self,
            staging_buffer,
            inner: dsp,
            base_samplerate: samplerate,
        }
    }

    #[profiling::function]
    fn upsample(&mut self, input: &[T]) -> &mut [T] {
        assert!(input.len() <= self.max_block_size());
        if self.num_stages_active == 0 {
            let (_, output) = self.os_buffer.get_io_buffers(..input.len());
            output.copy_from_slice(input);
            return output;
        }

        let os_len = self.get_os_len(input.len());
        let mut len = input.len();
        let (_, output) = self.os_buffer.get_io_buffers(..len);
        output.copy_from_slice(input);
        for stage in &mut self.upsample[..self.num_stages_active] {
            self.os_buffer.switch();
            let (input, output) = self.os_buffer.get_io_buffers(..2 * len);
            stage.process_block(&input[..len], output);
            len *= 2;
        }
        let (_, output) = self.os_buffer.get_io_buffers(..os_len);
        output
    }

    #[profiling::function]
    fn downsample(&mut self, out: &mut [T]) {
        if self.num_stages_active == 0 {
            let inner_out = self.os_buffer.get_output_ref(..out.len());
            out.copy_from_slice(inner_out);
            return;
        }

        let os_len = self.get_os_len(out.len());
        let mut len = os_len;
        for stage in &mut self.downsample {
            self.os_buffer.switch();
            let (input, output) = self.os_buffer.get_io_buffers(..len);
            stage.process_block(input, &mut output[..len / 2]);
            len /= 2;
        }
        self.os_buffer.copy_into(out);
    }
}

/// Wraps a block processor to orversample it, and allow using it within other DSP blocks.
///
/// Oversampling is transparently performed over the inner block processor.
pub struct Oversampled<T, P> {
    oversampling: Oversample<T>,
    staging_buffer: Box<[T]>,
    /// Inner processor
    pub inner: P,
    base_samplerate: f32,
}

impl<T, P> Oversampled<T, P> {
    /// Return the current oversampling factor
    pub fn os_factor(&self) -> usize {
        self.oversampling.oversampling_amount()
    }

    /// Drops the oversampling filter, returning the inner processor.
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<T, P> Oversampled<T, P>
where
    T: Scalar,
    P: DSPProcessBlock<1, 1, Sample = T>,
{
    /// Sets the oversampling amount. See [`Oversample::set_oversampling_amount`] for more details.
    pub fn set_oversampling_amount(&mut self, amt: usize) {
        assert!(amt >= 1);
        self.oversampling.set_oversampling_amount(amt);
        self.set_samplerate(self.base_samplerate);
    }

    /// Returns the sample rate of the oversampled buffer.
    pub fn inner_samplerate(&self) -> f32 {
        self.base_samplerate * self.oversampling.oversampling_amount() as f32
    }
}

impl<T: Scalar, P: DSPMeta<Sample = T>> DSPMeta for Oversampled<T, P> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner
            .set_samplerate(self.os_factor() as f32 * samplerate);
    }

    fn latency(&self) -> usize {
        self.inner.latency()
    }

    fn reset(&mut self) {
        self.oversampling.reset();
        self.inner.reset();
    }
}

#[profiling::all_functions]
impl<T, P> DSPProcessBlock<1, 1> for Oversampled<T, P>
where
    Self: DSPMeta<Sample = T>,
    T: Scalar,
    P: DSPProcessBlock<1, 1, Sample = T>,
{
    fn process_block(&mut self, inputs: AudioBufferRef<T, 1>, mut outputs: AudioBufferMut<T, 1>) {
        let os_block = self.oversampling.upsample(inputs.get_channel(0));

        let mut inner_input =
            AudioBufferMut::new([&mut self.staging_buffer[..os_block.len()]]).unwrap();
        inner_input.copy_from_slice(0, os_block);
        let inner_output = AudioBufferMut::new([os_block]).unwrap();

        self.inner.process_block(inner_input.as_ref(), inner_output);

        self.oversampling.downsample(outputs.get_channel_mut(0));
    }

    fn max_block_size(&self) -> Option<usize> {
        Some(self.oversampling.max_block_size())
    }
}

impl<S, P: HasParameters> HasParameters for Oversampled<S, P> {
    type Name = P::Name;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        self.inner.set_parameter(param, value)
    }
}

#[cfg(test)]
mod tests {
    use numeric_literals::replace_float_literals;
    use valib_core::dsp::{buffer::AudioBufferBox, DSPProcess, DSPProcessBlock as _};
    use valib_core::Scalar;
    use valib_core::{
        dsp::{BlockAdapter, DSPMeta},
        util::tests::{Plot, Series},
    };

    use super::{Oversample, PingPongBuffer};

    #[test]
    fn ping_pong_works() {
        let mut pingpong = PingPongBuffer::new([0; 8]);
        let (input, output) = pingpong.get_io_buffers(..);
        assert_eq!(0, input[0]);
        assert_eq!(0, output[0]);

        output[0] = 1;
        pingpong.switch();

        let (input, output) = pingpong.get_io_buffers(..);
        assert_eq!(1, input[0]);
        assert_eq!(0, output[0]);
    }

    #[test]
    fn oversampled_dsp_block() {
        use plotters::prelude::*;

        struct NaiveSquare<T> {
            samplerate: T,
            frequency: T,
            phase: T,
        }
        impl<T: Scalar> DSPMeta for NaiveSquare<T> {
            type Sample = T;
        }

        impl<T: Scalar> DSPProcess<1, 1> for NaiveSquare<T> {
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
            samplerate: 4.0 * samplerate,
            frequency: freq,
            phase: 0.0,
        };
        let mut os = Oversample::<f32>::new(4, 64).with_dsp(samplerate, BlockAdapter(dsp));

        let input = AudioBufferBox::zeroed(64);
        let mut output = AudioBufferBox::zeroed(64);
        os.process_block(input.as_ref(), output.as_mut());
        Plot {
            title: "Oversample block",
            bode: false,
            series: &[
                Series {
                    label: "Oversampled",
                    samplerate: 4. * samplerate,
                    series: os.oversampling.os_buffer.get_output_ref(..),
                    color: &YELLOW,
                },
                Series {
                    label: "Output",
                    samplerate,
                    series: output.get_channel(0),
                    color: &BLUE,
                },
            ],
        }
        .create_svg("plots/oversample/dsp_block.svg");
        insta::assert_csv_snapshot!(output.get_channel(0), { "[]" => insta::rounded_redaction(3) });
    }
}
