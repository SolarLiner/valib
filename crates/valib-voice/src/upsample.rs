use num_traits::zero;
use valib_core::dsp::buffer::{AudioBufferBox, AudioBufferMut, AudioBufferRef};
use valib_core::dsp::{DSPMeta, DSPProcessBlock};
use valib_oversample::{PingPongBuffer, ResampleStage};

/// Upsampled voice adapter
pub struct UpsampledVoice<P: DSPMeta> {
    /// Inner voice
    pub inner: P,
    downsample_stages: Box<[ResampleStage<P::Sample, false>]>,
    ping_pong_buffer: PingPongBuffer<P::Sample>,
    num_active_stages: usize,
}

impl<P: DSPProcessBlock<0, 1>> DSPProcessBlock<0, 1> for UpsampledVoice<P> {
    fn process_block(
        &mut self,
        _: AudioBufferRef<Self::Sample, 0>,
        mut outputs: AudioBufferMut<Self::Sample, 1>,
    ) {
        let out_len = outputs.samples();
        let inner_len = out_len * self.upsampling_amount();
        let (_, input) = self.ping_pong_buffer.get_io_buffers(..inner_len);
        self.inner.process_block(
            AudioBufferBox::empty(input.len()).as_ref(),
            AudioBufferMut::from(input),
        );
        self.ping_pong_buffer.switch();
        let mut length = inner_len;
        for stage in &mut self.downsample_stages[..self.num_active_stages] {
            let (input, output) = self.ping_pong_buffer.get_io_buffers(..length);
            stage.process_block(input, output);
            self.ping_pong_buffer.switch();
            length /= 2;
        }
        let (input, _) = self.ping_pong_buffer.get_io_buffers(..out_len);
        outputs.copy_from_slice(0, input);
    }
}

impl<P: DSPMeta> DSPMeta for UpsampledVoice<P> {
    type Sample = P::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.inner
            .set_samplerate(self.upsampling_amount() as f32 * samplerate);
    }

    fn latency(&self) -> usize {
        self.downsample_stages[..self.num_active_stages]
            .iter()
            .map(|s| s.latency())
            .sum::<usize>()
            + self.inner.latency()
    }

    fn reset(&mut self) {
        for stage in &mut self.downsample_stages {
            stage.reset();
        }
        self.inner.reset();
    }
}

impl<V: DSPMeta> UpsampledVoice<V> {
    /// Create a new upsampled voice
    ///
    /// # Arguments
    ///
    /// * `max_upsample_amount`: Maximum upsampling amount
    /// * `inner`: Inner voice processor
    ///
    /// returns: UpsampledVoice<V>
    pub fn new(max_upsample_amount: usize, max_buffer_size: usize, inner: V) -> Self {
        let max_stages = max_upsample_amount.next_power_of_two().ilog2() as usize;
        let max_os_buffer_len = max_upsample_amount * max_buffer_size;
        Self {
            inner,
            downsample_stages: (0..max_stages).map(|_| ResampleStage::default()).collect(),
            ping_pong_buffer: PingPongBuffer::new(
                std::iter::repeat_with(zero).take(max_os_buffer_len),
            ),
            num_active_stages: max_stages,
        }
    }

    /// Current upsampling amount
    pub fn upsampling_amount(&self) -> usize {
        2usize.pow(self.num_active_stages as _)
    }

    /// Set the upsampling amount.
    ///
    /// # Arguments
    ///
    /// * `amt`: Upsampling amount. Will be set to the next power of two if it isn't already.
    ///
    /// returns: ()
    pub fn set_upsampling_amount(&mut self, amt: usize) {
        let num_stages = amt.next_power_of_two().ilog2() as usize;
        assert!(num_stages <= self.downsample_stages.len());
        self.num_active_stages = num_stages;
    }

    /// Drop the upsampled voice, returning the inner voice
    pub fn into_inner(self) -> V {
        self.inner
    }
}
