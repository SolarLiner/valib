use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut};

use nalgebra::Complex;
use simba::simd::SimdComplexField;

use crate::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use crate::dsp::parameter::HasParameters;
use crate::dsp::DSPProcessBlock;
use crate::dsp::{DSPMeta, DSPProcess};
use crate::filters::halfband;
use crate::filters::halfband::HalfbandFilter;
use crate::voice::VoiceManager;
use crate::Scalar;

const CASCADE: usize = 16;

#[derive(Debug, Clone)]
struct PingPongBuffer<T> {
    left: Box<[T]>,
    right: Box<[T]>,
    input_is_left: bool,
}

impl<T> PingPongBuffer<T> {
    fn new<I: IntoIterator<Item = T>>(contents: I) -> Self
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

    fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        self.left.fill(value);
        self.right.fill(value);
    }

    fn get_io_buffers<I: Clone>(&mut self, index: I) -> (&[T], &mut [T])
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

    fn get_output_ref<I>(&self, index: I) -> &[T]
    where
        [T]: std::ops::Index<I, Output = [T]>,
    {
        if self.input_is_left {
            &self.right[index]
        } else {
            &self.left[index]
        }
    }

    fn copy_into(&self, output: &mut [T])
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

    fn switch(&mut self) {
        self.input_is_left = !self.input_is_left;
    }

    fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    fn len(&self) -> usize {
        self.left.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct ResampleStage<T, const UPSAMPLE: bool> {
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
    fn latency(&self) -> usize {
        self.filter.latency()
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

impl<T: Scalar> ResampleStage<T, true> {
    #[allow(clippy::identity_op)]
    fn process_block(&mut self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len() * 2, output.len());
        for (i, s) in input.iter().copied().enumerate() {
            let [x0] = self.filter.process([s + s]);
            let [x1] = self.filter.process([T::zero()]);
            output[2 * i + 0] = x0;
            output[2 * i + 1] = x1;
        }
    }
}

impl<T: Scalar> ResampleStage<T, false> {
    #[allow(clippy::identity_op)]
    fn process_block(&mut self, input: &[T], output: &mut [T]) {
        assert_eq!(input.len(), 2 * output.len());
        for i in 0..output.len() {
            let [y] = self.filter.process([input[2 * i + 0]]);
            let [_] = self.filter.process([input[2 * i + 1]]);
            output[i] = y;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Oversample<T> {
    max_factor: usize,
    num_stages_active: usize,
    os_buffer: PingPongBuffer<T>,
    upsample: Box<[ResampleStage<T, true>]>,
    downsample: Box<[ResampleStage<T, false>]>,
}

impl<T: Scalar> Oversample<T> {
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

    pub fn oversampling_amount(&self) -> usize {
        usize::pow(2, self.num_stages_active as _)
    }

    pub fn set_oversampling_amount(&mut self, amt: usize) {
        assert!(amt <= self.max_factor);
        self.num_stages_active = amt.next_power_of_two().ilog2() as _;
    }

    pub fn latency(&self) -> usize {
        let upsample_latency = self.upsample.iter().map(|p| p.latency()).sum::<usize>();
        let downsample_latency = self.downsample.iter().map(|p| p.latency()).sum::<usize>();
        2 * self.num_stages_active + upsample_latency + downsample_latency
    }

    pub fn max_block_size(&self) -> usize {
        self.os_buffer.len() / usize::pow(2, self.num_stages_active as _)
    }

    pub fn get_os_len(&self, input_len: usize) -> usize {
        input_len * usize::pow(2, self.num_stages_active as _)
    }

    pub fn reset(&mut self) {
        self.os_buffer.fill(T::zero());
        for stage in &mut self.upsample {
            stage.reset();
        }
        for stage in &mut self.downsample {
            stage.reset();
        }
    }

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

pub struct Oversampled<T, P> {
    oversampling: Oversample<T>,
    staging_buffer: Box<[T]>,
    pub inner: P,
    base_samplerate: f32,
}

impl<T, P> Oversampled<T, P> {
    pub fn os_factor(&self) -> usize {
        usize::pow(2, self.oversampling.num_stages_active as _)
    }

    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<T, P> Oversampled<T, P>
where
    T: Scalar,
    P: DSPProcessBlock<1, 1, Sample = T>,
{
    pub fn set_oversampling_amount(&mut self, amt: usize) {
        assert!(amt >= 1);
        self.oversampling.set_oversampling_amount(amt);
        self.set_samplerate(self.base_samplerate);
    }

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

impl<S, P, const N: usize> VoiceManager<N> for Oversampled<S, P>
where
    P: VoiceManager<N>,
{
    fn note_on(&mut self, midi_note: u8, velocity: f32) {
        self.inner.note_on(midi_note, velocity);
    }

    fn note_off(&mut self, midi_note: u8, velocity: f32) {
        self.inner.note_off(midi_note, velocity);
    }

    fn choke(&mut self, midi_note: u8) {
        self.inner.choke(midi_note);
    }

    fn panic(&mut self) {
        self.inner.panic();
    }

    fn pitch_bend(&mut self, amount: f32) {
        self.inner.pitch_bend(amount)
    }

    fn aftertouch(&mut self, amount: f32) {
        self.inner.aftertouch(amount)
    }

    fn pressure(&mut self, midi_note: u8, pressure: f32) {
        self.inner.pressure(midi_note, pressure)
    }

    fn glide(&mut self, midi_note: u8, semitones: f32) {
        self.inner.glide(midi_note, semitones)
    }

    fn pan(&mut self, midi_note: u8, pan: f32) {
        self.inner.pan(midi_note, pan)
    }

    fn gain(&mut self, midi_note: u8, gain: f32) {
        self.inner.gain(midi_note, gain)
    }
}

#[cfg(test)]
mod tests {
    use numeric_literals::replace_float_literals;

    use crate::{
        dsp::{buffer::AudioBufferBox, DSPProcessBlock as _},
        Scalar,
    };
    use crate::{
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

        impl<T: Scalar> crate::dsp::DSPProcess<1, 1> for NaiveSquare<T> {
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
