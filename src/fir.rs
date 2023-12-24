use std::ops;

use numeric_literals::replace_float_literals;

use crate::{dsp::{DSPBlock, utils::{mono_block_to_slice, mono_block_to_slice_mut}, PerSampleBlockAdapter}, Scalar};

fn slice_add<T: Copy + ops::Add<T, Output = T>>(in1: &[T], in2: &[T], out: &mut [T]) {
    let len = in1.len().min(in2.len()).min(out.len());
    for i in 0..len {
        out[i] = in1[i] + in2[i];
    }
}

fn slice_add_to<T: Copy + ops::AddAssign<T>>(out: &mut [T], inp: &[T]) {
    for (o, i) in out.iter_mut().zip(inp.iter().copied()) {
        *o += i;
    }
}

fn slice_sub_to<T: Copy + ops::SubAssign<T>>(out: &mut [T], inp: &[T]) {
    for (o, i) in out.iter_mut().zip(inp.iter().copied()) {
        *o -= i;
    }
}

/// Perform arbitrary-length convolution with an O(n^log2(3)) complexity
/// Translated from Taken from https://www.musicdsp.org/en/latest/Filters/65-time-domain-convolution-with-o-n-log2-3.html
/// in1, in2: input signals (of size n)
/// out: output signal (of size 2n)
/// buffer: staging buffer (of size 2n)
#[replace_float_literals(T::from_f64(literal))]
fn convolution<T: Scalar>(in1: &[T], in2: &[T], out: &mut [T], buffer: &mut [T]) {
    let size = in1.len();
    debug_assert_eq!(size, in2.len());
    debug_assert_eq!(out.len(), 2 * size);
    debug_assert_eq!(buffer.len(), 2 * size);

    out.fill(0.0);

    if size == 1 {
        out[0] = in1[0] * in2[0];
        return;
    }

    // first calculate (A1 + A2 * z^-n)*(B1 + B2 * z^-n)*z^n
    {
        let (in11, in12) = in1.split_at(size / 2);
        slice_add(in11, in12, buffer);
    }
    {
        let (in21, in22) = in2.split_at(size / 2);
        slice_add(in21, in22, &mut buffer[size / 2..]);
    }
    {
        let (tmp1, tmp2) = buffer.split_at_mut(size / 2);
        let (tmp2, tmp3) = tmp2.split_at_mut(size);
        convolution(tmp1, tmp2, &mut out[size / 2..], tmp3);
    }

    // then add A1*B1 and substract A1*B1*z^n
    {
        let (tmp1, tmp2) = buffer.split_at_mut(size);
        convolution(
            &in1[..size / 2],
            &in2[..size / 2],
            &mut tmp1[..size / 2],
            &mut tmp2[..size / 2],
        );
        slice_add_to(out, tmp1);
        slice_sub_to(&mut out[size / 2..], &tmp1[..size / 2]);
    }

    // then add A2*B2 and substract A2*B2*z^-n
    {
        let (tmp1, tmp2) = buffer.split_at_mut(size);
        convolution(
            &in1[size / 2..],
            &in2[size / 2..],
            &mut tmp1[..size / 2],
            &mut tmp2[..size / 2],
        );
        slice_add_to(&mut out[size..], tmp1);
        slice_sub_to(&mut out[size / 2..], tmp1);
    }
}

pub struct Fir<T> {
    kernel: Box<[T]>,
    staging_buffer: Box<[T]>,
    output_buffer: Box<[T]>,
    kernel_latency: usize,
    out_index: usize,
}

impl<T: Scalar> Fir<T> {
    pub fn new(kernel: impl IntoIterator<Item = T>, kernel_latency: usize) -> Self {
        let kernel = Box::from_iter(kernel);
        let size = kernel.len();
        let staging_buffer = Box::from_iter(std::iter::repeat(T::from_f64(0.0)).take(2 * size));
        let output_buffer = staging_buffer.clone();

        Self {
            kernel,
            out_index: 0,
            staging_buffer,
            output_buffer,
            kernel_latency,
        }
    }

    pub fn stable_buffers(self) -> PerSampleBlockAdapter<Self, 1, 1> {
        PerSampleBlockAdapter::new(self)
    }
}

impl<T: Scalar> DSPBlock<1, 1> for Fir<T> {
    type Sample = T;

    fn latency(&self) -> usize {
        self.kernel_latency
    }

    fn max_block_size(&self) -> Option<usize> {
        Some(self.kernel.len())
    }

    fn process_block(&mut self, inputs: &[[Self::Sample; 1]], outputs: &mut [[Self::Sample; 1]]) {
        let inputs = mono_block_to_slice(inputs);
        let outputs = mono_block_to_slice_mut(outputs);
        assert_eq!(inputs.len(), self.kernel.len());

        convolution(inputs, &self.kernel, &mut self.output_buffer, &mut self.staging_buffer);
        let size = inputs.len();
        let start = size / 2;
        let _end = start + size;
        outputs.copy_from_slice(&self.output_buffer)
    }
}

#[cfg(test)]
mod tests {
    use crate::dsp::utils::{slice_to_mono_block, slice_to_mono_block_mut};

    use super::*;

    #[test]
    #[should_panic] // Block-based convolution is bugged
    fn test_fir_direct() {
        let input = Box::from_iter([1.0, 0.0, 0.0, 0.0].into_iter().cycle().take(16));
        let mut output = input.clone();
        let mut fir = Fir::new([0.25, 0.5, 0.25], 1).stable_buffers();

        output.fill(0.0);
        fir.process_block(slice_to_mono_block(&input), slice_to_mono_block_mut(&mut output));
        insta::assert_csv_snapshot!(&output, { "[]" => insta::rounded_redaction(4) })
    }
}