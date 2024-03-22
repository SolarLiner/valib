use std::{collections::VecDeque, ops};

use numeric_literals::replace_float_literals;

use crate::dsp::DSPMeta;
use crate::{dsp::DSPProcess, Scalar};

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
#[allow(unused)]
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
    memory: VecDeque<T>,
    kernel_latency: usize,
}

impl<T: Scalar> Fir<T> {
    pub fn lowpass(fc: T, bandwidth: f64) -> Self {
        let kernel = Vec::from(kernels::windowed_sinc(fc, bandwidth));
        let len = kernel.len();
        Self::new(kernel, len / 2)
    }

    pub fn new(kernel: impl IntoIterator<Item = T>, kernel_latency: usize) -> Self {
        let kernel = Box::from_iter(kernel);
        let memory = VecDeque::from(vec![T::from_f64(0.0); kernel.len()]);
        Self {
            kernel,
            memory,
            kernel_latency,
        }
    }
}

impl<T: Scalar> DSPMeta for Fir<T> {
    type Sample = T;

    fn latency(&self) -> usize {
        self.kernel_latency
    }

    fn reset(&mut self) {
        self.memory = VecDeque::from(vec![T::from_f64(0.0); self.kernel.len()]);
    }
}

impl<T: Scalar> DSPProcess<1, 1> for Fir<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let [x] = x;
        self.memory.pop_front();
        self.memory.push_back(x);
        let y = self
            .kernel
            .iter()
            .copied()
            .zip(self.memory.iter().copied())
            .fold(T::from_f64(0.0), |acc, (a, b)| acc + a * b);
        [y]
    }
}

pub mod kernels {
    use numeric_literals::replace_float_literals;

    use crate::Scalar;

    #[replace_float_literals(T::from_f64(literal))]
    pub fn windowed_sinc_in_place<T: Scalar>(fc: T, slice: &mut [T]) {
        debug_assert_eq!(slice.len() % 2, 1);
        let width = T::from_f64(slice.len() as _);
        let half_width = 0.5 * width;
        for (i, s) in slice.iter_mut().enumerate() {
            let i = T::from_f64(i as _);
            *s = T::simd_sin(T::simd_two_pi() * fc * (i - half_width)) / (i - half_width);
            *s *= 0.42 - 0.5 * T::simd_cos(T::simd_two_pi() * i / width)
                + 0.08 * T::simd_cos(2.0 * T::simd_two_pi() * i / width);
        }
        // Normalization
        let magnitude = slice.iter().copied().fold(0.0, |a, b| a + b);
        for s in slice {
            *s /= magnitude;
        }
    }

    pub fn windowed_sinc<T: Scalar>(fc: T, bandwidth: f64) -> Box<[T]> {
        let mut length = (4.0 / bandwidth) as usize;
        if length % 2 == 0 {
            length += 1;
        }

        let mut data = vec![T::from_f64(0.0); length];
        windowed_sinc_in_place(fc, &mut data[..]);
        data.into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use crate::dsp::buffer::AudioBuffer;
    use crate::dsp::DSPProcessBlock;

    use super::*;

    #[test]
    fn test_fir_direct() {
        let input = Box::from_iter([1.0, 0.0, 0.0, 0.0].into_iter().cycle().take(16));
        let input = AudioBuffer::new([input]).unwrap();
        let mut output = input.clone();
        let mut fir = Fir::new([0.25, 0.5, 0.25], 1);

        output.fill(0.0);
        fir.process_block(input.as_ref(), output.as_mut());
        insta::assert_csv_snapshot!(output.get_channel(0), { "[]" => insta::rounded_redaction(4) })
    }

    #[test]
    fn test_kernel_lowpass() {
        insta::assert_csv_snapshot!(&kernels::windowed_sinc(0.1, 0.1), { "[]" => insta::rounded_redaction(5) });
    }
}
