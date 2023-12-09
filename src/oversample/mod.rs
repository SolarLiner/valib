use crate::biquad::Biquad;
use crate::saturators::Linear;
use crate::Scalar;
use std::ops::{Deref, DerefMut};
use crate::dsp::DSP;

#[derive(Debug, Clone)]
pub struct Oversample<T> {
    os_factor: usize,
    os_buffer: Vec<T>,
    // TODO: Implement Series/Parallel for arrays and replace this
    pre_filter: [Biquad<T, Linear>; 8],
    post_filter: [Biquad<T, Linear>; 8],
    // pre_filter: Biquad<T, Linear>,
    // post_filter: Biquad<T, Linear>,
}

impl<T: Scalar> Oversample<T> {
    pub fn new(os_factor: usize, max_block_size: usize) -> Self {
        assert!(os_factor > 1);
        let os_buffer = vec![T::zero(); max_block_size * os_factor];
        let filters = std::array::from_fn(|_| Biquad::lowpass(
            T::from_f64(2.0 * os_factor as f64).simd_recip(),
            T::from_f64(0.707),
        ));
        Self {
            os_factor,
            os_buffer,
            pre_filter: filters,
            post_filter: filters,
        }
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
        // self.pre_filter.reset();
        // self.post_filter.reset();
        for f in self
            .pre_filter
            .iter_mut()
            .chain(self.post_filter.iter_mut())
        {
            f.reset();
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

#[cfg(test)]
mod tests {
    use super::Oversample;
    use std::{f32::consts::TAU, hint::black_box};

    #[test]
    fn oversample_no_dc_offset() {
        let _csv = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path("oversample.tsv")
            .unwrap();
        let inp: [f32; 512] = std::array::from_fn(|i| (TAU * i as f32 / 64.).sin());
        let mut os = Oversample::new(4, 512);
        let osblock = black_box(os.oversample(&inp));
        insta::assert_csv_snapshot!(&*osblock);
    }
}
