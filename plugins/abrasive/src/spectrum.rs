use nih_plug::prelude::*;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::fmt::{self, Formatter};
use std::sync::Arc;
use nih_plug::util::window::multiply_with_window;
use triple_buffer::{Input, Output, TripleBuffer};

#[derive(Clone)]
pub struct Spectrum {
    pub window_size: usize,
    pub data: Box<[f32]>,
}

impl Spectrum {
    fn new(window_size: usize) -> Spectrum {
        Self {
            window_size,
            data: vec![0.; window_size/2+1].into_boxed_slice(),
        }
    }
}

impl Spectrum {
    pub fn updating(mut self, f: impl FnOnce(usize, &mut [f32])) -> Self {
        self.update(f);
        self
    }

    pub fn update(&mut self, f: impl FnOnce(usize, &mut [f32]) + Sized) {
        f(self.window_size, &mut self.data);
    }
}

impl fmt::Debug for Spectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Spectrum")
            .field("window_size", &self.window_size)
            .finish_non_exhaustive()
    }
}

pub struct SpectrumInput {
    stft: util::StftHelper,
    channels: usize,
    input: Input<Spectrum>,
    spectrum_scratch: Spectrum,
    smoothing_weight: f32,
    plan: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    fft_buffer: Vec<Complex32>,
    window_size: usize,
}

const STFT_OVERLAP: usize = 2;

impl SpectrumInput {
    pub fn new(channels: usize, window_size: usize) -> (SpectrumInput, Output<Spectrum>) {
        let (buf_in, buf_out) = TripleBuffer::new(&Spectrum {
            window_size,
            data: vec![0.; window_size / 2 + 1].into_boxed_slice(),
        })
        .split();
        let input = Self {
            stft: util::StftHelper::new(channels, window_size, 0),
            channels,
            smoothing_weight: 0.,
            spectrum_scratch: Spectrum::new(window_size),
            input: buf_in,
            plan: RealFftPlanner::new().plan_fft_forward(window_size),
            window: util::window::hann(window_size)
                .into_iter()
                .map(|x| x / window_size as f32)
                .collect(),
            fft_buffer: vec![Complex32::default(); window_size / 2 + 1],
            window_size,
        };

        (input, buf_out)
    }

    pub fn update_smoothing(&mut self, samplerate: f32, ms: f32) {
        let actual_sr = samplerate / self.window_size as f32 * STFT_OVERLAP as f32 * self.channels as f32;
        let decay = (ms / 1e3 * actual_sr) as f64;
        self.smoothing_weight = 0.25f64.powf(decay.recip()) as f32;
    }

    pub fn next_block(&mut self, buffer: &Buffer) {
        self.stft.process_analyze_only(buffer, STFT_OVERLAP, |_, buffer| {
            multiply_with_window(buffer, &self.window);
            self.plan.process_with_scratch(buffer, &mut self.fft_buffer, &mut []).unwrap();
            for (bin, result) in self.fft_buffer.iter().zip(&mut *self.spectrum_scratch.data) {
                let r = bin.norm();
                if r > *result {
                    *result = r;
                } else {
                    *result = *result * self.smoothing_weight + r * (1. - self.smoothing_weight);
                }
                // *result = result.max(*result * self.smoothing_weight + r * (1. - self.smoothing_weight));
            }
        });

        self.input.input_buffer().data.copy_from_slice(&self.spectrum_scratch.data);
        self.input.publish();
    }
}
