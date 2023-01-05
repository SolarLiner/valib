use std::{
    cell::Cell,
    fmt::{self, Formatter},
    marker::Destruct,
    sync::{atomic::Ordering::Relaxed, Arc},
};

use atomic_float::AtomicF32;
use nih_plug::{
    prelude::*,
    util::{window::multiply_with_window, StftHelper},
};
use realfft::{num_complex::Complex32, num_traits::Zero, RealFftPlanner, RealToComplex};
use triple_buffer::{Input, Output, TripleBuffer};

pub struct Spectrum {
    pub window_size: usize,
    pub samplerate: f32,
    pub data: Box<[f32]>,
}

impl Clone for Spectrum {
    fn clone(&self) -> Self {
        let mut this = Self::new(self.window_size, self.samplerate);
        this.data.copy_from_slice(&self.data);
        this
    }

    fn clone_from(&mut self, source: &Self)
    where
        Self: ~const Destruct,
    {
        self.window_size = source.window_size;
        self.samplerate = source.samplerate;
        self.data.copy_from_slice(&source.data);
    }
}

impl Spectrum {
    fn new(window_size: usize, samplerate: f32) -> Spectrum {
        Self {
            window_size,
            samplerate,
            data: vec![0.; window_size / 2 + 1].into_boxed_slice(),
        }
    }
}

impl fmt::Debug for Spectrum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Spectrum")
            .field("window_size", &self.window_size)
            .finish_non_exhaustive()
    }
}

pub struct Analyzer {
    stft: StftHelper,
    input: Input<Spectrum>,
    scratch: Spectrum,
    samplerate: Arc<AtomicF32>,
    plan: Arc<dyn RealToComplex<f32>>,
    fft_buffer: Vec<Complex32>,
    window: Vec<f32>,
    decay: Cell<f32>,
}

impl Analyzer {
    pub fn new(
        samplerate: f32,
        num_channels: usize,
        window_size: usize,
    ) -> (Self, Output<Spectrum>) {
        let scratch = Spectrum::new(window_size, samplerate);
        let (input, output) = TripleBuffer::new(&scratch).split();
        let this = Self {
            stft: StftHelper::new(num_channels, window_size, 0),
            input,
            scratch,
            samplerate: Arc::new(AtomicF32::new(samplerate)),
            plan: RealFftPlanner::new().plan_fft_forward(window_size),
            fft_buffer: vec![Complex32::zero(); window_size / 2 + 1],
            window: util::window::hann(window_size)
                .into_iter()
                .map(|x| x / window_size as f32)
                .collect(),
            decay: Cell::new(100e-3),
        };
        (this, output)
    }

    pub fn set_samplerate(&self, samplerate: f32) {
        self.samplerate.store(samplerate, Relaxed);
    }

    pub fn set_window_size(&mut self, window_size: usize) {
        self.stft.set_block_size(window_size);
        self.plan = RealFftPlanner::new().plan_fft_forward(window_size);
        self.fft_buffer
            .resize(window_size / 2 + 1, Complex32::zero());
        self.window = util::window::hann(window_size)
            .into_iter()
            .map(|x| x / window_size as f32)
            .collect();
    }

    pub fn set_decay(&self, ms: f32) {
        self.decay.set(ms * 1e-3);
    }

    pub fn process_buffer(&mut self, buffer: &Buffer) {
        self.scratch.samplerate = self.samplerate.load(Relaxed);
        self.stft.process_analyze_only(buffer, 2, |_, buffer| {
            multiply_with_window(buffer, &self.window);
            if let Err(_) = self
                .plan
                .process_with_scratch(buffer, &mut self.fft_buffer, &mut [])
            {
                self.fft_buffer.fill(Complex32::zero());
            }
            for (scratch, fft) in self
                .scratch
                .data
                .iter_mut()
                .zip(self.fft_buffer.iter_mut().map(|c| c.norm()))
            {
                // let mix = 1.
                //     - f32::exp(
                //         -self.scratch.samplerate / self.window.len() as f32 / 2. * self.decay.get(),
                //     );
                let decay = f32::ln(1e-3) / self.decay.get();
                let mix = f32::exp(decay * 1024. / self.scratch.samplerate);
                *scratch = lerp(mix, fft, *scratch).max(fft);
            }
        });
        self.input.input_buffer().clone_from(&self.scratch);
        self.input.publish();
    }
}

fn lerp(t: f32, a: f32, b: f32) -> f32 {
    a + (b - a) * t
}
