#![feature(array_methods)]
#![feature(const_trait_impl)]

extern crate core;

use crate::filter::{Filter, FilterParams};
use crate::spectrum::Analyzer;
use atomic_float::AtomicF32;
use nih_plug::params::persist::PersistentField;
use nih_plug::prelude::*;
use std::sync::{Arc, Mutex};

mod editor;
mod filter;
mod spectrum;

#[derive(Debug, Params)]
struct AbrasiveParams<const N: usize> {
    #[id = "drive"]
    drive: FloatParam,
    #[id = "scale"]
    scale: FloatParam,
    #[nested(array)]
    params: [Arc<FilterParams>; N],
    #[id = "spsm"]
    analyzer_smooth: FloatParam,
}

impl<const N: usize> Default for AbrasiveParams<N> {
    fn default() -> Self {
        Self {
            drive: FloatParam::new(
                "Drive",
                1.,
                FloatRange::Skewed {
                    min: 1e-2,
                    max: 100.,
                    factor: FloatRange::gain_skew_factor(-40., 40.),
                },
            )
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_unit("dB")
            .with_smoother(SmoothingStyle::Exponential(100.)),
            scale: FloatParam::new("Scale", 1., FloatRange::Linear { min: -1., max: 1. })
                .with_string_to_value(formatters::s2v_f32_percentage())
                .with_value_to_string(formatters::v2s_f32_percentage(2))
                .with_smoother(SmoothingStyle::Exponential(100.))
                .with_unit("%"),
            params: std::array::from_fn(|_| Default::default()),
            analyzer_smooth: FloatParam::new(
                "UI Analyzer smoothing",
                2500.,
                FloatRange::Skewed {
                    min: 1.,
                    max: 10e3,
                    factor: FloatRange::skew_factor(-1.5),
                },
            ).with_unit("ms"),
        }
    }
}

struct Abrasive<const CHANNELS: usize, const N: usize> {
    params: Arc<AbrasiveParams<N>>,
    filters: [Filter<CHANNELS>; N],
    samplerate: Arc<AtomicF32>,
    analyzer_in: Analyzer,
    analyzer_input: editor::SpectrumUI,
    analyzer_out: Analyzer,
    analyzer_output: editor::SpectrumUI,
}

impl<const CHANNELS: usize, const N: usize> Default for Abrasive<CHANNELS, N> {
    fn default() -> Self {
        let params = AbrasiveParams::default();
        let filters = params
            .params
            .each_ref()
            .map(|p| Filter::new(44.1e3, p.clone()));
        let (analyzer_in, spectrum_in) = spectrum::Analyzer::new(44.1e3, CHANNELS, 2048);
        let analyzer_input = Arc::new(Mutex::new(spectrum_in));
        let (analyzer_out, spectrum_out) = spectrum::Analyzer::new(44.1e3, CHANNELS, 2048);
        let analyzer_output = Arc::new(Mutex::new(spectrum_out));
        let samplerate = Arc::new(AtomicF32::new(44.1e3));
        Self {
            params: Arc::new(params),
            filters,
            samplerate,
            analyzer_in,
            analyzer_input,
            analyzer_out,
            analyzer_output,
        }
    }
}

impl<const CHANNELS: usize> Plugin for Abrasive<CHANNELS, 2> {
    const NAME: &'static str = "Abrasive";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/solarliner/abrasive";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const DEFAULT_INPUT_CHANNELS: u32 = CHANNELS as _;
    const DEFAULT_OUTPUT_CHANNELS: u32 = CHANNELS as _;
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&self, _: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            editor::Data {
                spectrum_out: self.analyzer_output.clone(),
                spectrum_in: self.analyzer_input.clone(),
                samplerate: self.samplerate.clone(),
                params: self.params.clone(),
                selected: None,
            },
            editor::default_state(),
        )
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate;
        self.samplerate.set(sr);
        self.analyzer_in.set_samplerate(sr);
        self.analyzer_out.set_samplerate(sr);
        self.analyzer_in.set_decay(self.params.analyzer_smooth.value());
        self.analyzer_out.set_decay(self.params.analyzer_smooth.value());
        self.set_filterbank_samplerate(sr);
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.analyzer_in.set_decay(self.params.analyzer_smooth.value());
        self.analyzer_out.set_decay(self.params.analyzer_smooth.value());
        self.analyzer_in.process_buffer(buffer);
        self.process_filter_bank::<256>(buffer);
        self.analyzer_out.process_buffer(buffer);
        ProcessStatus::Normal
    }
}

impl<const CHANNELS: usize> Abrasive<CHANNELS, 2> {
    fn set_filterbank_samplerate(&mut self, sr: f32) {
        for filter in self.filters.iter_mut() {
            filter.reset(sr);
        }
    }

    fn process_filter_bank<const BLOCK_SIZE: usize>(&mut self, buffer: &mut Buffer) {
        let mut drive = [0.; BLOCK_SIZE];
        let mut scale = drive;

        self.params.drive.smoothed.next_block_exact(&mut drive);
        self.params.scale.smoothed.next_block_exact(&mut scale);

        for (_, mut block) in buffer.iter_blocks(BLOCK_SIZE) {
            for (samples, drive) in block.iter_samples().zip(drive) {
                for sample in samples.into_iter() {
                    *sample *= drive;
                }
            }

            for filt in &mut self.filters {
                filt.process_block::<BLOCK_SIZE>(&mut block, scale);
            }
            for (samples, drive) in block.iter_samples().zip(drive) {
                for sample in samples {
                    if !sample.is_finite() {
                        // Set sample to zero on infinities
                        *sample = 0.;
                    }
                    *sample /= drive;
                }
            }
        }
    }
}

impl<const CHANNELS: usize> ClapPlugin for Abrasive<CHANNELS, 2> {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.Abrasive";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Configurable colorful parametric equalizer");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Stereo,
        ClapFeature::Filter,
        ClapFeature::Equalizer,
    ];
}

impl<const CHANNELS: usize> Vst3Plugin for Abrasive<CHANNELS, 2> {
    const VST3_CLASS_ID: [u8; 16] = *b"ValibAbrasiveSLN";
    const VST3_CATEGORIES: &'static str = "Fx|Filter|Equalizer";
}

nih_export_clap!(Abrasive<2, 2>);
nih_export_vst3!(Abrasive<2, 2>);
