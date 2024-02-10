#![feature(array_methods)]
#![feature(const_trait_impl)]

extern crate core;

use crate::{
    filter::{Filter, FilterParams},
    spectrum::Analyzer,
};
use atomic_float::AtomicF32;
use nih_plug::{params::persist::PersistentField, prelude::*};
use std::sync::{Arc, Mutex};
use valib::dsp::blocks::Series;
use valib::dsp::utils::{slice_to_mono_block, slice_to_mono_block_mut};
use valib::dsp::DSPBlock;
use valib::simd::{AutoF32x2, SimdValue};
use valib::Scalar;

pub mod editor;
mod filter;
mod spectrum;

pub const NUM_BANDS: usize = 5;
pub const OVERSAMPLE: usize = 2;

type Sample = AutoF32x2;

#[cfg(not(feature = "example"))]
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

#[cfg(not(feature = "example"))]
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
            )
            .with_unit("ms"),
        }
    }
}

#[cfg(not(feature = "example"))]
pub struct Abrasive<const N: usize> {
    params: Arc<AbrasiveParams<N>>,
    filters: Series<[Filter; N]>,
    samplerate: Arc<AtomicF32>,
    analyzer_in: Analyzer,
    analyzer_input: editor::SpectrumUI,
    analyzer_out: Analyzer,
    analyzer_output: editor::SpectrumUI,
}

#[cfg(not(feature = "example"))]
impl<const N: usize> Default for Abrasive<N> {
    fn default() -> Self {
        let params = AbrasiveParams::default();
        let filters = params
            .params
            .each_ref()
            .map(|p| Filter::new(44.1e3, p.clone()));
        let (analyzer_in, spectrum_in) = spectrum::Analyzer::new(44.1e3, 2, 2048);
        let analyzer_input = Arc::new(Mutex::new(spectrum_in));
        let (analyzer_out, spectrum_out) = spectrum::Analyzer::new(44.1e3, 2, 2048);
        let analyzer_output = Arc::new(Mutex::new(spectrum_out));
        let samplerate = Arc::new(AtomicF32::new(44.1e3));
        Self {
            params: Arc::new(params),
            filters: Series(filters),
            samplerate,
            analyzer_in,
            analyzer_input,
            analyzer_out,
            analyzer_output,
        }
    }
}

#[cfg(not(feature = "example"))]
impl Plugin for Abrasive<NUM_BANDS> {
    const NAME: &'static str = "Abrasive";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/solarliner/abrasive";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    type BackgroundTask = ();
    type SysExMessage = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
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
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate;
        self.samplerate.set(sr);
        self.analyzer_in.set_samplerate(sr);
        self.analyzer_out.set_samplerate(sr);
        self.analyzer_in
            .set_decay(self.params.analyzer_smooth.value());
        self.analyzer_out
            .set_decay(self.params.analyzer_smooth.value());
        self.set_filterbank_samplerate(sr);
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        _context.set_latency_samples(self.filters.latency() as _);
        self.analyzer_in
            .set_decay(self.params.analyzer_smooth.value());
        self.analyzer_out
            .set_decay(self.params.analyzer_smooth.value());
        self.analyzer_in.process_buffer(buffer);
        self.process_filter_bank::<256>(buffer);
        self.analyzer_out.process_buffer(buffer);
        ProcessStatus::Normal
    }

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }];
}

#[cfg(not(feature = "example"))]
impl Abrasive<NUM_BANDS> {
    fn set_filterbank_samplerate(&mut self, _sr: f32) {
        for filter in self.filters.0.iter_mut() {
            filter.reset();
        }
    }

    fn process_filter_bank<const BLOCK_SIZE: usize>(&mut self, buffer: &mut Buffer) {
        let mut drive = [0.; BLOCK_SIZE];
        let mut scale = drive;

        for (_, mut block) in buffer.iter_blocks(BLOCK_SIZE) {
            let len = block.samples();

            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
            self.params
                .scale
                .smoothed
                .next_block_exact(&mut scale[..len]);
            let mut simd_input = [Sample::from_f64(0.0); BLOCK_SIZE];
            let mut simd_output = simd_input;
            for (i, mut samples) in block.iter_samples().enumerate() {
                simd_input[i] =
                    Sample::from([*samples.get_mut(0).unwrap(), *samples.get_mut(1).unwrap()]);
                simd_input[i] *= Sample::splat(drive[i]);
            }

            self.filters.process_block(
                slice_to_mono_block(&simd_input[..len]),
                slice_to_mono_block_mut(&mut simd_output[..len]),
            );

            for (i, mut samples) in block.iter_samples().enumerate() {
                let drive = Sample::splat(drive[i]);
                simd_output[i] /= drive;
                *samples.get_mut(0).unwrap() = simd_output[i].extract(0);
                *samples.get_mut(1).unwrap() = simd_output[i].extract(1);
            }
        }
    }
}

#[cfg(not(feature = "example"))]
impl ClapPlugin for Abrasive<NUM_BANDS> {
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

#[cfg(not(feature = "example"))]
impl Vst3Plugin for Abrasive<NUM_BANDS> {
    const VST3_CLASS_ID: [u8; 16] = *b"ValibAbrasiveSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Analyzer,
        Vst3SubCategory::Stereo,
    ];
}

#[cfg(not(feature = "example"))]
nih_export_clap!(Abrasive<NUM_BANDS>);
#[cfg(not(feature = "example"))]
nih_export_vst3!(Abrasive<NUM_BANDS>);
