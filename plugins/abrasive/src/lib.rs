extern crate core;

use crate::spectrum::Analyzer;
use atomic_float::AtomicF32;
use nih_plug::{params::persist::PersistentField, prelude::*};
use serde::{Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use valib::contrib::nih_plug::process_buffer_simd;

use crate::dsp::{DspParams, OVERSAMPLE};
use valib::dsp::DSPMeta;
use valib::preset_manager::data::PresetData;
use valib::simd::AutoF32x2;

mod dsp;
mod editor;
mod spectrum;

pub const MAX_BLOCK_SIZE: usize = 64;

#[derive(Debug, Params)]
struct AbrasiveParams {
    #[nested]
    dsp_params: Arc<DspParams>,
    #[id = "spsm"]
    analyzer_smooth: FloatParam,
}

impl Default for AbrasiveParams {
    fn default() -> Self {
        Self {
            dsp_params: Arc::default(),
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AbrasiveParamsSerialized(BTreeMap<String, String>);

impl PresetData for AbrasiveParamsSerialized {
    const CURRENT_REVISION: u64 = 0;
    type PreviousRevision = ();
}

type Sample = AutoF32x2;

pub struct Abrasive {
    params: Arc<AbrasiveParams>,
    dsp: dsp::Dsp<Sample>,
    samplerate: Arc<AtomicF32>,
    analyzer_in: Analyzer,
    analyzer_input: editor::SpectrumUI,
    analyzer_out: Analyzer,
    analyzer_output: editor::SpectrumUI,
}

impl Default for Abrasive {
    fn default() -> Self {
        const DEFAULT_SAMPLERATE: f64 = 44100.0;
        let params = AbrasiveParams::default();
        let dsp = dsp::create(DEFAULT_SAMPLERATE as _, params.dsp_params.clone());
        let (analyzer_in, spectrum_in) = Analyzer::new(44.1e3, 2, 2048);
        let analyzer_input = Arc::new(Mutex::new(spectrum_in));
        let (analyzer_out, spectrum_out) = Analyzer::new(44.1e3, 2, 2048);
        let analyzer_output = Arc::new(Mutex::new(spectrum_out));
        let samplerate = Arc::new(AtomicF32::new(44.1e3));
        Self {
            params: Arc::new(params),
            dsp,
            samplerate,
            analyzer_in,
            analyzer_input,
            analyzer_out,
            analyzer_output,
        }
    }
}

impl Plugin for Abrasive {
    const NAME: &'static str = "Abrasive";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/solarliner/abrasive";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }];
    type SysExMessage = ();

    type BackgroundTask = ();

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
                show_save_dialog: false,
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
        self.dsp.set_samplerate(sr);
        self.samplerate
            .store(OVERSAMPLE as f32 * sr, Ordering::SeqCst);
        self.analyzer_in.set_samplerate(sr);
        self.analyzer_out.set_samplerate(sr);
        self.analyzer_in
            .set_decay(self.params.analyzer_smooth.value());
        self.analyzer_out
            .set_decay(self.params.analyzer_smooth.value());
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        _context.set_latency_samples(self.dsp.latency() as _);
        self.analyzer_in
            .set_decay(self.params.analyzer_smooth.value());
        self.analyzer_out
            .set_decay(self.params.analyzer_smooth.value());
        self.analyzer_in.process_buffer(buffer);
        process_buffer_simd::<_, _, MAX_BLOCK_SIZE>(&mut self.dsp, buffer);
        self.analyzer_out.process_buffer(buffer);
        ProcessStatus::Normal
    }
}

impl ClapPlugin for Abrasive {
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

impl Vst3Plugin for Abrasive {
    const VST3_CLASS_ID: [u8; 16] = *b"ValibAbrasiveSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Eq,
        Vst3SubCategory::Analyzer,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(Abrasive);
nih_export_vst3!(Abrasive);
