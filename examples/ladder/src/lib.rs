use std::sync::{atomic::AtomicBool, Arc};

use nih_plug::prelude::*;

use valib::math::interpolation::{Cubic, Interpolate};
use valib::oversample::Oversample;
use valib::{dsp::DSP, Scalar};
use valib::{
    filters::ladder::{Ideal, Ladder, Transistor, OTA},
    saturators::{clippers::DiodeClipperModel, Tanh},
    simd::{AutoF32x2, SimdValue},
};

const MAX_BUFFER_SIZE: usize = 512;
const OVERSAMPLE: usize = 2;

#[derive(Debug, Default, Enum, Clone, Copy, PartialEq, Eq)]
enum LadderTopology {
    #[default]
    Ideal,
    Ota,
    Transistor,
}

#[derive(Debug, Params)]
struct PluginParams {
    #[id = "drive"]
    drive: FloatParam,
    #[id = "fc"]
    fc: FloatParam,
    #[id = "q"]
    q: FloatParam,
    #[id = "topo"]
    topology: EnumParam<LadderTopology>,
    #[id = "comp"]
    compensated: BoolParam,
}

impl PluginParams {
    fn new(topology_changed: Arc<AtomicBool>) -> Self {
        Self {
            drive: FloatParam::new(
                "Drive",
                1.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 100.0,
                    factor: FloatRange::gain_skew_factor(0.0, 40.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_smoother(SmoothingStyle::Exponential(10.)),
            fc: FloatParam::new(
                "Frequency",
                300.,
                FloatRange::Skewed {
                    min: 20.,
                    max: 20e3,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_hz_then_khz_with_note_name(2, false))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_smoother(SmoothingStyle::Exponential(10.)),
            q: FloatParam::new("Q", 0.5, FloatRange::Linear { min: 0., max: 1.25 })
                .with_smoother(SmoothingStyle::Exponential(50.)),
            topology: EnumParam::new("Topology", LadderTopology::default()).with_callback(
                Arc::new(move |_| {
                    topology_changed.store(true, std::sync::atomic::Ordering::SeqCst)
                }),
            ),
            compensated: BoolParam::new("Compensated", true),
        }
    }
}

type Sample = AutoF32x2;
type DspT<Topo> = Ladder<Sample, Topo>;
type DspIdeal = DspT<Ideal>;
type DspOta = DspT<OTA<Tanh>>;
type DspTransistor = DspT<Transistor<DiodeClipperModel<Sample>>>;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
enum Dsp {
    Ideal(DspIdeal),
    Ota(DspOta),
    Transistor(DspTransistor),
}
impl Dsp {
    fn set_params(&mut self, freq: Sample, q: Sample) {
        match self {
            Self::Ideal(f) => {
                f.set_cutoff(freq);
                f.set_resonance(q);
            }
            Self::Ota(f) => {
                f.set_cutoff(freq);
                f.set_resonance(q);
            }
            Self::Transistor(f) => {
                f.set_cutoff(freq);
                f.set_resonance(q);
            }
        }
    }
}

impl DSP<1, 1> for Dsp {
    type Sample = Sample;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        match self {
            Self::Ideal(f) => f.process(x),
            Self::Ota(f) => f.process(x),
            Self::Transistor(f) => f.process(x),
        }
    }
}

#[derive(Debug)]
struct LadderFilterPlugin {
    topology_changed: Arc<AtomicBool>,
    params: Arc<PluginParams>,
    oversample: Oversample<Sample>,
    dsp: Dsp,
}
impl LadderFilterPlugin {
    fn setup_filter(&mut self, samplerate: impl Copy + Into<f64>) {
        if self
            .topology_changed
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            let fc = Sample::splat(self.params.fc.value());
            let q = Sample::splat(self.params.q.value());
            self.dsp = match self.params.topology.value() {
                LadderTopology::Ideal => Dsp::Ideal(DspIdeal::new(samplerate, fc, q)),
                LadderTopology::Ota => Dsp::Ota(DspOta::new(samplerate, fc, q)),
                LadderTopology::Transistor => {
                    Dsp::Transistor(DspTransistor::new(samplerate, fc, q))
                }
            };
            self.topology_changed
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let compensated = match &mut self.dsp {
            Dsp::Ideal(f) => &mut f.compensated,
            Dsp::Ota(f) => &mut f.compensated,
            Dsp::Transistor(f) => &mut f.compensated,
        };
        *compensated = self.params.compensated.value();
    }
}

impl Default for LadderFilterPlugin {
    fn default() -> Self {
        let topology_changed = Arc::new(AtomicBool::new(false));
        let params = Arc::new(PluginParams::new(topology_changed.clone()));
        let fc = Sample::splat(params.fc.default_plain_value());
        let q = Sample::splat(params.q.default_plain_value());
        let dsp = DspIdeal::new(44100.0, fc, q);
        Self {
            topology_changed,
            params,
            dsp: Dsp::Ideal(dsp),
            oversample: Oversample::new(OVERSAMPLE, MAX_BUFFER_SIZE),
        }
    }
}

impl Plugin for LadderFilterPlugin {
    const NAME: &'static str = "Ladder filter";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(2)),
        main_output_channels: Some(new_nonzero_u32(2)),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames {
            layout: Some("Stereo"),
            main_input: Some("input"),
            main_output: Some("output"),
            aux_inputs: &[],
            aux_outputs: &[],
        },
    }];
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.setup_filter(buffer_config.sample_rate * OVERSAMPLE as f32);
        true
    }

    fn reset(&mut self) {
        self.dsp.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.setup_filter(context.transport().sample_rate * OVERSAMPLE as f32);

        let mut drive = [0.; MAX_BUFFER_SIZE];
        let mut fc = [0.; MAX_BUFFER_SIZE];
        let mut q = [0.; MAX_BUFFER_SIZE];
        let mut simd_slice = [Sample::from_f64(0.0); MAX_BUFFER_SIZE];

        for (_, mut block) in buffer.iter_blocks(MAX_BUFFER_SIZE) {
            for (i, mut sample) in block.iter_samples().enumerate() {
                simd_slice[i] = Sample::new(
                    sample.get_mut(0).copied().unwrap(),
                    sample.get_mut(1).copied().unwrap(),
                );
            }
            let len = block.samples();
            let os_len = OVERSAMPLE * len;

            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
            self.params.fc.smoothed.next_block_exact(&mut fc[..len]);
            self.params.q.smoothed.next_block_exact(&mut q[..len]);

            let mut os_drive = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_fc = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];
            let mut os_q = [0.; OVERSAMPLE * MAX_BUFFER_SIZE];

            Cubic::interpolate_slice(&mut os_drive[..os_len], &drive[..len]);
            Cubic::interpolate_slice(&mut os_fc[..os_len], &fc[..len]);
            Cubic::interpolate_slice(&mut os_q[..os_len], &q[..len]);

            let buffer = &mut simd_slice[..len];
            let mut os_buffer = self.oversample.oversample(buffer);
            for (i, s) in os_buffer.iter_mut().enumerate() {
                let fc = Sample::splat(os_fc[i]);
                let q = Sample::splat(4.0 * os_q[i]);
                self.dsp.set_params(fc, q);
                let drive = Sample::splat(os_drive[i]);
                *s = self.dsp.process([*s * drive])[0] / drive;
            }
            os_buffer.finish(buffer);

            for (i, mut s) in block.iter_samples().enumerate() {
                *s.get_mut(0).unwrap() = buffer[i].extract(0);
                *s.get_mut(1).unwrap() = buffer[i].extract(1);
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for LadderFilterPlugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.LadderFilter";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for LadderFilterPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibLaddrFltSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(LadderFilterPlugin);
nih_export_vst3!(LadderFilterPlugin);
