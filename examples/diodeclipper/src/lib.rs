use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use nih_plug::{
    buffer::Block,
    prelude::*,
};

use valib::clippers::DiodeClipperModel;
use valib::dsp::DSP;
use valib::oversample::Oversample;
use valib::{
    biquad::Biquad,
    clippers::DiodeClipper,
    saturators::Linear,
    simd::{AutoF32x2, AutoF64x2, AutoSimd, SimdComplexField, SimdValue},
    Scalar,
};

#[cfg(debug_assertions)]
const OVERSAMPLE: usize = 4;
#[cfg(not(debug_assertions))]
const OVERSAMPLE: usize = 16;

const MAX_BLOCK_SIZE: usize = 512;

#[derive(Debug, Enum, Eq, PartialEq)]
enum DiodeType {
    Silicon,
    Germanium,
    LED,
}

impl DiodeType {
    #[inline]
    fn get_clipper_model<T: Scalar>(&self, nf: usize, nb: usize) -> DiodeClipper<T> {
        match self {
            Self::Silicon => DiodeClipper::new_silicon(nf, nb, T::from_f64(0.)),
            Self::Germanium => DiodeClipper::new_germanium(nf, nb, T::from_f64(0.)),
            Self::LED => DiodeClipper::new_led(nf, nb, T::from_f64(0.)),
        }
    }
    #[inline]
    fn get_model_params<T: Scalar>(&self, nf: u8, nb: u8) -> DiodeClipperModel<T> {
        match self {
            Self::Silicon => DiodeClipperModel::new_silicon(nf, nb),
            Self::Germanium => DiodeClipperModel::new_germanium(nf, nb),
            Self::LED => DiodeClipperModel::new_led(nf, nb),
        }
    }
}

#[derive(Debug, Params)]
struct ClipperParams {
    #[id = "type"]
    dtype: EnumParam<DiodeType>,
    #[id = "nf"]
    nf: IntParam,
    #[id = "nb"]
    nb: IntParam,
    #[id = "drive"]
    drive: FloatParam,
    #[id = "model"]
    model: BoolParam,
    #[id = "qlty"]
    quality: IntParam,
    #[id = "reset"]
    reset: BoolParam,
}

impl ClipperParams {
    fn new(reset_atomic: Arc<AtomicBool>) -> Self {
        Self {
            dtype: EnumParam::new("Diode", DiodeType::Silicon),
            nf: IntParam::new("# Forward", 1, IntRange::Linear { min: 1, max: 5 }),
            nb: IntParam::new("# Backward", 1, IntRange::Linear { min: 1, max: 5 }),
            drive: FloatParam::new(
                "Drive",
                100.,
                FloatRange::Skewed {
                    min: 1.,
                    max: 300.,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_unit(" dB"),
            model: BoolParam::new("Use Model", true),
            quality: IntParam::new("Sim Quality", 50, IntRange::Linear { min: 10, max: 100 }),
            reset: BoolParam::new("Reset", false).with_callback(Arc::new(move |_| {
                reset_atomic.store(true, Ordering::Release)
            })),
        }
    }
}

type Sample = AutoF32x2;
type Sample64 = AutoF64x2;

struct ClipperPlugin {
    params: Arc<ClipperParams>,
    clipper_nr: DiodeClipper<Sample64>,
    clipper_model: DiodeClipperModel<Sample64>,
    dc_couple_in: Biquad<Sample, Linear>,
    dc_couple_out: Biquad<Sample, Linear>,
    oversample: Oversample<Sample64>,
    force_reset: Arc<AtomicBool>,
}

impl Default for ClipperPlugin {
    fn default() -> Self {
        // let samplerate = 44.1e3 * OVERSAMPLE as f32;
        let force_reset = Arc::new(AtomicBool::new(false));
        Self {
            params: Arc::new(ClipperParams::new(force_reset.clone())),
            clipper_nr: DiodeClipper::new_silicon(1, 1, Sample64::from_f64(0.0)),
            clipper_model: DiodeClipperModel::new_silicon(1, 1),
            dc_couple_in: Biquad::highpass(Sample::from_f64(20. / 44.1e3), Sample::from_f64(1.0)),
            dc_couple_out: Biquad::highpass(Sample::from_f64(20. / 44.1e3), Sample::from_f64(1.0)),
            oversample: Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE),
            force_reset,
        }
    }
}

fn block_to_simd_array<T, const N: usize>(
    block: &mut Block,
    output: &mut [AutoSimd<[T; N]>],
    cast: impl Fn(f32) -> T,
) -> usize {
    let mut i = 0;
    for samples in block.iter_samples().take(output.len()) {
        let mut it = samples.into_iter();
        output[i] = AutoSimd(std::array::from_fn(|_| cast(it.next().copied().unwrap())));
        i += 1;
    }
    i
}

fn simd_array_to_block<T, const N: usize>(
    input: &[AutoSimd<[T; N]>],
    block: &mut Block,
    uncast: impl Fn(&T) -> f32,
) {
    for (inp, mut out_samples) in input.iter().zip(block.iter_samples()) {
        for i in 0..N {
            *out_samples.get_mut(i).unwrap() = uncast(&inp.0[i]);
        }
    }
}

fn apply<P: DSP<1, 1, Sample = AutoSimd<[f32; N]>>, const N: usize>(
    buffer: &mut Buffer,
    dsp: &mut P,
) where
    AutoSimd<[f32; N]>: SimdValue,
    <P::Sample as SimdValue>::Element: Copy,
{
    for mut samples in buffer.iter_samples() {
        let mut it = samples.iter_mut();
        let input = AutoSimd(std::array::from_fn(|_| it.next().copied().unwrap()));
        let [output] = dsp.process([input]);
        for (out, inp) in samples.into_iter().zip(output.0.into_iter()) {
            *out = inp;
        }
    }
}

impl Plugin for ClipperPlugin {
    const NAME: &'static str = "Diode Clipper";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = "0.0.0";
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: Some(new_nonzero_u32(2)),
        main_output_channels: Some(new_nonzero_u32(2)),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames {
            layout: Some("Stereo"),
            main_input: Some("Input"),
            main_output: Some("Output"),
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
        let samplerate = buffer_config.sample_rate as f64;
        self.dc_couple_in =
            Biquad::highpass(Sample::from_f64(20. / samplerate), Sample::from_f64(1.0));
        self.dc_couple_out =
            Biquad::highpass(Sample::from_f64(20. / samplerate), Sample::from_f64(1.0));
        true
    }

    fn reset(&mut self) {
        self.dc_couple_in.reset();
        self.dc_couple_out.reset();
        self.oversample.reset();
        self.clipper_nr.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let clipper_latency = match self.params.model.value() {
            true => self.clipper_model.latency(),
            false => self.clipper_nr.latency(),
        };
        let latency = self.dc_couple_in.latency() + self.dc_couple_out.latency() + self.oversample.latency() + clipper_latency;
        _context.set_latency_samples(latency as _);
        if self.force_reset.load(Ordering::Acquire) {
            self.force_reset.store(false, Ordering::Release);
            self.reset();
        }
        self.clipper_nr = self
            .params
            .dtype
            .value()
            .get_clipper_model(self.params.nf.value() as _, self.params.nb.value() as _);
        self.clipper_nr.max_iter = self.params.quality.value() as _;
        self.clipper_model = self
            .params
            .dtype
            .value()
            .get_model_params(self.params.nf.value() as _, self.params.nb.value() as _);

        apply(buffer, &mut self.dc_couple_in);

        for (_, mut block) in buffer.iter_blocks(MAX_BLOCK_SIZE) {
            let mut drive = [0.; MAX_BLOCK_SIZE];
            let mut drive_os = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
            let len = block.samples();
            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
            valib::util::lerp_block(&mut drive_os[..OVERSAMPLE * len], &drive[..len]);

            let mut simd_buffer = [Sample64::from_f64(0.0); MAX_BLOCK_SIZE];
            let actual_len = block_to_simd_array(&mut block, &mut simd_buffer, |f| f as f64);
            let simd_buffer = &mut simd_buffer[..actual_len];
            let mut os_buffer = self.oversample.oversample(simd_buffer);
            for (i, s) in os_buffer.iter_mut().enumerate() {
                let drive = Sample64::from_f64(drive_os[i] as _);
                let input = *s * drive;
                let [output] = if self.params.model.value() {
                    self.clipper_model.process([input])
                } else {
                    self.clipper_nr.process([input])
                };
                *s = output / drive.simd_asinh();
            }
            os_buffer.finish(simd_buffer);
            simd_array_to_block(simd_buffer, &mut block, |f| *f as f32);
        }

        apply(buffer, &mut self.dc_couple_out);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ClipperPlugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.DiodeClipper";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for ClipperPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDiodeClpSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(ClipperPlugin);
nih_export_vst3!(ClipperPlugin);
