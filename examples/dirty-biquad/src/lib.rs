use std::sync::Arc;

use nih_plug::prelude::*;

use valib::biquad::Biquad;
use valib::clippers::DiodeClipperModel;
use valib::dsp::DSP;
use valib::oversample::Oversample;
use valib::saturators::Dynamic;
use valib::simd::{AutoF32x2, AutoSimd, SimdComplexField, SimdValue};
use valib::util::lerp_block;
use valib::Scalar;

const OVERSAMPLE: usize = 2;
const MAX_BLOCK_SIZE: usize = 512;

type Sample = AutoF32x2;

#[derive(Debug, Enum, Eq, PartialEq)]
enum FilterType {
    Lowpass,
    Bandpass,
    Highpass,
}

#[derive(Debug, Enum, Eq, PartialEq)]
enum NLType {
    Linear,
    Clipped,
    Tanh,
    Diode,
}

impl NLType {
    pub fn as_dynamic_saturator(&self) -> Dynamic<Sample> {
        match self {
            Self::Linear => Dynamic::Linear,
            Self::Clipped => Dynamic::HardClipper,
            Self::Tanh => Dynamic::Tanh,
            Self::Diode => Dynamic::DiodeClipper(DiodeClipperModel::new_silicon(1, 1)),
        }
    }
}

#[derive(Debug, Params)]
struct PluginParams {
    #[id = "fc"]
    fc: FloatParam,
    #[id = "q"]
    q: FloatParam,
    #[id = "drive"]
    drive: FloatParam,
    #[id = "type"]
    filter_type: EnumParam<FilterType>,
    #[id = "nl"]
    nonlinearity: EnumParam<NLType>,
}

impl Default for PluginParams {
    fn default() -> Self {
        Self {
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
            q: FloatParam::new(
                "Q",
                0.5,
                FloatRange::Linear {
                    min: 1e-2,
                    max: 30.,
                },
            )
            .with_smoother(SmoothingStyle::Exponential(50.)),
            drive: FloatParam::new(
                "Drive",
                1.,
                FloatRange::SymmetricalSkewed {
                    min: util::db_to_gain(-36.),
                    max: util::db_to_gain(36.),
                    center: 1.,
                    factor: FloatRange::skew_factor(-2.),
                },
            )
            .with_unit(" dB")
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_smoother(SmoothingStyle::Exponential(100.)),
            filter_type: EnumParam::new("Filter Type", FilterType::Lowpass),
            nonlinearity: EnumParam::new("Nonlinearity", NLType::Linear),
        }
    }
}

#[derive(Debug)]
struct Plugin {
    params: Arc<PluginParams>,
    biquad: Biquad<Sample, Dynamic<Sample>>,
    oversample: Oversample<Sample>,
}

fn get_biquad(params: &PluginParams, samplerate: f32) -> Biquad<Sample, Dynamic<Sample>> {
    let fc = params.fc.value() / samplerate;
    let fc = Sample::splat(fc);
    let q = params.fc.value();
    let q = Sample::splat(q);
    match params.filter_type.value() {
        FilterType::Lowpass => Biquad::lowpass(fc, q),
        FilterType::Bandpass => Biquad::bandpass_peak0(fc, q),
        FilterType::Highpass => Biquad::highpass(fc, q),
    }
}

impl Plugin {
    fn set_filters(&mut self, samplerate: f32) {
        let biquad = get_biquad(&self.params, samplerate);
        let nltype = self.params.nonlinearity.value().as_dynamic_saturator();
        self.biquad.update_coefficients(&biquad);
        self.biquad.set_saturators(nltype, nltype);
    }
}

impl Default for Plugin {
    fn default() -> Self {
        let params = Arc::new(PluginParams::default());
        Self {
            params,
            biquad: Biquad::new([Sample::from([0.0; 2]); 3], [Sample::from([0.0; 2]); 2]),
            oversample: Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE),
        }
    }
}

fn block_to_simd_array<T, const N: usize>(
    block: &mut nih_plug::buffer::Block,
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
    block: &mut nih_plug::buffer::Block,
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

impl nih_plug::prelude::Plugin for Plugin {
    const NAME: &'static str = "Dirty Biquad";
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
    type BackgroundTask = ();
    type SysExMessage = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_config: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.set_filters(buffer_config.sample_rate);
        true
    }

    fn reset(&mut self) {
        self.biquad.reset();
        self.oversample.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        ctx: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let params = self.params.clone();
        let os_samplerate = ctx.transport().sample_rate * OVERSAMPLE as f32;
        let mut fc = [0.; MAX_BLOCK_SIZE];
        let mut q = [0.; MAX_BLOCK_SIZE];
        let mut drive = [0.; MAX_BLOCK_SIZE];
        let mut os_fc = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
        let mut os_q = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
        let mut os_drive = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
        for (_, mut block) in buffer.iter_blocks(MAX_BLOCK_SIZE) {
            let len = block.samples();
            let os_len = OVERSAMPLE * len;
            self.params.fc.smoothed.next_block_exact(&mut fc[..len]);
            self.params.q.smoothed.next_block_exact(&mut q[..len]);
            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
            lerp_block(&mut os_fc[..os_len], &fc[..len]);
            lerp_block(&mut os_q[..os_len], &q[..len]);
            lerp_block(&mut os_drive[..os_len], &drive[..len]);

            let mut simd_block = [Sample::from_f64(0.0); MAX_BLOCK_SIZE];
            let actual_len =
                block_to_simd_array(&mut block, &mut simd_block, std::convert::identity);
            let simd_block = &mut simd_block[..actual_len];
            let mut os_buffer = self.oversample.oversample(simd_block);
            for (i, s) in os_buffer.iter_mut().enumerate() {
                let drive = Sample::splat(os_drive[i]);
                self.biquad.update_coefficients(&get_biquad(&params, os_samplerate));
                let nltype = params.nonlinearity.value().as_dynamic_saturator();
                self.biquad.set_saturators(nltype, nltype);
                *s = self.biquad.process([*s * drive])[0] / drive.simd_asinh();
            }

            os_buffer.finish(simd_block);
            simd_array_to_block(simd_block, &mut block, |&v| v);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Plugin {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.DirtyBiquad";
    const CLAP_DESCRIPTION: Option<&'static str> = None;
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Stereo,
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for Plugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDirTYBiqUAD";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(Plugin);
nih_export_vst3!(Plugin);
