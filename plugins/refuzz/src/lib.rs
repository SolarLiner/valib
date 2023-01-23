use std::sync::{atomic::AtomicBool, Arc};

use nih_plug::prelude::*;
use simba::simd::SimdComplexField;

use valib::{clippers::DiodeClipper, oversample::Oversample, DSP};

const OVERSAMPLE: usize = 4;
const MAX_BLOCK_SIZE: usize = 512;

#[derive(Debug, Enum, Eq, PartialEq)]
enum Fuzz {
    I,
    II,
    III,
}

impl Fuzz {
    #[inline]
    fn get_parameters(&self) -> DiodeClipper<f32> {
        match self {
            Self::I => DiodeClipper::new_led(3, 3, 0.),
            Self::II => DiodeClipper::new_germanium(3, 3, 0.),
            Self::III => DiodeClipper::new_silicon(3, 3, 0.),
        }
    }
}

#[derive(Debug, Params)]
struct ClipperParams {
    #[id = "type"]
    fuzz: EnumParam<Fuzz>,
    #[id = "drive"]
    drive: FloatParam,
    #[id = "tone"]
    tone: FloatParam,
    #[id = "byp"]
    bypass: BoolParam,
}

impl Default for ClipperParams {
    fn default() -> Self {
        Self {
            fuzz: EnumParam::new("Fuzz", Fuzz::I),
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
            tone: FloatParam::new("Tone", 0.5, FloatRange::Linear { min: 0., max: 1. })
                .with_smoother(SmoothingStyle::Linear(50.)),
            bypass: BoolParam::new("Bypass", false),
        }
    }
}

struct Refuzz<const N: usize> {
    params: Arc<ClipperParams>,
    model: [PhysicalModel; N],
    oversample: [Oversample<f32>; N],
}

impl<const N: usize> Default for Refuzz<N> {
    fn default() -> Self {
        let samplerate = 44.1e3 * OVERSAMPLE as f32;
        let force_reset = Arc::new(AtomicBool::new(false));
        Self {
            params: Arc::new(ClipperParams::default()),
            model: std::array::from_fn(|_| PhysicalModel::default()),
            oversample: std::array::from_fn(|_| Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE)),
        }
    }
}

impl<const N: usize> Plugin for Refuzz<N> {
    const NAME: &'static str = "Diode Clipper";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = "0.0.0";
    const DEFAULT_INPUT_CHANNELS: u32 = N as u32;
    const DEFAULT_OUTPUT_CHANNELS: u32 = N as u32;
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        config.num_input_channels == N as u32 && config.num_output_channels == N as u32
    }

    fn reset(&mut self) {
        for os in self.oversample.iter_mut() {
            os.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let samplerate = _context.transport().sample_rate * OVERSAMPLE as f32;
        for model in &mut self.model {
            model.params = self.params.fuzz.value().get_parameters();
        }

        for (_, mut block) in buffer.iter_blocks(MAX_BLOCK_SIZE) {
            let mut drive = [0.; MAX_BLOCK_SIZE];
            let mut tone = [0.; MAX_BLOCK_SIZE];
            let mut drive_os = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
            let mut tone_os = [0.; OVERSAMPLE * MAX_BLOCK_SIZE];
            let len = block.samples();
            self.params
                .drive
                .smoothed
                .next_block_exact(&mut drive[..len]);
            self.params.tone.smoothed.next_block_exact(&mut tone[..len]);
            valib::util::lerp_block(&mut drive_os[..OVERSAMPLE * len], &drive[..len]);
            valib::util::lerp_block(&mut tone_os[..OVERSAMPLE*len], &tone[..len]);
            for ch in 0..block.channels() {
                let buffer = block.get_mut(ch).unwrap();
                let mut os_buffer = self.oversample[ch].oversample(buffer);
                for (i, s) in os_buffer.iter_mut().enumerate() {
                    let drive = drive_os[i];
                    let tone = tone_os[i];
                    let model = &mut self.model[ch];
                    model.rlp = 50. * tone * (330. - 50.);
                    *s = model.process(samplerate, *s * drive);
                }
                os_buffer.finish(buffer);
            }
        }

        // for mut samples in buffer.iter_samples() {
        //     let drive = self.params.drive.smoothed.next();
        //     for (i, s) in samples.iter_mut().enumerate() {
        //         *s = self.dc_couple_in[i].process([*s])[0];
        //         if self.params.model.value() {
        //             *s = self.clipper_model[i].process([*s * drive])[0];
        //         } else {
        //             let [c] = self.clipper_nr[i].process([*s * drive]);
        //             *s = c;
        //         }
        //         *s = self.dc_couple_out[i].process([*s * 2. / drive])[0];
        //     }
        // }

        ProcessStatus::Normal
    }
}

impl<const N: usize> ClapPlugin for Refuzz<N> {
    const CLAP_ID: &'static str = "com.github.SolarLiner.valib.DiodeClipper";
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

impl<const N: usize> Vst3Plugin for Refuzz<N> {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDiodeClpSLN";
    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_clap!(Refuzz::<2>);
nih_export_vst3!(Refuzz::<2>);

struct PhysicalModel {
    rhp: f32,
    chp: f32,
    rlp: f32,
    clp: f32,
    params: DiodeClipper<f32>,
    last_vin: f32,
    last_vout: f32,
    last_dvout: f32,
}

impl Default for PhysicalModel {
    fn default() -> Self {
        Self {
            rhp: 16e3,
            chp: 3e-6,
            rlp: 330.,
            clp: 3e-6,
            params: DiodeClipper::new_led(3, 3, 0.),
            last_vout: 0.,
            last_dvout: 0.,
            last_vin: 0.,
        }
    }
}

impl PhysicalModel {
    fn acceleration(&self, vin: f32, dt: f32) -> f32 {
        let dvin = (vin - self.last_vin) * dt;
        let cir = self.chp * self.params.isat * self.rlp;
        let chv = self.chp * self.params.vt * self.params.n;
        let clp = self.clp * self.params.vt * self.params.n;
        let ivt = self.params.isat * self.params.vt * self.params.n;
        let exp = f32::exp(self.last_vout / (self.params.vt * self.params.n));
        let exp_2 = f32::exp(2. * self.last_vout / (self.params.vt * self.params.n));

        let num = cir * self.last_dvout * (exp + 1.)
            + chv * exp * (dvin - self.last_dvout)
            + clp * exp * self.last_dvout
            + ivt * (exp_2 - 1.);
        let num = num * f32::exp(-self.last_vout / (self.params.vt * self.params.n));
        let den = self.chp * self.clp * self.rlp * self.params.vt * self.params.n;
        num / den
    }

    fn process(&mut self, samplerate: f32, vin: f32) -> f32 {
        let dt = samplerate.recip();
        let acc = self.acceleration(vin, dt);
        self.last_vin = vin;
        self.last_dvout += acc * dt;
        self.last_vout += self.last_dvout * dt;
        self.last_vout
    }
}
