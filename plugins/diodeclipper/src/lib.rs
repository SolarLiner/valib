use std::sync::Arc;

use nih_plug::prelude::*;

use valib::clippers::DiodeClipperModel;
use valib::{biquad::Biquad, clippers::DiodeClipper, saturators::Linear, DSP};

#[derive(Debug, Enum, Eq, PartialEq)]
enum DiodeType {
    Silicon,
    Germanium,
    LED,
}

impl DiodeType {
    #[inline]
    fn get_clipper_model(&self, nf: usize, nb: usize) -> DiodeClipper<f32> {
        match self {
            Self::Silicon => DiodeClipper::new_silicon(nf, nb, 0.),
            Self::Germanium => DiodeClipper::new_germanium(nf, nb, 0.),
            Self::LED => DiodeClipper::new_led(nf, nb, 0.),
        }
    }
    #[inline]
    fn get_model_params(&self, nf: u8, nb: u8) -> DiodeClipperModel<f32> {
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
}

impl Default for ClipperParams {
    fn default() -> Self {
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
            model: BoolParam::new("Use Model", false),
            quality: IntParam::new("Sim Quality", 50, IntRange::Linear { min: 10, max: 100 }),
        }
    }
}

struct ClipperPlugin {
    params: Arc<ClipperParams>,
    clipper_nr: [DiodeClipper<f32>; 2],
    clipper_model: [DiodeClipperModel<f32>; 2],
    dc_couple_in: [Biquad<f32, Linear>; 2],
    dc_couple_out: [Biquad<f32, Linear>; 2],
}

impl Default for ClipperPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(ClipperParams::default()),
            clipper_nr: std::array::from_fn(|_| DiodeClipper::new_germanium(1, 1, 0.)),
            clipper_model: std::array::from_fn(|_| DiodeClipperModel::new_silicon(1, 1)),
            dc_couple_in: std::array::from_fn(|_| Biquad::highpass(3. / 44.1e3, 1.)),
            dc_couple_out: std::array::from_fn(|_| Biquad::highpass(5. / 44.1e3, 1.)),
        }
    }
}

impl Plugin for ClipperPlugin {
    const NAME: &'static str = "Diode Clipper";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = "0.0.0";
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        config.num_input_channels == 2 && config.num_output_channels == 2
    }

    fn initialize(
        &mut self,
        bus_config: &BusConfig,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let samplerate = buffer_config.sample_rate;
        self.dc_couple_in = std::array::from_fn(|_| Biquad::highpass(3. / samplerate, 1.));
        self.dc_couple_out = std::array::from_fn(|_| Biquad::highpass(5. / samplerate, 1.));
        true
    }

    fn reset(&mut self) {
        for ele in self
            .dc_couple_in
            .iter_mut()
            .chain(self.dc_couple_out.iter_mut())
        {
            ele.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for cl in &mut self.clipper_nr {
            *cl = self.params.dtype.value().get_clipper_model(
                self.params.nf.value() as usize,
                self.params.nb.value() as usize,
            );
            cl.max_iter = self.params.quality.value() as usize;
        }
        for cl in &mut self.clipper_model {
            *cl = self
                .params
                .dtype
                .value()
                .get_model_params(self.params.nf.value() as _, self.params.nb.value() as _);
        }

        for mut samples in buffer.iter_samples() {
            let drive = self.params.drive.smoothed.next();
            for (i, s) in samples.iter_mut().enumerate() {
                *s = self.dc_couple_in[i].process([*s])[0];
                if self.params.model.value() {
                    *s = self.clipper_model[i].process([*s * drive])[0];
                } else {
                    let [c] = self.clipper_nr[i].process([*s * drive]);
                    *s = c;
                }
                *s = self.dc_couple_out[i].process([*s * 2. / drive])[0];
            }
        }

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
        ClapFeature::Mono,
    ];
}

impl Vst3Plugin for ClipperPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibDiodeClpSLN";
    const VST3_CATEGORIES: &'static str = "Fx|Filter";
}

nih_export_clap!(ClipperPlugin);
nih_export_vst3!(ClipperPlugin);
