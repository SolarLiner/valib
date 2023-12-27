use std::sync::Arc;

use nih_plug::{prelude::*, util::db_to_gain};
use valib::{
    dsp::DSP,
    saturators::Slew,
    simd::{AutoF32x2, SimdValue},
};

type Sample = AutoF32x2;

#[derive(Debug, Params)]
struct SlewParams {
    #[id="rate"]
    rate: FloatParam,
}

impl Default for SlewParams {
    fn default() -> Self {
        Self {
            rate: FloatParam::new(
                "Rate",
                1.0,
                FloatRange::Skewed {
                    min: db_to_gain(-20.0),
                    max: db_to_gain(100.0),
                    factor: FloatRange::gain_skew_factor(-20.0, 100.0),
                },
            )
            .with_unit("dB")
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2)),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct SlewPlugin {
    params: Arc<SlewParams>,
    dsp: Slew<Sample>,
}

impl Plugin for SlewPlugin {
    const NAME: &'static str = "Slew";

    const VENDOR: &'static str = "SolarLiner";

    const URL: &'static str = "https://github.com/solarliner/valib";

    const EMAIL: &'static str = "me@solarliner.dev";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        aux_input_ports: &[],
        aux_output_ports: &[],
        main_input_channels: Some(new_nonzero_u32(2)),
        main_output_channels: Some(new_nonzero_u32(2)),
        names: PortNames {
            layout: Some("stereo"),
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
        self.dsp.set_max_diff(Sample::splat(self.params.rate.value()), Sample::splat(buffer_config.sample_rate));
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
        let sample_rate = Sample::splat(context.transport().sample_rate);
        context.set_latency_samples(self.dsp.latency() as _);

        for s in buffer.iter_samples() {
            self.dsp.set_max_diff(Sample::splat(self.params.rate.smoothed.next()), sample_rate);
            let mut it = s.into_iter();
            let stereo = [it.next().unwrap(), it.next().unwrap()];
            let s = Sample::new(*stereo[0], *stereo[1]);
            let s = self.dsp.process([s]);
            *stereo[0] = s[0].extract(0);
            *stereo[1] = s[0].extract(1);
        }

        ProcessStatus::Normal
    }
}

impl Vst3Plugin for SlewPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"SolarLinerSlewEx";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Distortion,
    ];
}

impl ClapPlugin for SlewPlugin {
    const CLAP_ID: &'static str = "dev.solarliner.valib.Slew";

    const CLAP_DESCRIPTION: Option<&'static str> = Some("Slew rate limiter example");

    const CLAP_MANUAL_URL: Option<&'static str> = None;

    const CLAP_SUPPORT_URL: Option<&'static str> = Some("https://github.com/solarliner/valib");

    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Filter,
        ClapFeature::Distortion,
    ];
}

nih_export_clap!(SlewPlugin);
nih_export_vst3!(SlewPlugin);