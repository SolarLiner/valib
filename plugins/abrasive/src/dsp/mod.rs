use crate::{
    dsp::filter::{FilterModule, FilterParams},
    MAX_BLOCK_SIZE,
};
use nalgebra::Complex;
use nih_plug::prelude::*;
use std::sync::Arc;
use valib::Scalar;
use valib::{
    dsp::{analysis::DspAnalysis, blocks::Series, BlockAdapter},
    oversample::{Oversample, Oversampled},
};
use valib::{
    dsp::{DSPMeta, DSPProcess},
    simd::SimdComplexField,
};

pub mod filter;

pub const NUM_BANDS: usize = 5;

#[derive(Debug, Params)]
pub struct DspParams {
    #[id = "drive"]
    pub drive: FloatParam,
    #[id = "scale"]
    pub scale: FloatParam,
    #[nested(array)]
    pub filters: [Arc<FilterParams>; NUM_BANDS],
}

impl Default for DspParams {
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
            filters: std::array::from_fn(|_| Default::default()),
        }
    }
}

pub struct Equalizer<T> {
    params: Arc<DspParams>,
    dsp: Series<[FilterModule<T>; NUM_BANDS]>,
}

impl<T: Scalar> DSPMeta for Equalizer<T> {
    type Sample = T;
    fn set_samplerate(&mut self, samplerate: f32) {
        self.dsp.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.dsp.latency()
    }

    fn reset(&mut self) {
        self.dsp.reset()
    }
}

impl<T: Scalar> DSPProcess<1, 1> for Equalizer<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let drive = T::from_f64(self.params.drive.smoothed.next() as _);
        let scale = T::from_f64(self.params.scale.smoothed.next() as _);
        for filter in &mut self.dsp.0 {
            filter.set_scale(scale);
        }
        let [y] = self.dsp.process([drive * x]);
        [y / drive]
    }
}

impl<T: Scalar> DspAnalysis<1, 1> for Equalizer<T> {
    fn h_z(&self, z: Complex<Self::Sample>) -> [[Complex<Self::Sample>; 1]; 1] {
        self.dsp.h_z(z)
    }

    fn freq_response(
        &self,
        samplerate: Self::Sample,
        f: Self::Sample,
    ) -> [[Complex<Self::Sample>; 1]; 1]
    where
        Complex<Self::Sample>: SimdComplexField,
    {
        self.dsp.freq_response(samplerate, f)
    }
}

impl<T: Scalar> Equalizer<T> {
    pub fn new(samplerate: T, params: Arc<DspParams>) -> Self {
        let dsp = Series(std::array::from_fn(|i| {
            FilterModule::new(samplerate, params.filters[i].clone())
        }));
        Self { params, dsp }
    }

    pub fn use_param_values(&mut self, use_modulated: bool) {
        for filter in &mut self.dsp.0 {
            filter.use_param_values(use_modulated);
        }
    }
}

pub const OVERSAMPLE: usize = 4;

pub type Dsp<T> = Oversampled<T, BlockAdapter<Equalizer<T>>>;
pub fn create<T: Scalar>(samplerate: f32, params: Arc<DspParams>) -> Dsp<T>
where
    Complex<T>: SimdComplexField,
{
    let target_samplerate = OVERSAMPLE as f32 * samplerate;
    Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE).with_dsp(
        samplerate,
        BlockAdapter(Equalizer::new(T::from_f64(target_samplerate as _), params)),
    )
}
