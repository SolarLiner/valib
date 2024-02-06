use crate::OpAmp;
use enum_map::{enum_map, Enum, EnumMap};
use nalgebra::SMatrix;
use valib::dsp::blocks::{ModMatrix, Series2};
use valib::dsp::parameter::{FilteredParam, HasParameters, Parameter, SmoothedParam};
use valib::dsp::{DSPBlock, DSP};
use valib::filters::svf::Svf;
use valib::oversample::Oversampled;
use valib::simd::{AutoSimd, SimdValue};

type Sample = AutoSimd<[f32; 2]>;

pub type Dsp = Oversampled<Sample, DspInner>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Enum)]
pub enum DspParam {
    Drive,
    Cutoff,
    Resonance,
    LpGain,
    BpGain,
    HpGain,
}

type Filter = Svf<Sample, OpAmp<Sample>>;

pub struct DspInner {
    params: EnumMap<DspParam, SmoothedParam>,
    filter: Filter,
    mod_matrix: ModMatrix<Sample, 3, 1>,
}

impl DspInner {
    pub(crate) fn set_samplerate(&mut self, samplerate: f32) {
        for (_, param) in self.params.iter_mut() {
            param.set_samplerate(samplerate);
        }
        self.filter.set_samplerate(Sample::splat(samplerate));
    }
}

impl DspInner {
    pub fn new(samplerate: f32) -> Self {
        let params = enum_map! {
            DspParam::Drive => Parameter::new(1.0).named("Drive").smoothed_linear(10.0),
            DspParam::Cutoff => Parameter::new(3000.0).named("Cutoff").smoothed_linear(10.0),
            DspParam::Resonance => Parameter::new(0.5).named("Resonance").smoothed_linear(10.0),
            DspParam::LpGain => Parameter::new(1.0).named("LP Gain").smoothed_linear(10.0),
            DspParam::BpGain => Parameter::new(0.0).named("BP Gain").smoothed_linear(10.0),
            DspParam::HpGain => Parameter::new(0.0).named("HP Gain").smoothed_linear(10.0),
        };
        let filter = Filter::new(
            Sample::splat(samplerate),
            Sample::splat(3000.0),
            Sample::splat(0.5),
        );
        let mod_matrix = ModMatrix {
            weights: SMatrix::<_, 1, 3>::new(
                Sample::splat(1.0),
                Sample::splat(0.0),
                Sample::splat(0.0),
            ),
            ..ModMatrix::default()
        };
        Self {
            params,
            filter,
            mod_matrix,
        }
    }
}

impl HasParameters for DspInner {
    type Enum = DspParam;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        &self.params[param].param
    }
}

impl DSP<1, 1> for DspInner {
    type Sample = Sample;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.filter
            .set_cutoff(Sample::splat(self.params[DspParam::Cutoff].next_sample()));
        self.filter.set_r(Sample::splat(
            self.params[DspParam::Resonance].next_sample(),
        ));
        self.mod_matrix.weights.x = Sample::splat(self.params[DspParam::LpGain].next_sample());
        self.mod_matrix.weights.y = Sample::splat(self.params[DspParam::BpGain].next_sample());
        self.mod_matrix.weights.z = Sample::splat(self.params[DspParam::HpGain].next_sample());

        self.mod_matrix.process(self.filter.process(x))
    }
}
