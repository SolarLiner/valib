use enum_map::{enum_map, Enum, EnumMap};
use nalgebra::SMatrix;
use nih_plug::util::db_to_gain_fast;
use valib::dsp::blocks::ModMatrix;
use valib::dsp::parameter::{HasParameters, Parameter, SmoothedParam};
use valib::dsp::{DSPMeta, DSPProcess};
use valib::filters::svf::Svf;
use valib::oversample::Oversampled;
use valib::saturators::{Clipper, Saturator, Slew};
use valib::simd::{AutoSimd, SimdValue};
use valib::Scalar;

pub(crate) type Sample = AutoSimd<[f32; 2]>;

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
        self.filter.set_samplerate(samplerate);
    }
}

impl DspInner {
    pub fn new(samplerate: f32) -> Self {
        let params = enum_map! {
            DspParam::Drive => Parameter::new(1.0).named("Drive").smoothed_linear(samplerate, 10.0),
            DspParam::Cutoff => Parameter::new(3000.0).named("Cutoff").smoothed_linear(samplerate, 1e-6),
            DspParam::Resonance => Parameter::new(0.5).named("Resonance").smoothed_linear(samplerate, 10.0),
            DspParam::LpGain => Parameter::new(1.0).named("LP Gain").smoothed_linear(samplerate, 10.0),
            DspParam::BpGain => Parameter::new(0.0).named("BP Gain").smoothed_linear(samplerate, 10.0),
            DspParam::HpGain => Parameter::new(0.0).named("HP Gain").smoothed_linear(samplerate, 10.0),
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

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        for s in self.params.values_mut() {
            s.set_samplerate(samplerate);
        }
        self.filter.set_samplerate(samplerate);
        self.mod_matrix.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.filter.latency() + self.mod_matrix.latency()
    }

    fn reset(&mut self) {
        self.filter.reset();
        self.mod_matrix.reset();
    }
}

impl DSPProcess<1, 1> for DspInner {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.filter
            .set_cutoff(Sample::splat(self.params[DspParam::Cutoff].next_sample()));
        self.filter.set_r(Sample::splat(
            1.0 - self.params[DspParam::Resonance].next_sample(),
        ));
        self.mod_matrix.weights.x = Sample::splat(self.params[DspParam::LpGain].next_sample());
        self.mod_matrix.weights.y = Sample::splat(self.params[DspParam::BpGain].next_sample());
        self.mod_matrix.weights.z = Sample::splat(self.params[DspParam::HpGain].next_sample());

        let drive = Sample::splat(db_to_gain_fast(self.params[DspParam::Drive].next_sample()));
        let [out] = self.mod_matrix.process(self.filter.process([x * drive]));
        [out / drive]
    }
}

#[derive(Debug, Clone, Copy)]
struct OpAmp<T>(Clipper, Slew<T>);

impl<T: Scalar> Default for OpAmp<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: Scalar> Saturator<T> for OpAmp<T>
where
    Clipper: Saturator<T>,
    Slew<T>: Saturator<T>,
{
    fn saturate(&self, x: T) -> T {
        self.1.saturate(self.0.saturate(x))
    }

    fn update_state(&mut self, x: T, y: T) {
        let xc = self.0.saturate(x);
        self.0.update_state(x, xc);
        self.1.update_state(xc, y);
    }

    fn sat_diff(&self, x: T) -> T {
        self.0.sat_diff(x) * self.1.sat_diff(self.0.saturate(x))
    }
}
