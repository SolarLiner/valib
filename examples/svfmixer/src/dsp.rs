use nalgebra::SMatrix;
use nih_plug::util::db_to_gain_fast;

use valib::dsp::blocks::ModMatrix;
use valib::dsp::parameter::{HasParameters, ParamId, ParamMap, ParamName, SmoothedParam};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess};
use valib::filters::svf::Svf;
use valib::oversample::Oversampled;
use valib::saturators::{Clipper, Saturator, Slew};
use valib::simd::{AutoSimd, SimdComplexField, SimdValue};
use valib_core::Scalar;

pub(crate) type Sample = AutoSimd<[f32; 2]>;

#[derive(Debug, Copy, Clone, Default)]
struct Sinh;

impl Saturator<Sample> for Sinh {
    fn saturate(&self, x: Sample) -> Sample {
        x.simd_sinh()
    }
}

pub type Dsp = Oversampled<Sample, BlockAdapter<DspInner>>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ParamName)]
pub enum DspParam {
    Drive,
    Cutoff,
    Resonance,
    LpGain,
    BpGain,
    HpGain,
}

type Filter = Svf<Sample, Sinh>;

pub struct DspInner {
    params: ParamMap<DspParam, SmoothedParam>,
    filter: Filter,
    mod_matrix: ModMatrix<Sample, 3, 1>,
}

impl DspInner {
    pub fn new(samplerate: f32) -> Self {
        let params = ParamMap::new(|p| match p {
            DspParam::Drive => SmoothedParam::linear(1.0, samplerate, 10.0),
            DspParam::Cutoff => SmoothedParam::linear(3000.0, samplerate, 1e-6),
            DspParam::Resonance => SmoothedParam::linear(0.5, samplerate, 10.0),
            DspParam::LpGain => SmoothedParam::linear(1.0, samplerate, 10.0),
            DspParam::BpGain => SmoothedParam::linear(0.0, samplerate, 10.0),
            DspParam::HpGain => SmoothedParam::linear(0.0, samplerate, 10.0),
        });
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
    type Name = DspParam;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        self.params[param].param = value;
    }
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        for (_, s) in self.params.iter_mut() {
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
struct OpAmp<T>(Clipper<T>, Slew<T>);

impl<T: Scalar> Default for OpAmp<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: Scalar> Saturator<T> for OpAmp<T>
where
    Clipper<T>: Saturator<T>,
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
