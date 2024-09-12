use std::fmt;
use std::fmt::Formatter;

use nih_plug::prelude::Enum;

use valib::dsp::parameter::{HasParameters, ParamId, ParamName, RemoteControlled, SmoothedParam};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess};
use valib::filters::biquad::Biquad;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::clippers::DiodeClipperModel;
use valib::saturators::{Dynamic, Linear};
use valib::simd::{AutoF32x2, SimdComplexField, SimdValue};

use crate::{MAX_BLOCK_SIZE, OVERSAMPLE};

pub type Sample = AutoF32x2;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
pub enum FilterType {
    Lowpass,
    Bandpass,
    Highpass,
    Notch,
    Allpass,
}

impl fmt::Display for FilterType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lowpass => write!(f, "Low pass"),
            Self::Bandpass => write!(f, "Band pass"),
            Self::Highpass => write!(f, "High pass"),
            Self::Notch => write!(f, "Notch"),
            Self::Allpass => write!(f, "Allpass"),
        }
    }
}

impl FilterType {
    pub fn as_biquad(&self, fc: Sample, res: Sample) -> Biquad<Sample, Linear> {
        match self {
            Self::Lowpass => Biquad::lowpass(fc, res),
            Self::Bandpass => Biquad::bandpass_peak0(fc, res),
            Self::Highpass => Biquad::highpass(fc, res),
            Self::Notch => Biquad::notch(fc, res),
            Self::Allpass => Biquad::allpass(fc, res),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
pub enum SaturatorType {
    Linear,
    Tanh,
    DiodeSym,
    DiodeAssym,
}

impl fmt::Display for SaturatorType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::Tanh => write!(f, "Tanh"),
            Self::DiodeSym => write!(f, "Diode (symmetric)"),
            Self::DiodeAssym => write!(f, "Diode (asymmetric)"),
        }
    }
}

impl SaturatorType {
    pub fn saturator(&self) -> Dynamic<Sample> {
        match self {
            Self::Linear => Dynamic::Linear,
            Self::Tanh => Dynamic::Tanh,
            Self::DiodeSym => Dynamic::DiodeClipper(DiodeClipperModel::new_silicon(1, 1)),
            Self::DiodeAssym => Dynamic::DiodeClipper(DiodeClipperModel::new_germanium(1, 2)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ParamName)]
pub enum DspParameters {
    Drive,
    Cutoff,
    Resonance,
    FilterType,
    SaturatorType,
}

pub struct DspInner {
    samplerate: f32,
    drive: SmoothedParam,
    fc: SmoothedParam,
    resonance: SmoothedParam,
    ftype: FilterType,
    saturator: SaturatorType,
    biquad: Biquad<Sample, Dynamic<Sample>>,
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = samplerate;
        self.drive.set_samplerate(samplerate);
        self.fc.set_samplerate(samplerate);
        self.resonance.set_samplerate(samplerate);
        self.biquad.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.biquad.latency()
    }

    fn reset(&mut self) {
        self.drive.reset();
        self.fc.reset();
        self.resonance.reset();
        self.biquad.reset();
    }
}

impl DSPProcess<1, 1> for DspInner {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.update_biquad();
        let drive = Sample::splat(self.drive.next_sample());
        let x = x.map(|x| x * drive);
        self.biquad.process(x).map(|x| x / drive.simd_asinh())
    }
}

impl HasParameters for DspInner {
    type Name = DspParameters;

    fn set_parameter(&mut self, param: Self::Name, value: f32) {
        match param {
            DspParameters::Drive => {
                self.drive.param = value;
            }
            DspParameters::Cutoff => {
                self.fc.param = value;
            }
            DspParameters::Resonance => {
                self.resonance.param = value;
            }
            DspParameters::FilterType => {
                self.ftype = FilterType::from_index(value as _);
            }
            DspParameters::SaturatorType => {
                let nl = SaturatorType::from_index(value as usize).saturator();
                self.biquad.set_saturators(nl, nl);
            }
        }
    }
}

impl DspInner {
    fn update_biquad(&mut self) {
        let fc = Sample::splat(self.fc.next_sample() / self.samplerate);
        let res = Sample::splat(self.resonance.next_sample());
        let biquad = self.ftype.as_biquad(fc, res);
        self.biquad.update_coefficients(&biquad);
    }
}

pub type Dsp = Oversampled<Sample, BlockAdapter<DspInner>>;

pub fn create(samplerate: f32) -> RemoteControlled<Dsp> {
    let dsp = DspInner {
        samplerate,
        drive: SmoothedParam::exponential(1.0, samplerate, 10.0),
        fc: SmoothedParam::exponential(3000.0, samplerate, 50.0),
        resonance: SmoothedParam::linear(0.5, samplerate, 10.0),
        ftype: FilterType::Lowpass,
        saturator: SaturatorType::Linear,
        biquad: Biquad::lowpass(Sample::splat(3000.0 / samplerate), Sample::splat(0.5))
            .with_saturators(Dynamic::default(), Dynamic::default()),
    };
    let dsp = Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE).with_dsp(samplerate, BlockAdapter(dsp));
    RemoteControlled::new(samplerate, 1e3, dsp)
}
