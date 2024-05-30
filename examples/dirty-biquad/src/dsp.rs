use enum_map::Enum;
use std::fmt;
use std::fmt::Formatter;

use crate::{MAX_BLOCK_SIZE, OVERSAMPLE};
use valib::dsp::parameter::{HasParameters, Parameter, SmoothedParam};
use valib::dsp::{DSPMeta, DSPProcess};
use valib::filters::biquad::Biquad;
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::clippers::DiodeClipperModel;
use valib::saturators::Dynamic;
use valib::simd::{AutoF32x2, SimdComplexField, SimdValue};

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
    pub fn as_biquad(&self, fc: Sample, res: Sample) -> Biquad<Sample, Dynamic<Sample>> {
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
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
    ftype: Parameter,
    saturator: Parameter,
    biquad: Biquad<Sample, Dynamic<Sample>>,
}

impl DspInner {
    fn update_params(&mut self) {
        let fc = Sample::splat(self.fc.next_sample() / self.samplerate);
        let res = Sample::splat(self.resonance.next_sample());
        let biquad = self.ftype.get_enum::<FilterType>().as_biquad(fc, res);
        self.biquad.update_coefficients(&biquad);
        if self.saturator.has_changed() {
            let nl = self.saturator.get_enum::<SaturatorType>().saturator();
            self.biquad.set_saturators(nl, nl);
        }
    }
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
        self.update_params();
        let drive = Sample::splat(self.drive.next_sample());
        println!("Drive {}", drive.extract(0));
        let x = x.map(|x| x * drive);
        self.biquad.process(x).map(|x| x / drive.simd_asinh())
    }
}

impl HasParameters for DspInner {
    type Name = DspParameters;

    fn get_parameter(&self, param: Self::Name) -> &Parameter {
        match param {
            DspParameters::Drive => &self.drive.param,
            DspParameters::Cutoff => &self.fc.param,
            DspParameters::Resonance => &self.resonance.param,
            DspParameters::FilterType => &self.ftype,
            DspParameters::SaturatorType => &self.saturator,
        }
    }
}

pub type Dsp = Oversampled<Sample, DspInner>;

pub fn create(samplerate: f32) -> Dsp {
    let dsp = DspInner {
        samplerate,
        drive: Parameter::new(1.0).smoothed_exponential(samplerate, 50.0),
        fc: Parameter::new(3000.0).smoothed_exponential(samplerate, 50.0),
        resonance: Parameter::new(0.5).smoothed_linear(samplerate, 50.0),
        ftype: Parameter::new(0.0),
        saturator: Parameter::new(0.0),
        biquad: Biquad::lowpass(Sample::splat(3000.0 / samplerate), Sample::splat(0.5)),
    };
    Oversample::new(OVERSAMPLE, MAX_BLOCK_SIZE).with_dsp(samplerate, dsp)
}
