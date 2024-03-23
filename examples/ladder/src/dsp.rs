use std::fmt;
use std::fmt::Formatter;

use enum_map::Enum;

use valib::dsp::parameter::{HasParameters, Parameter, SmoothedParam};
use valib::dsp::{DSPMeta, DSPProcess};
use valib::filters::ladder::{Ideal, Ladder, Transistor, OTA};
use valib::oversample::{Oversample, Oversampled};
use valib::saturators::clippers::DiodeClipperModel;
use valib::saturators::Tanh;
use valib::simd::{AutoF32x2, SimdValue};

use crate::{MAX_BUFFER_SIZE, OVERSAMPLE};

pub type Sample = AutoF32x2;

#[allow(clippy::large_enum_variant)]
enum DspLadder {
    Ideal(Ladder<Sample, Ideal>),
    Transistor(Ladder<Sample, Transistor<DiodeClipperModel<Sample>>>),
    Ota(Ladder<Sample, OTA<Tanh>>),
}

impl DSPMeta for DspLadder {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        match self {
            Self::Ideal(ladder) => ladder.set_samplerate(samplerate),
            Self::Transistor(ladder) => ladder.set_samplerate(samplerate),
            Self::Ota(ladder) => ladder.set_samplerate(samplerate),
        }
    }

    fn latency(&self) -> usize {
        4 // Inlined from the ladder implementation
    }

    fn reset(&mut self) {
        match self {
            Self::Ideal(ladder) => ladder.reset(),
            Self::Transistor(ladder) => ladder.reset(),
            Self::Ota(ladder) => ladder.reset(),
        }
    }
}

impl DSPProcess<1, 1> for DspLadder {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        match self {
            Self::Ideal(ladder) => ladder.process(x),
            Self::Transistor(ladder) => ladder.process(x),
            Self::Ota(ladder) => ladder.process(x),
        }
    }
}

impl DspLadder {
    fn set_cutoff(&mut self, fc: Sample) {
        match self {
            Self::Ideal(ladder) => ladder.set_cutoff(fc),
            Self::Transistor(ladder) => ladder.set_cutoff(fc),
            Self::Ota(ladder) => ladder.set_cutoff(fc),
        }
    }

    fn set_resonance(&mut self, res: Sample) {
        match self {
            Self::Ideal(ladder) => ladder.set_resonance(res),
            Self::Transistor(ladder) => ladder.set_resonance(res),
            Self::Ota(ladder) => ladder.set_resonance(res),
        }
    }

    fn set_compensated(&mut self, compensated: bool) {
        match self {
            Self::Ideal(ladder) => ladder.compensated = compensated,
            Self::Transistor(ladder) => ladder.compensated = compensated,
            Self::Ota(ladder) => ladder.compensated = compensated,
        }
    }
}

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Enum)]
pub enum LadderType {
    #[default]
    Ideal,
    Transistor,
    Ota,
}

impl fmt::Display for LadderType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ideal => write!(f, "Ideal"),
            Self::Transistor => write!(f, "Transistor"),
            Self::Ota => write!(f, "OTA"),
        }
    }
}

impl LadderType {
    fn as_ladder(&self, samplerate: f32, fc: Sample, res: Sample) -> DspLadder {
        match self {
            Self::Ideal => DspLadder::Ideal(Ladder::new(samplerate, fc, res)),
            Self::Transistor => DspLadder::Transistor(Ladder::new(samplerate, fc, res)),
            Self::Ota => DspLadder::Ota(Ladder::new(samplerate, fc, res)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Enum)]
pub enum DspParameters {
    LadderType,
    Drive,
    Cutoff,
    Resonance,
    Compensated,
}

pub struct DspInner {
    ladder_type: Parameter,
    drive: SmoothedParam,
    cutoff: SmoothedParam,
    resonance: SmoothedParam,
    compensated: Parameter,
    ladder: DspLadder,
    samplerate: f32,
}

impl DspInner {
    fn update_from_parameters(&mut self) {
        let fc = Sample::splat(self.cutoff.next_sample());
        let res = Sample::splat({
            let k = 4.0 * self.resonance.next_sample();
            if self.ladder_type.get_enum::<LadderType>() == LadderType::Ideal {
                k.min(3.95)
            } else {
                k
            }
        });
        if self.ladder_type.has_changed() {
            self.ladder =
                self.ladder_type
                    .get_enum::<LadderType>()
                    .as_ladder(self.samplerate, fc, res);
        }

        self.ladder.set_cutoff(fc);
        self.ladder.set_resonance(res);
        self.ladder.set_compensated(self.compensated.get_bool());
    }
}

impl DSPMeta for DspInner {
    type Sample = Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = samplerate;
        self.drive.set_samplerate(samplerate);
        self.cutoff.set_samplerate(samplerate);
        self.resonance.set_samplerate(samplerate);
        self.ladder.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.ladder.latency()
    }

    fn reset(&mut self) {
        self.drive.reset();
        self.cutoff.reset();
        self.resonance.reset();
        self.ladder.reset();
    }
}

impl DSPProcess<1, 1> for DspInner {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.update_from_parameters();
        let drive = Sample::splat(self.drive.next_sample() / 4.0);
        self.ladder.process(x.map(|x| x * drive)).map(|x| x / drive)
    }
}

impl HasParameters for DspInner {
    type Enum = DspParameters;

    fn get_parameter(&self, param: Self::Enum) -> &Parameter {
        match param {
            DspParameters::LadderType => &self.ladder_type,
            DspParameters::Drive => &self.drive.param,
            DspParameters::Cutoff => &self.cutoff.param,
            DspParameters::Resonance => &self.resonance.param,
            DspParameters::Compensated => &self.compensated,
        }
    }
}

pub type Dsp = Oversampled<Sample, DspInner>;

pub fn create(samplerate: f32) -> Dsp {
    let dsp = DspInner {
        ladder_type: Parameter::new(0.0),
        drive: Parameter::new(1.0).smoothed_exponential(samplerate, 50.0),
        cutoff: Parameter::new(300.0).smoothed_exponential(samplerate, 10.0),
        resonance: Parameter::new(0.5).smoothed_exponential(samplerate, 50.0),
        ladder: LadderType::Ideal.as_ladder(samplerate, Sample::splat(300.0), Sample::splat(0.5)),
        compensated: Parameter::new(0.0),
        samplerate,
    };
    Oversample::new(OVERSAMPLE, MAX_BUFFER_SIZE).with_dsp(samplerate, dsp)
}
