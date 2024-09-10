use crate::dsp::DSPMeta;
use crate::{
    dsp::DSPProcess,
    util::{midi_to_freq, semitone_to_ratio},
};
use num_traits::{One, Zero};
use valib_core::Scalar;

#[allow(unused_variables)]
pub trait VoiceManager<const N: usize> {
    // Note trigger
    fn note_on(&mut self, midi_note: u8, velocity: f32);
    fn note_off(&mut self, midi_note: u8, velocity: f32);
    fn choke(&mut self, midi_note: u8);
    fn panic(&mut self);

    // Channel modulation
    fn pitch_bend(&mut self, amount: f32) {}
    fn aftertouch(&mut self, amount: f32) {}

    // MPE extensions
    fn pressure(&mut self, midi_note: u8, pressure: f32) {}
    fn glide(&mut self, midi_note: u8, semitones: f32) {}
    fn pan(&mut self, midi_note: u8, pan: f32) {}
    fn gain(&mut self, midi_note: u8, gain: f32) {}
}

/// Voice trait, implemted by full voice objects.
/// Inherits `DSP<5, 1>`, where inputs are:
/// - Frequency (Hz)
/// - Gate (> 0.5 indicates on)
/// - Pressure (unipolar)
/// - Velocity (unipolar)
/// - Pan (bipolar)
pub trait Voice: DSPProcess<5, 1> {
    /// Called when this voice is about to be used. Useful to reset some state, or prepare a new voice.
    fn reuse(&mut self, freq: f32, pressure: f32, velocity: f32, pan: f32);

    /// Returns true when this voice is done playing.
    fn done(&self) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct VoiceController<V: Voice> {
    pub id: u64,
    pub voice: V,
    pub used: bool,
    pub midi_note: u8,
    pub center_freq: f32,
    pub glide_semi: f32,
    pub pressure: f32,
    pub gate: bool,
    pub velocity: f32,
    pub gain: f32,
    pub pan: f32,
}

impl<V: Voice> VoiceController<V> {
    pub fn new_inactive(id: u64, voice: V) -> Self {
        Self {
            id,
            voice,
            used: false,
            midi_note: 0,
            center_freq: 0.0,
            glide_semi: 0.0,
            pressure: 0.0,
            gate: false,
            velocity: 0.0,
            gain: 1.0,
            pan: 0.0,
        }
    }
}

impl<V: Voice> DSPMeta for VoiceController<V> {
    type Sample = V::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.voice.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.voice.latency()
    }

    fn reset(&mut self) {
        self.voice.reset();
    }
}

impl<V: Voice> DSPProcess<2, 1> for VoiceController<V>
where
    V::Sample: Scalar,
{
    fn process(&mut self, [bend_st, aftertouch]: [Self::Sample; 2]) -> [Self::Sample; 1] {
        let freq = self.center_freq * semitone_to_ratio(self.glide_semi);
        let freq = Self::Sample::from_f64(freq as _) * semitone_to_ratio(bend_st);
        let [osc] = self.voice.process([
            freq,
            if self.gate {
                Self::Sample::one()
            } else {
                Self::Sample::zero()
            },
            Self::Sample::from_f64(self.pressure as _) + aftertouch,
            Self::Sample::from_f64((self.velocity * self.gain) as _),
            Self::Sample::from_f64(self.pan as _),
        ]);
        if self.voice.done() {
            self.used = false;
        }
        [osc * Self::Sample::from_f64(self.gain as f64)]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Monophonic<V: Voice> {
    voice: VoiceController<V>,
    aftertouch: f32,
}

impl<V: Voice> DSPMeta for Monophonic<V> {
    type Sample = V::Sample;

    fn reset(&mut self) {
        self.voice.reset();
    }
}

impl<V: Voice> DSPProcess<0, 1> for Monophonic<V> {
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        if self.voice.used {
            self.voice.process([
                Self::Sample::zero(),
                Self::Sample::from_f64(self.aftertouch as _),
            ])
        } else if self.voice.voice.done() {
            self.voice.used = false;
            [Self::Sample::zero()]
        } else {
            [Self::Sample::zero()]
        }
    }
}

impl<V: Voice> VoiceManager<1> for Monophonic<V> {
    fn note_on(&mut self, midi_note: u8, velocity: f32) {
        let freq = midi_to_freq(midi_note);
        if self.voice.used {
            self.voice.gate = true;
            self.voice.midi_note = midi_note;
            self.voice.center_freq = freq;
            self.voice.velocity = velocity;
        } else {
            self.voice.voice.reuse(freq, 0.0, velocity, 0.0);
            self.voice.midi_note = midi_note;
            self.voice.center_freq = freq;
            self.voice.glide_semi = 0.0;
            self.voice.pressure = 0.0;
            self.voice.gate = true;
            self.voice.velocity = velocity;
            self.voice.gain = 1.0;
            self.voice.pan = 0.0;
        }
    }

    fn note_off(&mut self, _midi_note: u8, _velocity: f32) {
        self.voice.gate = false;
    }

    fn choke(&mut self, _midi_note: u8) {
        self.voice.used = false;
    }

    fn panic(&mut self) {
        self.voice.used = false;
    }

    fn aftertouch(&mut self, amount: f32) {
        self.aftertouch = amount;
    }
}
