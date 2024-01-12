use crate::{
    dsp::DSP,
    util::{midi_to_freq, semitone_to_ratio},
    Scalar,
};
use num_traits::{One, Zero};

#[allow(unused_variables)]
pub trait VoiceManager<const N: usize>: DSP<0, N> {
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
/// Inherits `DSP<4, 1>`, where inputs are:
/// - Frequency (Hz)
/// - Gate (> 0.5 indicates on)
/// - Pressure (unipolar)
/// - Velocity (unipolar)
/// - Pan (bipolar)
pub trait Voice: DSP<5, 1> {
    fn create(freq: f32, pressure: f32, velocity: f32, pan: f32) -> Self;
    fn done(&self) -> bool;
}

#[derive(Debug, Clone, Copy)]
pub struct VoiceController<V: Voice> {
    voice: V,
    midi_note: u8,
    center_freq: f32,
    glide_semi: f32,
    pressure: f32,
    gate: bool,
    velocity: f32,
    gain: f32,
    pan: f32,
}

impl<V: Voice> DSP<2, 1> for VoiceController<V>
where
    V::Sample: Scalar,
{
    type Sample = V::Sample;

    fn latency(&self) -> usize {
        self.voice.latency()
    }

    fn reset(&mut self) {
        self.voice.reset();
    }

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
        [osc * Self::Sample::from_f64(self.gain as f64)]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Monophonic<V: Voice> {
    voice: Option<VoiceController<V>>,
    aftertouch: f32,
}

impl<V: Voice> DSP<0, 1> for Monophonic<V> {
    type Sample = V::Sample;

    fn reset(&mut self) {
        self.voice.take();
    }

    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        if self.voice.as_ref().is_some_and(|v| !v.voice.done()) {
            self.voice.as_mut().unwrap().process([
                Self::Sample::zero(),
                Self::Sample::from_f64(self.aftertouch as _),
            ])
        } else if self.voice.is_some() {
            self.voice.take();
            [Self::Sample::zero()]
        } else {
            [Self::Sample::zero()]
        }
    }
}

impl<V: Voice> VoiceManager<1> for Monophonic<V> {
    fn note_on(&mut self, midi_note: u8, velocity: f32) {
        let freq = midi_to_freq(midi_note);
        if let Some(voice_ctrl) = &mut self.voice {
            voice_ctrl.gate = true;
            voice_ctrl.midi_note = midi_note;
            voice_ctrl.center_freq = freq;
            voice_ctrl.velocity = velocity;
        } else {
            self.voice.replace(VoiceController {
                voice: V::create(freq, 0.0, velocity, 0.0),
                midi_note,
                center_freq: freq,
                glide_semi: 0.0,
                pressure: 0.0,
                gate: true,
                velocity,
                gain: 1.0,
                pan: 0.0,
            });
        }
    }

    fn note_off(&mut self, midi_note: u8, velocity: f32) {
        if let Some(ctrl) = &mut self.voice {
            if ctrl.midi_note == midi_note {
                ctrl.velocity = velocity;
                ctrl.gate = false;
            }
        }
    }

    fn choke(&mut self, midi_note: u8) {
        if self
            .voice
            .as_ref()
            .is_some_and(|v| v.midi_note == midi_note)
        {
            self.voice.take();
        }
    }

    fn panic(&mut self) {
        self.voice.take();
    }

    fn aftertouch(&mut self, amount: f32) {
        self.aftertouch = amount;
    }
}
