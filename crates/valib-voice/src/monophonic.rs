//! # Monophonic voice manager
//!
//! Provides a monophonic voice manager which can optionally do legato.

use crate::{NoteData, Voice, VoiceManager};
use num_traits::zero;
use valib_core::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib_core::dsp::{DSPMeta, DSPProcess, DSPProcessBlock};
use valib_core::util::lerp;
use valib_core::Scalar;

/// Monophonic voice manager over a single voice.
pub struct Monophonic<V: Voice> {
    /// Minimum pitch bend amount (semitones)
    pub pitch_bend_min_st: V::Sample,
    /// Maximum pitch bend amount (semitones)
    pub pitch_bend_max_st: V::Sample,
    create_voice: Box<dyn Fn(f32, NoteData<V::Sample>) -> V>,
    voice: Option<V>,
    base_frequency: V::Sample,
    pitch_bend_st: V::Sample,
    released: bool,
    legato: bool,
    samplerate: f32,
}

impl<V: Voice> DSPMeta for Monophonic<V> {
    type Sample = V::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = samplerate;
        if let Some(voice) = &mut self.voice {
            voice.set_samplerate(samplerate);
        }
    }

    fn latency(&self) -> usize {
        self.voice.as_ref().map(|v| v.latency()).unwrap_or(0)
    }

    fn reset(&mut self) {
        self.voice = None;
    }
}

impl<V: Voice> Monophonic<V> {
    /// Create a new monophonic voice manager.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Sample rate of the voices
    /// * `create_voice`: Closure to create the voice from the note data
    /// * `legato`: Whether to make the voice legato (don't reset the voice) or not
    ///
    /// returns: Monophonic<V>
    pub fn new(
        samplerate: f32,
        create_voice: impl Fn(f32, NoteData<V::Sample>) -> V + 'static,
        legato: bool,
    ) -> Self {
        Self {
            pitch_bend_min_st: V::Sample::from_f64(-2.),
            pitch_bend_max_st: V::Sample::from_f64(2.),
            create_voice: Box::new(create_voice),
            voice: None,
            released: false,
            base_frequency: V::Sample::from_f64(440.),
            pitch_bend_st: zero(),
            legato,
            samplerate,
        }
    }

    /// Whether the monophonic voice manager
    pub fn legato(&self) -> bool {
        self.legato
    }

    /// Set the monophonic voice manager is in legato mode or not
    pub fn set_legato(&mut self, legato: bool) {
        self.legato = legato;
    }
}

impl<V: Voice> VoiceManager<V> for Monophonic<V> {
    type ID = ();

    fn capacity(&self) -> usize {
        1
    }

    fn get_voice(&self, _id: Self::ID) -> Option<&V> {
        self.voice.as_ref()
    }

    fn get_voice_mut(&mut self, _id: Self::ID) -> Option<&mut V> {
        self.voice.as_mut()
    }

    fn all_voices(&self) -> impl Iterator<Item = Self::ID> {
        [()].into_iter()
    }

    fn active(&self) -> usize {
        if self.voice.as_ref().is_some_and(|v| v.active()) {
            1
        } else {
            0
        }
    }

    fn note_on(&mut self, note_data: NoteData<V::Sample>) -> Self::ID {
        self.base_frequency = note_data.frequency;
        self.pitch_bend_st = zero();
        if let Some(voice) = &mut self.voice {
            *voice.note_data_mut() = note_data;
            if self.released || !self.legato {
                voice.reuse();
            }
        } else {
            self.voice = Some((self.create_voice)(self.samplerate, note_data));
        }
    }

    fn note_off(&mut self, _id: Self::ID) {
        if let Some(voice) = &mut self.voice {
            voice.release();
        }
    }

    fn choke(&mut self, _id: Self::ID) {
        self.voice.take();
    }

    fn panic(&mut self) {
        self.voice.take();
    }

    fn pitch_bend(&mut self, amount: f64) {
        self.pitch_bend_st = lerp(
            V::Sample::from_f64(0.5 + amount / 2.),
            self.pitch_bend_min_st,
            self.pitch_bend_max_st,
        );
    }

    fn aftertouch(&mut self, amount: f64) {
        if let Some(voice) = &mut self.voice {
            voice.note_data_mut().pressure = V::Sample::from_f64(amount);
        }
    }

    fn pressure(&mut self, _: Self::ID, pressure: f32) {
        if let Some(voice) = &mut self.voice {
            voice.note_data_mut().pressure = V::Sample::from_f64(pressure as _);
        }
    }
    fn glide(&mut self, _: Self::ID, semitones: f32) {
        self.pitch_bend_st = V::Sample::from_f64(semitones as _);
    }
}

impl<V: Voice + DSPProcess<0, 1>> DSPProcess<0, 1> for Monophonic<V> {
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        if let Some(voice) = &mut self.voice {
            voice.process([])
        } else {
            [zero()]
        }
    }
}

impl<V: Voice + DSPProcessBlock<0, 1>> DSPProcessBlock<0, 1> for Monophonic<V> {
    fn process_block(
        &mut self,
        inputs: AudioBufferRef<Self::Sample, 0>,
        mut outputs: AudioBufferMut<Self::Sample, 1>,
    ) {
        if let Some(voice) = &mut self.voice {
            voice.process_block(inputs, outputs);
        } else {
            outputs.fill(zero())
        }
    }
    fn max_block_size(&self) -> Option<usize> {
        self.voice.as_ref().and_then(|v| v.max_block_size())
    }
}
