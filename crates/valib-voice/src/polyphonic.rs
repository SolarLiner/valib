//! # Polyphonic voice manager
//!
//! Provides a polyphonic voice manager with rotating voice allocation.

use crate::{NoteData, Voice, VoiceManager};
use num_traits::zero;
use numeric_literals::replace_float_literals;
use std::fmt;
use std::fmt::Formatter;
use std::ops::Range;
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::util::lerp;
use valib_core::Scalar;

/// Polyphonic voice manager with rotating voice allocation
pub struct Polyphonic<V: Voice> {
    pub pitch_bend_st: Range<V::Sample>,
    create_voice: Box<dyn 'static + Send + Sync + Fn(f32, NoteData<V::Sample>) -> V>,
    voice_pool: Box<[Option<V>]>,
    active_voices: usize,
    next_voice: usize,
    samplerate: f32,
    pitch_bend: V::Sample,
}

impl<V: Voice> fmt::Debug for Polyphonic<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Polyphonic")
            .field(
                "create_voice",
                &"Box<dyn Fn(f32, NoteData<V::Sample>) -> V>",
            )
            .field("voice_pool", &"Box<[Option<impl Voice>]>")
            .field("next_voice", &self.next_voice)
            .field("samplerate", &self.samplerate)
            .finish()
    }
}

impl<V: Voice> Polyphonic<V> {
    /// Create a new polyphonice voice manager.
    ///
    /// # Arguments
    ///
    /// * `samplerate`: Sample rate the voices will run at
    /// * `voice_capacity`: Maximum voice capacity
    /// * `create_voice`: Closure to create a voice given the given note data
    ///
    /// returns: Polyphonic<V>
    pub fn new(
        samplerate: f32,
        voice_capacity: usize,
        create_voice: impl 'static + Send + Sync + Fn(f32, NoteData<V::Sample>) -> V + 'static,
    ) -> Self {
        Self {
            pitch_bend_st: V::Sample::from_f64(-2.)..V::Sample::from_f64(2.),
            create_voice: Box::new(create_voice),
            next_voice: 0,
            voice_pool: (0..voice_capacity).map(|_| None).collect(),
            active_voices: 0,
            samplerate,
            pitch_bend: zero(),
        }
    }

    /// Clean inactive voices to prevent them being processed for nothing.
    pub fn clean_inactive_voices(&mut self) {
        for slot in &mut self.voice_pool {
            if slot.as_ref().is_some_and(|v| !v.active()) {
                slot.take();
                self.active_voices -= 1;
            }
        }
    }

    fn update_voices_pitchmod(&mut self) {
        let mod_st = self.get_pitch_bend();
        for voice in self.voice_pool.iter_mut().filter_map(|opt| opt.as_mut()) {
            voice.note_data_mut().modulation_st = mod_st;
        }
    }

    #[replace_float_literals(V::Sample::from_f64(literal))]
    fn get_pitch_bend(&self) -> V::Sample {
        let t = 0.5 * self.pitch_bend + 0.5;
        lerp(t, self.pitch_bend_st.start, self.pitch_bend_st.end)
    }
}

impl<V: Voice> DSPMeta for Polyphonic<V> {
    type Sample = V::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = samplerate;
        for voice in self.voice_pool.iter_mut().flatten() {
            voice.set_samplerate(samplerate);
        }
    }

    fn latency(&self) -> usize {
        self.voice_pool
            .iter()
            .flatten()
            .map(|v| v.latency())
            .max()
            .unwrap_or(0)
    }

    fn reset(&mut self) {
        self.voice_pool.iter_mut().flatten().for_each(|v| v.reset());
    }
}

impl<V: Voice> VoiceManager for Polyphonic<V> {
    type Voice = V;
    type ID = usize;

    fn capacity(&self) -> usize {
        self.voice_pool.len()
    }

    fn get_voice(&self, id: Self::ID) -> Option<&V> {
        self.voice_pool[id].as_ref()
    }

    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut V> {
        self.voice_pool[id].as_mut()
    }

    fn all_voices(&self) -> impl Iterator<Item = Self::ID> {
        0..self.capacity()
    }

    fn note_on(&mut self, note_data: NoteData<V::Sample>) -> Self::ID {
        if self.active_voices == self.capacity() {
            // At capacity, we must steal a voice
            let id = self.next_voice;

            if let Some(voice) = &mut self.voice_pool[id] {
                *voice.note_data_mut() = note_data;
                voice.reuse();
            } else {
                self.voice_pool[id] = Some((self.create_voice)(self.samplerate, note_data));
            }

            self.next_voice = (self.next_voice + 1) % self.voice_pool.len();
            id
        } else {
            // Find first available slot
            while self.voice_pool[self.next_voice].is_some() {
                self.next_voice = (self.next_voice + 1) % self.voice_pool.len();
            }

            let id = self.next_voice;
            self.voice_pool[id] = Some((self.create_voice)(self.samplerate, note_data));
            self.next_voice = (self.next_voice + 1) % self.voice_pool.len();
            self.active_voices += 1;
            id
        }
    }

    fn note_off(&mut self, id: Self::ID, release_velocity: f32) {
        if let Some(voice) = &mut self.voice_pool[id] {
            voice.release(release_velocity);
        }
    }

    fn choke(&mut self, id: Self::ID) {
        self.voice_pool[id] = None;
        self.active_voices -= 1;
    }

    fn panic(&mut self) {
        self.voice_pool.fill_with(|| None);
        self.active_voices = 0;
    }

    fn pitch_bend(&mut self, amount: f64) {
        self.pitch_bend = V::Sample::from_f64(amount);
        self.update_voices_pitchmod();
    }

    fn aftertouch(&mut self, amount: f64) {
        let pressure = V::Sample::from_f64(amount);
        for voice in self.voice_pool.iter_mut().filter_map(|x| x.as_mut()) {
            voice.note_data_mut().pressure = pressure;
        }
    }

    fn pressure(&mut self, id: Self::ID, pressure: f32) {
        if let Some(voice) = &mut self.voice_pool[id] {
            voice.note_data_mut().pressure = V::Sample::from_f64(pressure as _);
        }
    }

    fn glide(&mut self, id: Self::ID, semitones: f32) {
        let mod_st = V::Sample::from_f64(semitones as _);
        if let Some(voice) = &mut self.voice_pool[id] {
            voice.note_data_mut().modulation_st = mod_st;
        }
    }
}

impl<V: Voice + DSPProcess<0, 1>> DSPProcess<0, 1> for Polyphonic<V> {
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let mut out = zero();
        for voice in self.voice_pool.iter_mut().flatten() {
            let [y] = voice.process([]);
            out += y;
        }
        [out]
    }
}
