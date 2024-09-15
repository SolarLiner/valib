//! # Polyphonic voice manager
//!
//! Provides a polyphonic voice manager with rotating voice allocation.
use crate::{NoteData, Voice, VoiceManager};
use num_traits::zero;
use valib_core::dsp::{DSPMeta, DSPProcess};

/// Polyphonic voice manager with rotating voice allocation
pub struct Polyphonic<V: Voice> {
    create_voice: Box<dyn Fn(f32, NoteData<V::Sample>) -> V>,
    voice_pool: Box<[Option<V>]>,
    next_voice: usize,
    samplerate: f32,
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
        create_voice: impl Fn(f32, NoteData<V::Sample>) -> V + 'static,
    ) -> Self {
        Self {
            create_voice: Box::new(create_voice),
            next_voice: 0,
            voice_pool: (0..voice_capacity).map(|_| None).collect(),
            samplerate,
        }
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

impl<V: Voice> VoiceManager<V> for Polyphonic<V> {
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
        let id = self.next_voice;
        self.next_voice += 1;

        if let Some(voice) = &mut self.voice_pool[id] {
            *voice.note_data_mut() = note_data;
            voice.reuse();
        } else {
            self.voice_pool[id] = Some((self.create_voice)(self.samplerate, note_data));
        }

        id
    }

    fn note_off(&mut self, id: Self::ID) {
        if let Some(voice) = &mut self.voice_pool[id] {
            voice.release();
        }
    }

    fn choke(&mut self, id: Self::ID) {
        self.voice_pool[id] = None;
    }

    fn panic(&mut self) {
        self.voice_pool.fill_with(|| None);
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
