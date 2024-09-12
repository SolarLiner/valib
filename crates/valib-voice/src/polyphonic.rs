use crate::{NoteData, Voice, VoiceManager};
use valib_core::dsp::DSPMeta;

pub struct Polyphonic<V: Voice> {
    create_voice: Box<dyn Fn(f32, NoteData<V::Sample>) -> V>,
    voice_pool: Box<[Option<V>]>,
    next_voice: usize,
    samplerate: f32,
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
        self.voice_pool.fill(None);
    }
}
