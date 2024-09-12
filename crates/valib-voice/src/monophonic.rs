use crate::{NoteData, Voice, VoiceManager};
use valib_core::dsp::DSPMeta;

pub struct Monophonic<V: Voice> {
    create_voice: Box<dyn Fn(f32, NoteData<V::Sample>) -> V>,
    voice: Option<V>,
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
    pub fn new(
        samplerate: f32,
        create_voice: impl Fn(f32, NoteData<V::Sample>) -> V + 'static,
        legato: bool,
    ) -> Self {
        Self {
            create_voice: Box::new(create_voice),
            voice: None,
            released: false,
            legato,
            samplerate,
        }
    }

    pub fn legato(&self) -> bool {
        self.legato
    }

    pub fn set_legato(&mut self, legato: bool) {
        self.legato = legato;
    }
}

#[derive(Debug, Default, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum MonophonicVoiceId {
    #[default]
    Mono,
}

impl<V: Voice> VoiceManager<V> for Monophonic<V> {
    type ID = MonophonicVoiceId;
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
        [MonophonicVoiceId::Mono].into_iter()
    }

    fn active(&self) -> usize {
        self.voice
            .as_ref()
            .is_some_and(|v| v.active())
            .then_some(1)
            .unwrap_or(0)
    }

    fn note_on(&mut self, note_data: NoteData<V::Sample>) -> Self::ID {
        if let Some(voice) = &mut self.voice {
            *voice.note_data_mut() = note_data;
            if self.released || !self.legato {
                voice.reuse();
            }
        } else {
            self.voice = Some((self.create_voice)(self.samplerate, note_data));
        }
        MonophonicVoiceId::Mono
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
}
