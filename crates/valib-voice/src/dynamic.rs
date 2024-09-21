use crate::monophonic::Monophonic;
use crate::polyphonic::Polyphonic;
use crate::{NoteData, Voice, VoiceManager};
use std::fmt;
use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;
use valib_core::dsp::{DSPMeta, DSPProcess};

#[derive(Debug)]
enum Impl<V: Voice> {
    Monophonic(Monophonic<V>),
    Polyphonic(Polyphonic<V>),
}

impl<V: Voice> DSPMeta for Impl<V> {
    type Sample = V::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        match self {
            Impl::Monophonic(mono) => mono.set_samplerate(samplerate),
            Impl::Polyphonic(poly) => poly.set_samplerate(samplerate),
        }
    }

    fn latency(&self) -> usize {
        match self {
            Impl::Monophonic(mono) => mono.latency(),
            Impl::Polyphonic(poly) => poly.latency(),
        }
    }

    fn reset(&mut self) {
        match self {
            Impl::Monophonic(mono) => mono.reset(),
            Impl::Polyphonic(poly) => poly.reset(),
        }
    }
}

pub struct DynamicVoice<V: Voice> {
    pitch_bend_st: Range<V::Sample>,
    poly_voice_capacity: usize,
    create_voice: Arc<dyn 'static + Send + Sync + Fn(f32, NoteData<V::Sample>) -> V>,
    current_manager: Impl<V>,
    legato: bool,
    samplerate: f32,
}

impl<V: Voice + fmt::Debug> fmt::Debug for DynamicVoice<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicVoice")
            .field("pitch_bend_st", &self.pitch_bend_st)
            .field("poly_voice_capacity", &self.poly_voice_capacity)
            .field("create_voice", &"Arc<dyn Fn(f32, NoteData<V::Sample>) -> V")
            .field("current_manager", &self.current_manager)
            .field("legato", &self.legato)
            .field("samplerate", &self.samplerate)
            .finish()
    }
}

impl<V: 'static + Voice> DynamicVoice<V> {
    pub fn new_mono(
        samplerate: f32,
        poly_voice_capacity: usize,
        legato: bool,
        create_voice: impl 'static + Send + Sync + Fn(f32, NoteData<V::Sample>) -> V,
    ) -> Self {
        let create_voice = Arc::new(create_voice);
        let mono = Monophonic::new(
            samplerate,
            {
                let create_voice = create_voice.clone();
                move |sr, nd| create_voice.clone()(sr, nd)
            },
            legato,
        );
        let pitch_bend_st = mono.pitch_bend_min_st..mono.pitch_bend_max_st;
        Self {
            pitch_bend_st,
            poly_voice_capacity,
            create_voice,
            current_manager: Impl::Monophonic(mono),
            legato,
            samplerate,
        }
    }

    pub fn new_poly(
        samplerate: f32,
        capacity: usize,
        legato: bool,
        create_voice: impl 'static + Send + Sync + Fn(f32, NoteData<V::Sample>) -> V,
    ) -> Self {
        let create_voice = Arc::new(create_voice);
        let poly = Polyphonic::new(samplerate, capacity, {
            let create_voice = create_voice.clone();
            move |sr, nd| create_voice.clone()(sr, nd)
        });
        let pitch_bend_st = poly.pitch_bend_st.clone();
        Self {
            pitch_bend_st,
            poly_voice_capacity: capacity,
            create_voice,
            current_manager: Impl::Polyphonic(poly),
            legato,
            samplerate,
        }
    }

    pub fn switch(&mut self, polyphonic: bool) {
        let new = match self.current_manager {
            Impl::Monophonic(..) if polyphonic => {
                let create_voice = self.create_voice.clone();
                let mut poly =
                    Polyphonic::new(self.samplerate, self.poly_voice_capacity, move |sr, nd| {
                        create_voice.clone()(sr, nd)
                    });
                poly.pitch_bend_st = self.pitch_bend_st.clone();
                Impl::Polyphonic(poly)
            }
            Impl::Polyphonic(..) if !polyphonic => {
                let create_voice = self.create_voice.clone();
                let mut mono = Monophonic::new(
                    self.samplerate,
                    move |sr, nd| create_voice.clone()(sr, nd),
                    self.legato,
                );
                mono.pitch_bend_min_st = self.pitch_bend_st.start;
                mono.pitch_bend_max_st = self.pitch_bend_st.end;
                Impl::Monophonic(mono)
            }
            _ => {
                return;
            }
        };
        self.current_manager = new;
    }

    pub fn is_monophonic(&self) -> bool {
        matches!(self.current_manager, Impl::Monophonic(..))
    }

    pub fn is_polyphonic(&self) -> bool {
        matches!(self.current_manager, Impl::Polyphonic(..))
    }

    pub fn legato(&self) -> bool {
        self.legato
    }

    pub fn set_legato(&mut self, legato: bool) {
        self.legato = legato;
        if let Impl::Monophonic(ref mut mono) = self.current_manager {
            mono.set_legato(legato);
        }
    }

    pub fn clean_inactive_voices(&mut self) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.clean_voice_if_inactive(),
            Impl::Polyphonic(ref mut poly) => poly.clean_inactive_voices(),
        }
    }
}

impl<V: Voice> DSPMeta for DynamicVoice<V> {
    type Sample = V::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = samplerate;
        self.current_manager.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.current_manager.latency()
    }

    fn reset(&mut self) {
        self.current_manager.reset();
    }
}

impl<V: Voice> VoiceManager for DynamicVoice<V> {
    type Voice = V;
    type ID = <Polyphonic<V> as VoiceManager>::ID;

    fn capacity(&self) -> usize {
        self.poly_voice_capacity
    }

    fn get_voice(&self, id: Self::ID) -> Option<&Self::Voice> {
        match self.current_manager {
            Impl::Monophonic(ref mono) => mono.get_voice(()),
            Impl::Polyphonic(ref poly) => poly.get_voice(id),
        }
    }

    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut Self::Voice> {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.get_voice_mut(()),
            Impl::Polyphonic(ref mut poly) => poly.get_voice_mut(id),
        }
    }

    fn all_voices(&self) -> impl Iterator<Item = Self::ID> {
        0..self.poly_voice_capacity
    }

    fn note_on(&mut self, note_data: NoteData<Self::Sample>) -> Self::ID {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => {
                mono.note_on(note_data);
                0
            }
            Impl::Polyphonic(ref mut poly) => poly.note_on(note_data),
        }
    }

    fn note_off(&mut self, id: Self::ID, release_velocity: f32) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => {
                mono.note_off((), release_velocity);
            }
            Impl::Polyphonic(ref mut poly) => {
                poly.note_off(id, release_velocity);
            }
        }
    }

    fn choke(&mut self, id: Self::ID) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.choke(()),
            Impl::Polyphonic(ref mut poly) => poly.choke(id),
        }
    }

    fn panic(&mut self) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.panic(),
            Impl::Polyphonic(ref mut poly) => poly.panic(),
        }
    }

    fn pitch_bend(&mut self, amount: f64) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.pitch_bend(amount),
            Impl::Polyphonic(ref mut poly) => poly.pitch_bend(amount),
        }
    }

    fn aftertouch(&mut self, amount: f64) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.aftertouch(amount),
            Impl::Polyphonic(ref mut poly) => poly.aftertouch(amount),
        }
    }

    fn pressure(&mut self, id: Self::ID, pressure: f32) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.glide((), pressure),
            Impl::Polyphonic(ref mut poly) => poly.glide(id, pressure),
        }
    }

    fn glide(&mut self, id: Self::ID, semitones: f32) {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.glide((), semitones),
            Impl::Polyphonic(ref mut poly) => poly.glide(id, semitones),
        }
    }
}

impl<V: Voice + DSPProcess<0, 1>> DSPProcess<0, 1> for DynamicVoice<V> {
    fn process(&mut self, []: [Self::Sample; 0]) -> [Self::Sample; 1] {
        match self.current_manager {
            Impl::Monophonic(ref mut mono) => mono.process([]),
            Impl::Polyphonic(ref mut poly) => poly.process([]),
        }
    }
}
