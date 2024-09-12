use valib_core::dsp::DSPMeta;
use valib_core::simd::SimdRealField;
use valib_core::Scalar;

pub mod monophonic;
pub mod polyphonic;

pub trait Voice: DSPMeta {
    fn active(&self) -> bool;
    fn note_data(&self) -> &NoteData<Self::Sample>;
    fn note_data_mut(&mut self) -> &mut NoteData<Self::Sample>;
    fn release(&mut self);
    fn reuse(&mut self);
    fn reset(&mut self);
}

pub struct Velocity<T> {
    value: T,
    sqrt: T,
}

impl<T: Copy> Velocity<T> {
    pub fn value(&self) -> T {
        self.value
    }

    pub fn sqrt(&self) -> T {
        self.sqrt
    }
}

impl<T: Copy + SimdRealField> Velocity<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            sqrt: value.simd_sqrt(),
        }
    }
}

pub struct Gain<T> {
    linear: T,
    db: T,
}

impl<T: Copy> Gain<T> {
    pub fn linear(&self) -> T {
        self.linear
    }

    pub fn db(&self) -> T {
        self.db
    }
}

impl<T: Scalar> Gain<T> {
    pub fn from_linear(value: T) -> Self {
        Self {
            linear: value,
            db: T::from_f64(20.) * value.simd_log10(),
        }
    }

    pub fn from_db(value: T) -> Self {
        Self {
            db: value,
            linear: T::from_f64(20.).simd_powf(value / T::from_f64(20.)),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct NoteData<T> {
    pub frequency: T,
    pub velocity: Velocity<T>,
    pub gain: Gain<T>,
    pub pan: T,
    pub pressure: T,
}

#[allow(unused_variables)]
pub trait VoiceManager<V: Voice>: DSPMeta<Sample = V::Sample> {
    type ID: Copy + Eq + Ord;

    fn capacity(&self) -> usize;

    fn get_voice(&self, id: Self::ID) -> Option<&V>;
    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut V>;

    fn is_voice_active(&self, id: Self::ID) -> bool {
        self.get_voice(id).is_some_and(|v| v.active())
    }
    fn all_voices(&self) -> impl Iterator<Item = Self::ID>;
    fn active(&self) -> usize {
        self.all_voices()
            .filter(|id| self.is_voice_active(*id))
            .count()
    }

    fn note_on(&mut self, note_data: NoteData<V::Sample>) -> Self::ID;
    fn note_off(&mut self, id: Self::ID);
    fn choke(&mut self, id: Self::ID);
    fn panic(&mut self);

    // Channel modulation
    fn pitch_bend(&mut self, amount: f64) {}
    fn aftertouch(&mut self, amount: f64) {}

    // MPE extensions
    fn pressure(&mut self, id: Self::ID, pressure: f32) {}
    fn glide(&mut self, id: Self::ID, semitones: f32) {}
    fn pan(&mut self, id: Self::ID, pan: f32) {}
    fn gain(&mut self, id: Self::ID, gain: f32) {}
}
