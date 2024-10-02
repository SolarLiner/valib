#![warn(missing_docs)]
//! # Voice abstractions
//!
//! This crate provides abstractions around voice processing and voice management.
use valib_core::dsp::{BlockAdapter, DSPMeta, DSPProcessBlock, SampleAdapter};
use valib_core::simd::SimdRealField;
use valib_core::util::{midi_to_freq, semitone_to_ratio};
use valib_core::Scalar;

pub mod dynamic;
pub mod monophonic;
pub mod polyphonic;
#[cfg(feature = "resampled")]
pub mod upsample;

/// Trait for types which process a synth voice.
pub trait Voice: DSPMeta {
    /// Returns true if this voice is currently active.
    fn active(&self) -> bool;
    /// Return a reference to the voice's note data
    fn note_data(&self) -> &NoteData<Self::Sample>;
    /// Return a mutable reference to the voice's note data
    fn note_data_mut(&mut self) -> &mut NoteData<Self::Sample>;
    /// Release the note (corresponding to a note off)
    fn release(&mut self, release_velocity: f32);
    /// Reuse the note (corresponding to a soft reset)
    fn reuse(&mut self);
}

impl<V: DSPProcessBlock<I, O> + Voice, const I: usize, const O: usize> Voice
    for SampleAdapter<V, I, O>
{
    fn active(&self) -> bool {
        self.inner.active()
    }

    fn note_data(&self) -> &NoteData<Self::Sample> {
        self.inner.note_data()
    }

    fn note_data_mut(&mut self) -> &mut NoteData<Self::Sample> {
        self.inner.note_data_mut()
    }

    fn release(&mut self, release_velocity: f32) {
        self.inner.release(release_velocity);
    }

    fn reuse(&mut self) {
        self.inner.reuse();
    }
}

impl<V: Voice> Voice for BlockAdapter<V> {
    fn active(&self) -> bool {
        self.0.active()
    }

    fn note_data(&self) -> &NoteData<Self::Sample> {
        self.0.note_data()
    }

    fn note_data_mut(&mut self) -> &mut NoteData<Self::Sample> {
        self.0.note_data_mut()
    }

    fn release(&mut self, release_velocity: f32) {
        self.0.release(release_velocity);
    }

    fn reuse(&mut self) {
        self.0.reuse();
    }
}

/// Value representing velocity. The square root is precomputed to be used in voices directly.
#[derive(Debug, Copy, Clone)]
pub struct Velocity<T> {
    value: T,
    sqrt: T,
}

impl<T: Copy> Velocity<T> {
    /// Linear value of the velocity.
    pub fn value(&self) -> T {
        self.value
    }

    /// Square root of the velocity. Useful for voices so that volume feels more natural.
    pub fn sqrt(&self) -> T {
        self.sqrt
    }
}

impl<T: Copy + SimdRealField> Velocity<T> {
    /// Create a new `Velocity` value with the linear velocity value.
    ///
    /// # Arguments
    ///
    /// * `value`: Linear velocity value
    ///
    /// returns: Velocity<T>
    pub fn new(value: T) -> Self {
        Self {
            value,
            sqrt: value.simd_sqrt(),
        }
    }
}

/// Gain type, with precomputed linear and decibel values
#[derive(Debug, Copy, Clone)]
pub struct Gain<T> {
    linear: T,
    db: T,
}

impl<T: Copy> Gain<T> {
    /// Return the linear gain value
    pub fn linear(&self) -> T {
        self.linear
    }

    /// Return the decibel gain value
    pub fn db(&self) -> T {
        self.db
    }
}

impl<T: Scalar> Gain<T> {
    /// Create a `Gain` type from a linear gain value
    ///
    /// # Arguments
    ///
    /// * `value`: Linear gain value
    ///
    /// returns: Gain<T>
    pub fn from_linear(value: T) -> Self {
        Self {
            linear: value,
            db: T::from_f64(20.) * value.simd_log10(),
        }
    }

    /// Create a `Gain` type from a decibel gain value
    ///
    /// # Arguments
    ///
    /// * `value`: Decibel gain value
    ///
    /// returns: Gain<T>
    pub fn from_db(value: T) -> Self {
        Self {
            db: value,
            linear: T::from_f64(20.).simd_powf(value / T::from_f64(20.)),
        }
    }
}

/// Note data type containing major data about voice expression
#[derive(Debug, Copy, Clone)]
pub struct NoteData<T> {
    /// Note frequency
    pub frequency: T,
    /// Frequency modulation (pitch bend, glide)
    pub modulation_st: T,
    /// Note velocity
    pub velocity: Velocity<T>,
    /// Note gain
    pub gain: Gain<T>,
    /// Note pan
    pub pan: T,
    /// Note pressure
    pub pressure: T,
}

impl<T: Scalar> NoteData<T> {
    pub fn from_midi(midi_note: u8, velocity: f32) -> Self {
        let frequency = midi_to_freq(midi_note);
        let velocity = Velocity::new(T::from_f64(velocity as _));
        let gain = Gain::from_linear(T::one());
        let pan = T::zero();
        let pressure = T::zero();
        Self {
            frequency,
            modulation_st: T::zero(),
            velocity,
            gain,
            pan,
            pressure,
        }
    }

    pub fn resolve_frequency(&self) -> T {
        semitone_to_ratio(self.modulation_st) * self.frequency
    }
}

/// Trait for types which manage voices.
#[allow(unused_variables)]
pub trait VoiceManager:
    DSPMeta<Sample = <<Self as VoiceManager>::Voice as DSPMeta>::Sample>
{
    /// Type of the inner voice.
    type Voice: Voice;
    /// Type for the voice ID.
    type ID: Copy;

    /// Number of voices available in this voice manager
    fn capacity(&self) -> usize;

    /// Get the voice by its ID
    fn get_voice(&self, id: Self::ID) -> Option<&Self::Voice>;
    /// Get the voice mutably by its ID
    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut Self::Voice>;

    /// Return true if the voice referred by the given ID is currently active
    fn is_voice_active(&self, id: Self::ID) -> bool {
        self.get_voice(id).is_some_and(|v| v.active())
    }
    /// Return all the voice IDs that this type manages.
    fn all_voices(&self) -> impl Iterator<Item = Self::ID>;

    /// Count the number of active voices.
    fn active(&self) -> usize {
        self.all_voices()
            .filter(|id| self.is_voice_active(*id))
            .count()
    }

    /// Indicate a note on event, with the given note data to instanciate the voice.
    fn note_on(&mut self, note_data: NoteData<Self::Sample>) -> Self::ID;
    /// Indicate a note off event on the given voice ID.
    fn note_off(&mut self, id: Self::ID, release_velocity: f32);
    /// Choke the voice, causing all processing on that voice to stop.
    fn choke(&mut self, id: Self::ID);
    /// Choke all the notes.
    fn panic(&mut self);

    // Channel modulation
    /// Set the pitch bend amount on the channel
    fn pitch_bend(&mut self, amount: f64) {}
    /// Set the channel aftertouch
    fn aftertouch(&mut self, amount: f64) {}

    // MPE extensions
    /// Note pressure
    fn pressure(&mut self, id: Self::ID, pressure: f32) {}
    /// Note glide
    fn glide(&mut self, id: Self::ID, semitones: f32) {}
    /// Note pan
    fn pan(&mut self, id: Self::ID, pan: f32) {}
    /// Note gain
    fn gain(&mut self, id: Self::ID, gain: f32) {}
}

impl<V: VoiceManager> VoiceManager for BlockAdapter<V> {
    type Voice = V::Voice;
    type ID = V::ID;

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn get_voice(&self, id: Self::ID) -> Option<&Self::Voice> {
        self.0.get_voice(id)
    }

    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut Self::Voice> {
        self.0.get_voice_mut(id)
    }

    fn is_voice_active(&self, id: Self::ID) -> bool {
        self.0.is_voice_active(id)
    }

    fn all_voices(&self) -> impl Iterator<Item = Self::ID> {
        self.0.all_voices()
    }

    fn active(&self) -> usize {
        self.0.active()
    }

    fn note_on(&mut self, note_data: NoteData<Self::Sample>) -> Self::ID {
        self.0.note_on(note_data)
    }

    fn note_off(&mut self, id: Self::ID, release_velocity: f32) {
        self.0.note_off(id, release_velocity)
    }

    fn choke(&mut self, id: Self::ID) {
        self.0.choke(id)
    }

    fn panic(&mut self) {
        self.0.panic()
    }

    fn pitch_bend(&mut self, amount: f64) {
        self.0.pitch_bend(amount)
    }

    fn aftertouch(&mut self, amount: f64) {
        self.0.aftertouch(amount)
    }

    fn pressure(&mut self, id: Self::ID, pressure: f32) {
        self.0.pressure(id, pressure)
    }

    fn glide(&mut self, id: Self::ID, semitones: f32) {
        self.0.glide(id, semitones)
    }

    fn pan(&mut self, id: Self::ID, pan: f32) {
        self.0.pan(id, pan)
    }

    fn gain(&mut self, id: Self::ID, gain: f32) {
        self.0.gain(id, gain)
    }
}

impl<V: DSPProcessBlock<I, O> + VoiceManager, const I: usize, const O: usize> VoiceManager
    for SampleAdapter<V, I, O>
{
    type Voice = V::Voice;
    type ID = V::ID;

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn get_voice(&self, id: Self::ID) -> Option<&Self::Voice> {
        self.inner.get_voice(id)
    }

    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut Self::Voice> {
        self.inner.get_voice_mut(id)
    }

    fn is_voice_active(&self, id: Self::ID) -> bool {
        self.inner.is_voice_active(id)
    }

    fn all_voices(&self) -> impl Iterator<Item = Self::ID> {
        self.inner.all_voices()
    }

    fn active(&self) -> usize {
        self.inner.active()
    }

    fn note_on(&mut self, note_data: NoteData<Self::Sample>) -> Self::ID {
        self.inner.note_on(note_data)
    }

    fn note_off(&mut self, id: Self::ID, release_velocity: f32) {
        self.inner.note_off(id, release_velocity)
    }

    fn choke(&mut self, id: Self::ID) {
        self.inner.choke(id)
    }

    fn panic(&mut self) {
        self.inner.panic()
    }

    fn pitch_bend(&mut self, amount: f64) {
        self.inner.pitch_bend(amount)
    }

    fn aftertouch(&mut self, amount: f64) {
        self.inner.aftertouch(amount)
    }

    fn pressure(&mut self, id: Self::ID, pressure: f32) {
        self.inner.pressure(id, pressure)
    }

    fn glide(&mut self, id: Self::ID, semitones: f32) {
        self.inner.glide(id, semitones)
    }

    fn pan(&mut self, id: Self::ID, pan: f32) {
        self.inner.pan(id, pan)
    }

    fn gain(&mut self, id: Self::ID, gain: f32) {
        self.inner.gain(id, gain)
    }
}

/// Inner voice of the voice manager.
pub type InnerVoice<V> = <V as VoiceManager>::Voice;
/// Inner voice ID of the voice manager.
pub type VoiceId<V> = <V as VoiceManager>::ID;
