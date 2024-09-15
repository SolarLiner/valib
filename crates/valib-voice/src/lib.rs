#![warn(missing_docs)]
//! # Voice abstractions
//!
//! This crate provides abstractions around voice processing and voice management.
use valib_core::dsp::DSPMeta;
use valib_core::simd::SimdRealField;
use valib_core::Scalar;

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
    fn release(&mut self);
    /// Reuse the note (corresponding to a soft reset)
    fn reuse(&mut self);
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
    /// Note velocity
    pub velocity: Velocity<T>,
    /// Note gain
    pub gain: Gain<T>,
    /// Note pan
    pub pan: T,
    /// Note pressure
    pub pressure: T,
}

/// Trait for types which manage voices.
#[allow(unused_variables)]
pub trait VoiceManager<V: Voice>: DSPMeta<Sample = V::Sample> {
    /// Type for the voice ID.
    type ID: Copy;

    /// Number of voices available in this voice manager
    fn capacity(&self) -> usize;

    /// Get the voice by its ID
    fn get_voice(&self, id: Self::ID) -> Option<&V>;
    /// Get the voice mutably by its ID
    fn get_voice_mut(&mut self, id: Self::ID) -> Option<&mut V>;

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
    fn note_on(&mut self, note_data: NoteData<V::Sample>) -> Self::ID;
    /// Indicate a note off event on the given voice ID.
    fn note_off(&mut self, id: Self::ID);
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
