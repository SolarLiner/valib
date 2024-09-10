use num_traits::Zero;
use std::collections::Bound;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeBounds};

use crate::Scalar;

/// AudioBuffer abstraction over containers of contiguous slices. This supports owned and non-owned,
/// immutable and mutable slices.
#[derive(Debug, Copy, Clone)]
pub struct AudioBuffer<C, const CHANNELS: usize> {
    containers: [C; CHANNELS],
    inner_size: usize,
}

impl<C> Default for AudioBuffer<C, 0> {
    fn default() -> Self {
        Self {
            containers: [],
            inner_size: 0,
        }
    }
}

impl<C, const CHANNELS: usize> Index<usize> for AudioBuffer<C, CHANNELS> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        &self.containers[index]
    }
}

impl<C, const CHANNELS: usize> IndexMut<usize> for AudioBuffer<C, CHANNELS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.containers[index]
    }
}

impl<C, const CHANNELS: usize> AudioBuffer<C, CHANNELS> {
    /// Returns a reference to the slice associated with the given channel.
    ///
    /// # Arguments
    ///
    /// * `ch`: Channel index
    pub const fn get_channel(&self, ch: usize) -> &C {
        &self.containers[ch]
    }

    /// Returns a mutable reference to the slice associated with the given channel.
    ///
    /// # Arguments
    ///
    /// * `ch`: Channel index
    pub fn get_channel_mut(&mut self, ch: usize) -> &mut C {
        assert!(ch < CHANNELS);
        &mut self.containers[ch]
    }

    /// Number of samples contained in this buffer.
    pub const fn samples(&self) -> usize {
        self.inner_size
    }
}

fn bounds_into_range(
    bounds: impl RangeBounds<usize> + Sized,
    max_bounds: Range<usize>,
) -> Range<usize> {
    let start = match bounds.start_bound().cloned() {
        Bound::Included(i) => i,
        Bound::Excluded(i) => i - 1,
        Bound::Unbounded => max_bounds.start,
    }
    .max(max_bounds.start);
    let end = match bounds.end_bound().cloned() {
        Bound::Included(i) => i + 1,
        Bound::Excluded(i) => i,
        Bound::Unbounded => max_bounds.end,
    }
    .min(max_bounds.end);

    start..end
}

impl<T, const LENGTH: usize, const CHANNELS: usize> AudioBuffer<[T; LENGTH], CHANNELS> {
    pub const fn const_new(containers: [[T; LENGTH]; CHANNELS]) -> Self {
        Self {
            containers,
            inner_size: LENGTH,
        }
    }

    pub fn array_slice(&self, bounds: impl RangeBounds<usize>) -> AudioBufferRef<T, CHANNELS> {
        let range = bounds_into_range(bounds, 0..self.inner_size);
        AudioBuffer {
            containers: std::array::from_fn(|i| &self.containers[i][range.clone()]),
            inner_size: range.len(),
        }
    }

    pub fn array_slice_mut(
        &mut self,
        bounds: impl RangeBounds<usize>,
    ) -> AudioBufferMut<T, CHANNELS> {
        let range = bounds_into_range(bounds, 0..self.inner_size);
        AudioBuffer {
            containers: self.containers.each_mut().map(|i| &mut i[range.clone()]),
            inner_size: range.len(),
        }
    }
}

impl<T, C: Deref<Target = [T]>, const CHANNELS: usize> AudioBuffer<C, CHANNELS> {
    /// Create an audio buffer supported by the given containers of audio slices.
    ///
    /// This method returns `None` when the channels have mismatching lengths.
    pub fn new(containers: [C; CHANNELS]) -> Option<Self> {
        if CHANNELS == 0 {
            return Some(Self {
                containers,
                inner_size: 0,
            });
        }

        let len = containers[0].len();
        if CHANNELS == 1 {
            Some(Self {
                containers,
                inner_size: len,
            })
        } else {
            containers[1..]
                .iter()
                .all(|i| i.len() == len)
                .then_some(Self {
                    containers,
                    inner_size: len,
                })
        }
    }

    pub fn frame_ref(&self, index: usize) -> [&T; CHANNELS] {
        std::array::from_fn(|ch| &self.containers[ch][index])
    }

    /// Get a multi-channel sample at the given index.
    pub fn get_frame(&self, index: usize) -> [T; CHANNELS]
    where
        T: Clone,
    {
        std::array::from_fn(|ch| self.containers[ch][index].clone())
    }

    pub fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = [&'a T; CHANNELS]>
    where
        T: 'a,
    {
        (0..self.inner_size).map(|i| self.frame_ref(i))
    }

    /// Return a non-owning buffer that refers to the content of this audio buffer.
    pub fn as_ref(&self) -> AudioBufferRef<T, CHANNELS> {
        AudioBuffer {
            containers: std::array::from_fn(|i| self.containers[i].deref()),
            inner_size: self.inner_size,
        }
    }

    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> AudioBufferRef<T, CHANNELS> {
        let range = bounds_into_range(bounds, 0..self.inner_size);
        AudioBuffer {
            inner_size: range.len(),
            containers: std::array::from_fn(|i| &self.containers[i][range.clone()]),
        }
    }
}

impl<T, C: DerefMut<Target = [T]>, const CHANNELS: usize> AudioBuffer<C, CHANNELS> {
    /// Return a non-owning mutable buffer that refers to the content of this audio buffer.
    pub fn as_mut(&mut self) -> AudioBufferMut<T, CHANNELS> {
        // We need to use `MaybeUninit` here to be able to split the incoming mutable reference on self into references
        // on the individual channels' slices. The borrow checker cannot see through our intent, and so we need to drop
        // into unsafe code in order to manually create the mutable references ourselves.
        // Note that the only unsafe operation is `MaybeUninit::assume_init` here, as the reference split operation is
        // already a safe method on slices; we only need `MaybeUninit` to create a static array of the correct size for
        // usage in the `AudioBufferMut` we want to create.
        let containers = {
            use std::mem::MaybeUninit;

            let mut containers = std::array::from_fn(|_| MaybeUninit::uninit());
            let mut data = &mut self.containers as &mut [C];
            let mut i = 0;
            while let Some((head, rest)) = data.split_first_mut() {
                containers[i].write(head.deref_mut());
                data = rest;
                i += 1;
            }
            containers.map(|mu| {
                // # Safety
                //
                // This is safe as long as this `MaybeUninit` has successfully been written with a valid, unique (non-
                // overlapping) mutable slice reference, which is guaranteed to be unique by the `split_first_mut`
                // method used above.
                // This block only executes when all references have been successfully created.
                unsafe { mu.assume_init() }
            })
        };
        AudioBuffer {
            containers,
            inner_size: self.inner_size,
        }
    }

    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        for container in &mut self.containers {
            container.fill(value)
        }
    }

    pub fn fill_with(&mut self, mut fill: impl FnMut() -> T) {
        for container in &mut self.containers {
            container.fill_with(&mut fill);
        }
    }

    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> AudioBufferMut<T, CHANNELS> {
        let range = bounds_into_range(bounds, 0..self.inner_size);
        AudioBuffer {
            inner_size: range.len(),
            containers: self.containers.each_mut().map(|i| &mut i[range.clone()]),
        }
    }
}

impl<T: Copy, C: DerefMut<Target = [T]>, const CHANNELS: usize> AudioBuffer<C, CHANNELS> {
    /// Copy a slice into a specific channel of this audio buffer.
    ///
    /// The buffers must match length, as reported by [`Self::samples()`].
    pub fn copy_from_slice(&mut self, ch: usize, slice: &[T]) {
        self.containers[ch].copy_from_slice(slice);
    }

    /// Copy a buffer into this buffer.
    ///
    /// The buffers must match length, as reported by [`Self::samples()`].
    pub fn copy_from(&mut self, buffer: AudioBufferRef<T, CHANNELS>) {
        for i in 0..CHANNELS {
            self.containers[i].copy_from_slice(buffer.containers[i]);
        }
    }

    /// Set a multi-channel sample at the given index.
    pub fn set_frame(&mut self, index: usize, frame: [T; CHANNELS]) {
        for (channel, sample) in self.containers.iter_mut().zip(frame.iter().copied()) {
            channel[index] = sample;
        }
    }
}

impl<T: Scalar, C: DerefMut<Target = [T]>, const CHANNELS: usize> AudioBuffer<C, CHANNELS> {
    /// Mix another buffer into this audio buffer, at the specified per-channel gain.
    pub fn mix<C2: Deref<Target = [T]>>(
        &mut self,
        other: &AudioBuffer<C2, CHANNELS>,
        mix: [T; CHANNELS],
    ) {
        assert_eq!(self.inner_size, other.inner_size);
        for ((this_channel, other_channel), mix) in self
            .containers
            .iter_mut()
            .zip(other.containers.iter())
            .zip(mix.iter().copied())
        {
            for i in 0..self.inner_size {
                this_channel[i] += other_channel[i] * mix;
            }
        }
    }
}

pub type AudioBufferRef<'a, T, const CHANNELS: usize> = AudioBuffer<&'a [T], CHANNELS>;

impl<'a, T> From<&'a [T]> for AudioBufferRef<'a, T, 1> {
    fn from(value: &'a [T]) -> Self {
        let inner_size = value.len();
        Self {
            containers: [value],
            inner_size,
        }
    }
}

impl<C> AudioBuffer<C, 0> {
    /// Creates a 0-channel empty buffer with the specified buffer size. This constructor is required to provide a non-
    /// zero block size that matches the companion buffer passed into `process_block`.
    ///
    /// Better API design is needed to remove this need.
    pub fn empty(block_size: usize) -> Self {
        Self {
            containers: [],
            inner_size: block_size,
        }
    }
}

pub type AudioBufferMut<'a, T, const CHANNELS: usize> = AudioBuffer<&'a mut [T], CHANNELS>;

impl<'a, T> From<&'a mut [T]> for AudioBufferMut<'a, T, 1> {
    fn from(value: &'a mut [T]) -> Self {
        let inner_size = value.len();
        Self {
            containers: [value],
            inner_size,
        }
    }
}

pub type AudioBufferBox<T, const CHANNELS: usize> = AudioBuffer<Box<[T]>, CHANNELS>;

impl<T> FromIterator<T> for AudioBufferBox<T, 1> {
    fn from_iter<It: IntoIterator<Item = T>>(iter: It) -> Self {
        let slice: Box<[T]> = iter.into_iter().collect();
        let inner_size = slice.len();
        Self {
            containers: [slice],
            inner_size,
        }
    }
}

impl<T: Zero, const CHANNELS: usize> AudioBufferBox<T, CHANNELS> {
    pub fn zeroed(size: usize) -> Self {
        Self {
            containers: std::array::from_fn(|_| {
                std::iter::repeat_with(T::zero).take(size).collect()
            }),
            inner_size: size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_mut_write() {
        let mut buffer = AudioBuffer::<_, 1>::zeroed(1);
        let mut slice_mut = buffer.as_mut();
        slice_mut[0][0] = 1;

        assert_eq!(1, buffer[0][0]);
    }
}
