/// Transmutes a mono block slice into a regular slice.
#[inline(always)]
pub fn mono_block_to_slice<T>(inputs: [&[T]; 1]) -> &[T] {
    inputs[0]
}

/// Transmutes a mutable mono block slice into a regular slice.
#[inline(always)]
pub fn mono_block_to_slice_mut<T>(inputs: [&mut [T]; 1]) -> &mut [T] {
    inputs[0]
}

/// Transmutes a slice into a mono block slice.
#[inline(always)]
pub fn slice_to_mono_block<T>(inputs: &[T]) -> [&[T]; 1] {
    [inputs]
}

/// Transmutes a mutable slice into a mutable mono block slice.
#[inline(always)]
pub fn slice_to_mono_block_mut<T>(inputs: &mut [T]) -> [&mut [T]; 1] {
    [inputs]
}
