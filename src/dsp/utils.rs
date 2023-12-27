/// Transmutes a mono block slice into a regular slice.
/// Safety: safe because [T; 1] has the same representation as T, so in turn
/// any access of &[T; 1] is also a valid access of &T.
#[inline(always)]
pub fn mono_block_to_slice<T>(inputs: &[[T; 1]]) -> &[T] {
    unsafe { std::mem::transmute(inputs) }
}

/// Transmutes a mutable mono block slice into a regular slice.
/// Safety: safe because [T; 1] has the same representation as T, so in turn
/// any access of &mut [T; 1] is also a valid access of &mut T.
#[inline(always)]
pub fn mono_block_to_slice_mut<T>(inputs: &mut [[T; 1]]) -> &mut [T] {
    unsafe { std::mem::transmute(inputs) }
}

/// Transmutes a slice into a mono block slice.
/// Safety: safe because T has the same representation as [T; 1], so in turn
/// any access of &T is also a valid access of &[T; 1].
#[inline(always)]
pub fn slice_to_mono_block<T>(inputs: &[T]) -> &[[T; 1]] {
    unsafe { std::mem::transmute(inputs) }
}

/// Transmutes a mutable slice into a mutable mono block slice.
/// Safety: safe because T has the same representation as [T; 1], so in turn
/// any access of &mut T is also a valid access of &mut [T; 1].
#[inline(always)]
pub fn slice_to_mono_block_mut<T>(inputs: &mut [T]) -> &mut [[T; 1]] {
    unsafe { std::mem::transmute(inputs) }
}
