#![warn(missing_docs)]
//! # `nih-plug` integration into `valib`
//!
//! This crates provides integrations of `valib`'s parameters into `nih-plug` parameter system, as
//! well as functions to drive processors with `nih-plug`'s [`Buffer`] type.
use std::sync::Arc;

use nih_plug::buffer::Buffer;
use nih_plug::nih_debug_assert;
use nih_plug::params::FloatParam;
use nih_plug::prelude::*;
use valib_core::dsp::buffer::AudioBuffer;

use valib_core::dsp::parameter::{ParamName, RemoteControl};
use valib_core::dsp::DSPProcessBlock;
use valib_core::Scalar;

/// Bind a [`valib`] [`Parameter`] to a [`nig_plug`] parameter.
pub trait BindToParameter<P: ParamName> {
    /// Bind a [`Parameter`] to a nih-plug [`FloatParam`].
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self;
}

impl<P: 'static + Send + Sync + ParamName> BindToParameter<P> for FloatParam {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |value| {
            profiling::scope!(
                "set_parameter",
                &format!("{}: {value}", param.name().as_ref())
            );
            set.set_parameter(param, value)
        }))
    }
}

impl<P: 'static + Send + Sync + ParamName> BindToParameter<P> for IntParam {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |x| {
            profiling::scope!("set_parameter", &format!("{}: {x}", param.name().as_ref()));
            set.set_parameter(param, x as _)
        }))
    }
}

impl<P: 'static + Send + Sync + ParamName> BindToParameter<P> for BoolParam {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |b| {
            profiling::scope!("set_parameter", &format!("{}: {b}", param.name().as_ref()));
            set.set_parameter(param, if b { 1.0 } else { 0.0 })
        }))
    }
}

impl<E: 'static + PartialEq + Enum, P: 'static + Send + Sync + ParamName> BindToParameter<P>
    for EnumParam<E>
{
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |e| {
            let ix = e.to_index();
            profiling::scope!(
                "set_parameter",
                &format!("{}: {}", param.name().as_ref(), E::variants()[ix])
            );
            set.set_parameter(param, ix as _)
        }))
    }
}

/// Processes a [`nih-plug`] buffer in its entirety with a [`DSPBlock`] instance, where inputs in
/// the dsp instance correspond to channels in the buffer.
///
/// # Arguments
///
/// * `dsp`: [`DSPBlock`] instance to process the buffer with
/// * `buffer`: Buffer to process
///
/// panics if the scalar type has more channels than the buffer holds.
#[profiling::function]
pub fn process_buffer<
    T: Scalar<Element = f32>,
    Dsp,
    const CHANNELS: usize,
    const MAX_BUF_SIZE: usize,
>(
    dsp: &mut Dsp,
    buffer: &mut Buffer,
) where
    Dsp: DSPProcessBlock<CHANNELS, CHANNELS, Sample = T>,
{
    assert_eq!(
        CHANNELS,
        buffer.channels(),
        "Channel mismatch between nih-plug channel count and requested buffer size"
    );
    let mut input = AudioBuffer::const_new([[T::zero(); MAX_BUF_SIZE]; CHANNELS]);
    let mut output = input;
    let max_buffer_size = dsp
        .max_block_size()
        .map(|mbf| mbf.min(MAX_BUF_SIZE))
        .unwrap_or(MAX_BUF_SIZE);

    for (_, mut block) in buffer.iter_blocks(max_buffer_size) {
        let mut input = input.array_slice_mut(..block.samples());
        let mut output = output.array_slice_mut(..block.samples());
        for (i, mut s) in block.iter_samples().enumerate() {
            let mut frame = [T::zero(); CHANNELS];
            for (ch, s) in s.iter_mut().map(|s| *s).enumerate() {
                frame[ch] = T::splat(s);
            }
            input.set_frame(i, frame);
        }

        dsp.process_block(input.as_ref(), output.as_mut());

        for (i, mut s) in block.iter_samples().enumerate() {
            for (ch, s) in s.iter_mut().enumerate() {
                *s = output.get_frame(i)[ch].extract(0);
            }
        }
    }
}

/// Processes a [`nih-plug`] buffer in its entirety with a [`DSPBlock`] instance, mapping channels
/// to lanes in the scalar type.
///
/// This function automatically respects the value reported by [`DSPBlock::max_buffer_size`]. Up to
/// [`MAX_BUF_SIZE`] samples will be processed at once.
///
/// # Arguments
///
/// * `dsp`: [`DSPBlock`] instance to process the buffer with
/// * `buffer`: Buffer to process
///
/// panics if the scalar type has more channels than the buffer holds.
#[profiling::function]
pub fn process_buffer_simd<
    T: Scalar<Element = f32>,
    Dsp: DSPProcessBlock<1, 1, Sample = T>,
    const MAX_BUF_SIZE: usize,
>(
    dsp: &mut Dsp,
    buffer: &mut Buffer,
) {
    let channels = buffer.channels();
    assert!(T::LANES <= channels);
    let mut input = AudioBuffer::const_new([[T::from_f64(0.0); MAX_BUF_SIZE]]);
    let mut output = input;
    let max_buffer_size = dsp.max_block_size().unwrap_or(MAX_BUF_SIZE);
    nih_debug_assert!(max_buffer_size <= MAX_BUF_SIZE);
    for (_, mut block) in buffer.iter_blocks(max_buffer_size) {
        let mut input = input.array_slice_mut(..block.samples());
        let mut output = output.array_slice_mut(..block.samples());
        for (i, mut c) in block.iter_samples().enumerate() {
            let mut frame = T::zero();
            for (ch, s) in c.iter_mut().enumerate() {
                frame.replace(ch, *s);
            }
            input.set_frame(i, [frame]);
        }
        output.copy_from(input.as_ref());

        dsp.process_block(input.as_ref(), output.as_mut());

        for (i, mut c) in block.iter_samples().enumerate() {
            for (ch, s) in c.iter_mut().enumerate() {
                *s = output.get_frame(i)[0].extract(ch);
            }
        }
    }
}

/// Processes a [`nih-plug`] buffer in its entirety with a [`DSPBlock`] instance, mapping channels
/// to lanes in the scalar type.
///
/// This function automatically respects the value reported by [`DSPBlock::max_buffer_size`]. Up to
/// [`MAX_BUF_SIZE`] samples will be processed at once.
///
/// # Arguments
///
/// * `dsp`: [`DSPBlock`] instance to process the buffer with
/// * `buffer`: Buffer to process
///
/// panics if the scalar type has more channels than the buffer holds.
#[profiling::function]
pub fn process_buffer_simd64<
    T: Scalar<Element = f64>,
    Dsp: DSPProcessBlock<1, 1, Sample = T>,
    const MAX_BUF_SIZE: usize,
>(
    dsp: &mut Dsp,
    buffer: &mut Buffer,
) {
    let channels = buffer.channels();
    assert!(T::LANES <= channels);
    let mut input = AudioBuffer::const_new([[T::from_f64(0.0); MAX_BUF_SIZE]]);
    let mut output = input;
    let max_buffer_size = dsp.max_block_size().unwrap_or(MAX_BUF_SIZE);
    nih_debug_assert!(max_buffer_size <= MAX_BUF_SIZE);
    for (_, mut block) in buffer.iter_blocks(max_buffer_size) {
        let mut input = input.array_slice_mut(..block.samples());
        let mut output = output.array_slice_mut(..block.samples());
        for (i, mut c) in block.iter_samples().enumerate() {
            let mut frame = T::zero();
            for (ch, s) in c.iter_mut().enumerate() {
                frame.replace(ch, *s as f64);
            }
            input.set_frame(i, [frame]);
        }
        output.copy_from(input.as_ref());

        dsp.process_block(input.as_ref(), output.as_mut());

        for (i, mut c) in block.iter_samples().enumerate() {
            for (ch, s) in c.iter_mut().enumerate() {
                *s = output.get_frame(i)[0].extract(ch) as f32;
            }
        }
    }
}
