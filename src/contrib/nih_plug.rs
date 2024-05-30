#![cfg(feature = "nih-plug")]

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::dsp::buffer::AudioBuffer;
use nih_plug::nih_debug_assert;
use nih_plug::params::FloatParam;
use nih_plug::prelude::*;
use nih_plug::{buffer::Buffer, params::Param};

use crate::dsp::parameter::{HasParameters, ParamName, RemoteControl};
use crate::dsp::DSPProcessBlock;
use crate::Scalar;

pub fn enum_int_param<E: Enum + ToString>(
    param_name: impl Into<String>,
    default_value: E,
) -> IntParam {
    IntParam::new(
        param_name,
        default_value.into_usize() as _,
        IntRange::Linear {
            min: 0,
            max: (E::LENGTH - 1) as _,
        },
    )
    .with_value_to_string(Arc::new(|x| E::from_usize(x as _).to_string()))
}

/// Bind a [`valib`] [`Parameter`] to a [`nig_plug`] parameter..
pub trait BindToParameter<P: ParamName> {
    /// Bind a [`Parameter`] to a nih-plug [`FloatParam`].
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self;
}

impl<P: ParamName> BindToParameter<P> for FloatParam {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |value| set.set_parameter(param, value)))
    }
}

impl<P: ParamName> BindToParameter<P> for IntParam {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |x| set.set_parameter(param, x as _)))
    }
}

impl<P: ParamName> BindToParameter<P> for BoolParam {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |b| {
            set.set_parameter(param, if b { 1.0 } else { 0.0 })
        }))
    }
}

impl<E: 'static + PartialEq + Enum, P: ParamName> BindToParameter<P> for EnumParam<E> {
    fn bind_to_parameter(self, set: &RemoteControl<P>, param: P) -> Self {
        let set = set.clone();
        self.with_callback(Arc::new(move |e| {
            set.set_parameter(param, e.into_index() as _)
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
pub fn process_buffer_simd<
    T: Scalar<Element = f32>,
    Dsp: DSPProcessBlock<1, 1, Sample = T>,
    const MAX_BUF_SIZE: usize,
>(
    dsp: &mut Dsp,
    buffer: &mut Buffer,
) {
    let channels = buffer.channels();
    assert!(T::lanes() <= channels);
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
