#![cfg(feature = "nih-plug")]

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use enum_map::{Enum, EnumArray, EnumMap};
use nih_plug::nih_debug_assert;
use nih_plug::params::FloatParam;
use nih_plug::prelude::*;
use nih_plug::{buffer::Buffer, params::Param};

use crate::dsp::parameter::{HasParameters, Parameter};
use crate::dsp::utils::{slice_to_mono_block, slice_to_mono_block_mut};
use crate::dsp::DSPBlock;
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
pub trait BindToParameter {
    /// Bind a [`Parameter`] to a nih-plug [`FloatParam`].
    fn bind_to_parameter(self, param: &Parameter) -> Self;
}

impl BindToParameter for FloatParam {
    fn bind_to_parameter(self, param: &Parameter) -> Self {
        let param = param.clone();
        self.with_callback(Arc::new(move |value| param.set_value(value)))
    }
}

impl BindToParameter for IntParam {
    fn bind_to_parameter(self, param: &Parameter) -> Self {
        let param = param.clone();
        self.with_callback(Arc::new(move |x| param.set_value(x as _)))
    }
}

impl BindToParameter for BoolParam {
    fn bind_to_parameter(self, param: &Parameter) -> Self {
        let param = param.clone();
        self.with_callback(Arc::new(move |b| param.set_bool(b)))
    }
}

impl<E: 'static + PartialEq + Enum + nih_plug::params::enums::Enum> BindToParameter
    for EnumParam<E>
{
    fn bind_to_parameter(self, param: &Parameter) -> Self {
        let param = param.clone();
        self.with_callback(Arc::new(move |e| param.set_enum(e)))
    }
}

#[derive(Debug)]
pub enum AnyParam {
    FloatParam(FloatParam),
    IntParam(IntParam),
    BoolParam(BoolParam),
}

impl fmt::Display for AnyParam {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::FloatParam(fp) => write!(f, "{}", fp),
            Self::IntParam(fp) => write!(f, "{}", fp),
            Self::BoolParam(fp) => write!(f, "{}", fp),
        }
    }
}

impl From<FloatParam> for AnyParam {
    fn from(value: FloatParam) -> Self {
        Self::FloatParam(value)
    }
}

impl From<IntParam> for AnyParam {
    fn from(value: IntParam) -> Self {
        Self::IntParam(value)
    }
}

impl From<BoolParam> for AnyParam {
    fn from(value: BoolParam) -> Self {
        Self::BoolParam(value)
    }
}

impl BindToParameter for AnyParam {
    fn bind_to_parameter(self, param: &Parameter) -> Self {
        match self {
            Self::FloatParam(fp) => Self::FloatParam(fp.bind_to_parameter(param)),
            Self::IntParam(ip) => Self::IntParam(ip.bind_to_parameter(param)),
            Self::BoolParam(bp) => Self::BoolParam(bp.bind_to_parameter(param)),
        }
    }
}

impl AnyParam {
    pub fn as_ptr(&self) -> ParamPtr {
        match self {
            Self::FloatParam(fp) => fp.as_ptr(),
            Self::IntParam(fp) => fp.as_ptr(),
            Self::BoolParam(fp) => fp.as_ptr(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::FloatParam(fp) => fp.name(),
            Self::IntParam(ip) => ip.name(),
            Self::BoolParam(bp) => bp.name(),
        }
    }
}

/// Adapter struct for processors with parameters
#[derive(Debug)]
pub struct NihParamsController<P: HasParameters>
where
    P::Enum: EnumArray<AnyParam>,
{
    nih_map: EnumMap<P::Enum, AnyParam>,
}

impl<P: HasParameters> NihParamsController<P>
where
    P::Enum: EnumArray<AnyParam>,
{
    pub fn new(inner: &P, param_map: impl Fn(P::Enum, String) -> AnyParam) -> Self {
        let nih_map = EnumMap::from_fn(|k| {
            param_map(k, inner.full_name(k)).bind_to_parameter(inner.get_parameter(k))
        });
        Self { nih_map }
    }
}

unsafe impl<P: 'static + HasParameters> Params for NihParamsController<P>
where
    P::Enum: 'static + Send + Sync + EnumArray<AnyParam>,
    <P::Enum as EnumArray<AnyParam>>::Array: Send + Sync,
{
    fn param_map(&self) -> Vec<(String, ParamPtr, String)> {
        self.nih_map
            .iter()
            .map(|(k, p)| (k.into_usize().to_string(), p.as_ptr(), p.name().to_string()))
            .collect()
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
    Dsp: DSPBlock<CHANNELS, CHANNELS, Sample = T>,
{
    assert!(CHANNELS <= buffer.channels());
    let mut input = [[T::zero(); CHANNELS]; MAX_BUF_SIZE];
    let mut output = input;
    let max_buffer_size = dsp
        .max_block_size()
        .map(|mbf| mbf.min(MAX_BUF_SIZE))
        .unwrap_or(MAX_BUF_SIZE);

    for (_, mut block) in buffer.iter_blocks(max_buffer_size) {
        let input = &mut input[..block.samples()];
        let output = &mut output[..block.samples()];
        for (i, mut s) in block.iter_samples().enumerate() {
            for (ch, s) in s.iter_mut().map(|s| *s).enumerate() {
                input[i][ch] = T::splat(s);
            }
        }

        dsp.process_block(input, output);

        for (i, mut s) in block.iter_samples().enumerate() {
            for (ch, s) in s.iter_mut().enumerate() {
                *s = output[i][ch].extract(0);
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
    Dsp: DSPBlock<1, 1, Sample = T>,
    const MAX_BUF_SIZE: usize,
>(
    dsp: &mut Dsp,
    buffer: &mut Buffer,
) {
    let channels = buffer.channels();
    assert!(T::lanes() <= channels);
    let mut input = [T::from_f64(0.0); MAX_BUF_SIZE];
    let mut output = input;
    let max_buffer_size = dsp.max_block_size().unwrap_or(MAX_BUF_SIZE);
    nih_debug_assert!(max_buffer_size <= MAX_BUF_SIZE);
    for (_, mut block) in buffer.iter_blocks(max_buffer_size) {
        for (i, mut c) in block.iter_samples().enumerate() {
            for ch in 0..channels {
                input[i].replace(ch, c.get_mut(ch).copied().unwrap());
            }
            output[i] = input[i];
        }

        let input = &input[..block.samples()];
        let output = &mut output[..block.samples()];

        dsp.process_block(slice_to_mono_block(input), slice_to_mono_block_mut(output));

        for (i, mut c) in block.iter_samples().enumerate() {
            for (ch, s) in c.iter_mut().enumerate() {
                *s = output[i].extract(ch);
            }
        }
    }
}
