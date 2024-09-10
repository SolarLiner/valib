#![cfg_attr(feature = "biquad-design", feature(iter_array_chunks))]
#![cfg_attr(feature = "fundsp", feature(generic_const_exprs))]
#![cfg_attr(feature = "wdf", feature(downcast_unchecked))]
#![doc = include_str!("./README.md")]
extern crate core;

use az::CastFrom;
use num_traits::Zero;
use simba::simd::{AutoSimd, SimdRealField, SimdValue};

#[cfg(feature = "fundsp")]
pub use contrib::fundsp;
pub use simba::simd;
use valib_core::SimdCast;

pub mod benchmarking;
pub mod contrib;
pub mod dsp;
pub mod filters;
pub mod fir;
pub mod math;
pub mod oscillators;
#[cfg(feature = "oversample")]
pub mod oversample;
pub mod saturators;
pub mod util;
pub mod voice;
#[cfg(feature = "wdf")]
pub mod wdf;
