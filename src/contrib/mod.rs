#![doc = include_str!("./README.md")]
#[cfg(feature = "fundsp")]
pub mod fundsp;
#[cfg(feature = "nih-plug")]
pub mod nih_plug;
