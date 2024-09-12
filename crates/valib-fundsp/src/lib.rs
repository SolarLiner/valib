#![feature(generic_const_exprs)]
//! fundsp integration for statically-defined graphs.
//!
//! The integration provides impls for `An` objects, taking their defined input and output counts as the number of
//! input and output channels for the [`DSPProcess`] implementation.
//!
//! Conversly, a [`DspNode`] struct is defined for wrapping [`DSPProcess`] implementations into usable `fundsp` nodes.

use fundsp::audionode::{AudioNode, Frame};
use fundsp::combinator::An;
use fundsp::prelude::Size;
use fundsp::Float;
use numeric_array::ArrayLength;
use std::marker::PhantomData;
use typenum::{Const, Unsigned};
use valib_core::dsp::{DSPMeta, DSPProcess};

pub struct FunDSP<Node: AudioNode>(pub An<Node>);

impl<Node: AudioNode> DSPMeta for FunDSP<Node> {
    type Sample = f32;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.0.set_sample_rate(samplerate as _);
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

#[profiling::all_functions]
impl<Node: AudioNode<Inputs = Const<I>, Outputs = Const<O>>, const I: usize, const O: usize>
    DSPProcess<I, O> for FunDSP<Node>
{
    fn process(&mut self, x: [Self::Sample; I]) -> [Self::Sample; O] {
        let input = Frame::from_slice(&x);
        let output = self.tick(input);
        std::array::from_fn(|i| output[i])
    }
}

/// Wrap a [`DSPProcess`] impl as a `fundsp`  node.
///
/// This is the implementation struct; to us this node in `fundsp` graphs, refer to the [`dsp_node`] function.
#[derive(Debug, Clone)]
pub struct DspNode<P, const I: usize, const O: usize>(pub P);

/// Wrap a [`DSPProcess`] impl as a [`fundsp`]  node.
pub fn dsp_node<P: DSPProcess<I, O>, const I: usize, const O: usize>(dsp: P) -> DspNode<P, I, O> {
    DspNode(dsp)
}

impl<P: Send + Sync + DSPProcess<I, O, Sample = f32>, const I: usize, const O: usize> AudioNode
    for DspNode<P, I, O>
where
    Self: Clone,
    P::Sample: Float,
    Const<I>: Size<f32>,
    Const<O>: Size<f32>,
{
    const ID: u64 = 0;
    type Inputs = Const<I>;
    type Outputs = Const<O>;

    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let input = std::array::from_fn(|i| input[i]);
        let output = self.0.process(input);
        Frame::from_iter(output)
    }
}

#[cfg(test)]
mod tests {
    use valib_core::dsp::{buffer::AudioBufferBox, BlockAdapter, DSPProcessBlock};

    use fundsp::hacker32::*;
    use valib_core::dsp::blocks::Integrator;

    use super::*;

    #[test]
    fn test_wrapper() {
        let mut dsp = BlockAdapter(FunDSP(sine_hz(440.0) * sine_hz(10.0)));
        let input = AudioBufferBox::zeroed(512);
        let mut output = AudioBufferBox::zeroed(512);
        dsp.process_block(input.as_ref(), output.as_mut());

        insta::assert_csv_snapshot!(output.get_channel(0), { "[]" => insta::rounded_redaction(3) })
    }

    #[test]
    fn test_dsp_node() {
        let mut integrator_node = dsp_node::<U1, U1, _>(Integrator::<f32>::default());
        integrator_node.filter_mono(0.0);
        integrator_node.filter_mono(1.0);
        let actual = integrator_node.filter_mono(0.0);
        let expected = 1.0;

        assert_eq!(expected, actual);
    }
}
