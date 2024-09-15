#![warn(missing_docs)]
#![feature(generic_const_exprs)]
//! # `fundsp` integration into `valib`
//! fundsp integration for statically-defined graphs.
//!
//! The integration provides impls for `An` objects, taking their defined input and output counts as the number of
//! input and output channels for the [`DSPProcess`] implementation.
//!
//! Conversly, a [`DspNode`] struct is defined for wrapping [`DSPProcess`] implementations into usable `fundsp` nodes.

use fundsp::audionode::{AudioNode, Frame};
use fundsp::combinator::An;
use numeric_array::ArrayLength;
use typenum::{Const, ToUInt, Unsigned, U};
use valib_core::dsp::{DSPMeta, DSPProcess};

/// Wrapper DSP processor for FunDSP nodes
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
impl<Node: AudioNode>
    DSPProcess<{ <Node::Inputs as Unsigned>::USIZE }, { <Node::Outputs as Unsigned>::USIZE }>
    for FunDSP<Node>
{
    fn process(
        &mut self,
        x: [Self::Sample; <Node::Inputs as Unsigned>::USIZE],
    ) -> [Self::Sample; <Node::Outputs as Unsigned>::USIZE] {
        let input = Frame::from_slice(&x);
        let output = self.0.tick(input);
        std::array::from_fn(|i| output[i])
    }
}

/// Wrap a [`DSPProcess`] impl as a `fundsp`  node.
///
/// This is the implementation struct; to us this node in `fundsp` graphs, refer to the [`dsp_node`] function.
#[derive(Debug, Clone)]
pub struct DspNode<P, const I: usize, const O: usize>(pub P);

impl<P: Send + Sync + Clone + DSPProcess<I, O, Sample = f32>, const I: usize, const O: usize>
    AudioNode for DspNode<P, I, O>
where
    Self: Clone,
    Const<I>: ToUInt,
    <Const<I> as ToUInt>::Output: ArrayLength + Send + Sync,
    Const<O>: ToUInt,
    <Const<O> as ToUInt>::Output: ArrayLength + Send + Sync,
{
    const ID: u64 = 28000;
    type Inputs = U<I>;
    type Outputs = U<O>;

    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let input = std::array::from_fn(|i| input[i]);
        let output = self.0.process(input);
        Frame::from_iter(output)
    }
}

/// Wrap a [`DSPProcess`] impl as a [`fundsp`]  node.
pub fn dsp_node<
    P: Send + Sync + Clone + DSPProcess<I, O, Sample = f32>,
    const I: usize,
    const O: usize,
>(
    dsp: P,
) -> An<DspNode<P, I, O>>
where
    Const<I>: ToUInt,
    <Const<I> as ToUInt>::Output: ArrayLength + Send + Sync,
    Const<O>: ToUInt,
    <Const<O> as ToUInt>::Output: ArrayLength + Send + Sync,
{
    An(DspNode(dsp))
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
        let mut integrator_node = dsp_node::<_, 1, 1>(Integrator::<f32>::default());
        integrator_node.filter_mono(0.0);
        integrator_node.filter_mono(1.0);
        let actual = integrator_node.filter_mono(0.0);
        let expected = 1.0;

        assert_eq!(expected, actual);
    }
}
