//! fundsp integration for statically-defined graphs.
//!
//! The integration provides impls for `An` objects, taking their defined input and output counts as the number of
//! input and output channels for the [`DSPProcess`] implementation.
//!
//! Conversly, a [`DspNode`] struct is defined for wrapping [`DSPProcess`] implementations into usable `fundsp` nodes.

use crate::dsp::{DSPMeta, DSPProcess};
use fundsp::audionode::{AudioNode, Frame};
use fundsp::combinator::An;
use fundsp::Float;
use numeric_array::ArrayLength;
use std::marker::PhantomData;
use typenum::Unsigned;

impl<Node: AudioNode> DSPMeta for An<Node>
where
    Node::Sample: crate::Scalar,
{
    type Sample = Node::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        An::set_sample_rate(self, samplerate as _);
    }

    fn reset(&mut self) {
        An::reset(self);
    }
}

#[profiling::all_functions]
impl<Node: AudioNode> DSPProcess<{ Node::Inputs::USIZE }, { Node::Outputs::USIZE }> for An<Node>
where
    Node::Sample: crate::Scalar,
{
    fn process(
        &mut self,
        x: [Self::Sample; Node::Inputs::USIZE],
    ) -> [Self::Sample; Node::Outputs::USIZE] {
        let input = Frame::from_slice(&x);
        let output = self.tick(input);
        std::array::from_fn(|i| output[i])
    }
}

/// Wrap a [`DSPProcess`] impl as a `fundsp`  node.
///
/// This is the implementation struct; to us this node in `fundsp` graphs, refer to the [`dsp_node`] function.
#[derive(Debug, Clone)]
pub struct DspNode<In, Out, P>(PhantomData<In>, PhantomData<Out>, P);

/// Wrap a [`DSPProcess`] impl as a [`fundsp`]  node.
pub fn dsp_node<In: Unsigned, Out: Unsigned, P: DSPProcess<{ In::USIZE }, { Out::USIZE }>>(
    dsp: P,
) -> DspNode<In, Out, P> {
    DspNode(PhantomData, PhantomData, dsp)
}

impl<
        In: Send + Sync + Unsigned,
        Out: Send + Sync + Unsigned,
        P: Send + Sync + DSPProcess<{ In::USIZE }, { Out::USIZE }>,
    > AudioNode for DspNode<In, Out, P>
where
    Self: Clone,
    P::Sample: Float,
    In: ArrayLength<P::Sample>,
    Out: ArrayLength<P::Sample>,
{
    const ID: u64 = 0;
    type Sample = P::Sample;
    type Inputs = In;
    type Outputs = Out;
    type Setting = ();

    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let input = std::array::from_fn(|i| input[i]);
        let output = self.2.process(input);
        Frame::from_iter(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::dsp::{buffer::AudioBufferBox, BlockAdapter, DSPProcessBlock};

    use crate::dsp::blocks::Integrator;
    use fundsp::hacker32::*;

    use super::*;

    #[test]
    fn test_wrapper() {
        let mut dsp = BlockAdapter(sine_hz(440.0) * sine_hz(10.0));
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
