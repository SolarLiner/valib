use crate::dsp::DSP;
use fundsp::audionode::{AudioNode, Frame};
use fundsp::combinator::An;
use fundsp::Float;
use numeric_array::ArrayLength;
use std::marker::PhantomData;
use typenum::Unsigned;

impl<Node: AudioNode> DSP<{ Node::Inputs::USIZE }, { Node::Outputs::USIZE }> for An<Node>
where
    Node::Sample: crate::Scalar,
{
    type Sample = Node::Sample;

    fn process(
        &mut self,
        x: [Self::Sample; Node::Inputs::USIZE],
    ) -> [Self::Sample; Node::Outputs::USIZE] {
        let input = Frame::from_slice(&x);
        let output = self.tick(input);
        std::array::from_fn(|i| output[i])
    }
}

#[derive(Debug, Clone)]
pub struct DspNode<In, Out, P>(PhantomData<In>, PhantomData<Out>, P);

pub fn dsp_node<In: Unsigned, Out: Unsigned, P: DSP<{ In::USIZE }, { Out::USIZE }>>(
    dsp: P,
) -> DspNode<In, Out, P> {
    DspNode(PhantomData, PhantomData, dsp)
}

impl<
        In: Send + Sync + Unsigned,
        Out: Send + Sync + Unsigned,
        P: Send + Sync + DSP<{ In::USIZE }, { Out::USIZE }>,
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
        let output = self.2.process(input.into());
        Frame::from_iter(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::dsp::{utils::slice_to_mono_block_mut, DSPBlock};

    use crate::dsp::blocks::Integrator;
    use fundsp::hacker32::*;

    use super::*;

    #[test]
    fn test_wrapper() {
        let mut dsp = sine_hz(440.0) * sine_hz(10.0);
        let input = [[]; 512];
        let mut output = [0.0; 512];
        dsp.process_block(&input, slice_to_mono_block_mut(&mut output));

        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(3) })
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
