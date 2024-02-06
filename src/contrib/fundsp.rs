use crate::dsp::DSP;
use fundsp::audionode::{AudioNode, Frame};
use fundsp::combinator::An;
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

#[cfg(test)]
mod tests {
    use crate::dsp::{utils::slice_to_mono_block_mut, DSPBlock};

    use fundsp::hacker32::*;

    #[test]
    fn test_wrapper() {
        let mut dsp = sine_hz(440.0) * sine_hz(10.0);
        let input = [[]; 512];
        let mut output = [0.0; 512];
        dsp.process_block(&input, slice_to_mono_block_mut(&mut output));

        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(3) })
    }
}
