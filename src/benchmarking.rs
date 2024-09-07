use std::hint::black_box;
use num_traits::Zero;
use crate::dsp::DSPProcess;

pub fn benchmark_dsp<P: DSPProcess<I, O>, const I: usize, const O: usize>(amount: usize, mut dsp: P) {
    let frame = std::array::from_fn(|_| P::Sample::zero());
    for _ in 0..amount {
        black_box(black_box(&mut dsp).process(black_box(frame)));
    }
}