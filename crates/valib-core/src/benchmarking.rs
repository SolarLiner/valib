//! Module for benchmarking utilies.

use crate::dsp::DSPProcess;
use num_traits::Zero;
use std::hint::black_box;

/// Benchmark a DSP process by sending it a zeroed stream `amount` times. This automatically
/// blackboxes the process and input to ensure benchmarking provides relevant results.
///
/// # Arguments
///
/// * `amount`: Number of samples to process
/// * `dsp`: DSP process to benchmark
///
/// returns: ()
#[inline]
pub fn benchmark_dsp<P: DSPProcess<I, O>, const I: usize, const O: usize>(
    amount: usize,
    mut dsp: P,
) {
    let frame = std::array::from_fn(|_| P::Sample::zero());
    for _ in 0..amount {
        black_box(black_box(&mut dsp).process(black_box(frame)));
    }
}
