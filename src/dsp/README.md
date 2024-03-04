Base traits for DSP algorithms.

Includes 3 main traits that algorithms can implement:

- the [`DSP`] trait defines the main methods to implement for per-sample algorithms.
  Most algorithms implements this version.
- the [`DSPBlock`] trait defines the methods to implement for per-block processing.
  Also includes helpers for using block processing in a per-sample context.
  [`DSP`] implementers get an impl for this for free. This means that
  algorhithms that would benefit from a specialized per-block processing impl
  cannot currently do it. There is a fix coming soon.
- the [`DspAnalysis`] trait defines the methods to provide frequency and phase
  response analysis.