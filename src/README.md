# Introduction

`valib` is a library focusing on abstracting DSP algorithms to make them
reusable and composable. It's focusing on musical applications such as plugins
or embedded digital synths.

It features many useful algorithms amongst which:

- Oscillators (BLIT, wavetable (with interpolation))
- Filters (Ladder, SVF, Biquads, all self-resonant with nonlinearities)
    - State-space models that can directly be constructed from their constituent
      matrix
    - Biquads support saturators
- Saturators (Clipper/Tanh/Asinh/Diode Clipper with configurable ADAA)
- Oversampling
- Integrations
    - nih-plug (includes ability to bind to nih parameters)
  - fundsp (any statically-defined graph)
