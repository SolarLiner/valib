# Dirty Biquad

Adding nonlinearities to the classical Biquad algorithm. Implemented from [Jatin Chowdhury's *Nonlinear Biquad Filters*](https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-4-nonlinear-biquad-filters-ae6b3f23cb0e).

Unstable, resonating, gritty low-pass, band-pass and high-pass filters.

Dirty Biquad is provided for demonstration purposes, and is not a fully-featured plugin. It has no custom GUI.

## Download

You can download the latest nightly build from the Release page of the GitHub project, or find a permalink to the latest
nightly builds [here].

## Tips

- Try the various nonlinearities. The "Linear" one is no clipping at all, and "Clipped" is hard clipping, which both
  sound similar unless driven.
- The "nonlinear biquad" model does not attempt to counteract the drift of the cutoff introduced by the saturation, and
  so louder sounds will "open" the cutoff of the filter by themselves.
