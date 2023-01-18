# SVF Mixer

An implementation of an analog State Variable Filter, with a mixer mix each output into the main plugin output.

SVF Mixer is provided for demonstration purposes, and is not a fully-featured plugin. It has no custom GUI.

## Download

You can download the latest nightly build from the Release page of the GitHub project, or find a permalink to the latest
nightly builds [here].

## Tips

- Turn on both the low- and high-pass outputs to create a notch filter.
- On top of a notch filter, you can then add the band-pass output in reverse (turning the band-pass gain left) to create
  an allpass filter.
- Add some bandpass to your lowpass/highpass to emphasize the cutoff frequency.
