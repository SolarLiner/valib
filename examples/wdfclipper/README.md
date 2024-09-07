# Diode Clipper

Instantly make any sound harsher by including this classical bit of analog signal processing onto your sound.

Diode Clipper is an emulation from first principles of a clipping circuit, as often seen in overdrive/screamer pedals.  
Note that this plugin only emulates the clipping, and does not provide any tonal control. Use filters before and/or after
this to tame the harshness.

Diode Clipper is provided for demonstration purposes, and is not a fully-featured plugin. It has no custom GUI.

## Download

You can download the latest nightly build from a permalink to the latest nightly builds
[here](https://nightly.link/SolarLiner/valib/workflows/build/master).

## Tips

- The "Use Model" is given for demonstration purposes (eg. A/B testing the two methods of solving the circuit equations)
  and should otherwise be kept as "On". The implicit solver (in use when set to "Off") does not work properly, especially
  at high input gains or drive, and will mess up your DAW's audio engine.
- Perhaps counterintuitively, the number of diodes in each direction "relaxes" the saturation amount, allowing a louder
  signal to pass through before clipping.
- The Drive knob is gain-compensated, but perceptually higher gain sounds quieter because of it.
