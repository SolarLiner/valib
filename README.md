# valib

![GitHub Build Status](https://img.shields.io/github/actions/workflow/status/solarliner/valib/build.yml)
![GitHub Tests Status](https://img.shields.io/github/actions/workflow/status/solarliner/valib/test.yml?label=tests)
![GitHub License](https://img.shields.io/github/license/solarliner/valib)

Plugins: [Abrasive](plugins/abrasive)  
Examples: [Diode Clipper](examples/diodeclipper) | [SVF Mixer](examples/svfmixer) | [Dirty Biquad](examples/dirty-biquad)

Library of musical filters and reusable blocks for musical DSP.

## Features

- Generic over scalars (transparent SIMD support planned)
- Generic over saturators
- Oversampling filter
- Biquad filter
- State Variable Filter
- Ladder filter

## Download

You can download the plugins and examples from the permalink [here](https://nightly.link/SolarLiner/valib/workflows/build/master).

## Licensing

To comply with `nih-plug` licensing requirements (and it stemming from VST3's requirements), the project is dual-licensed.

The code for the library (that is, everything under the `src` directory) is available under the `MIT` license.  
Plugins and examples are available under the `GPLv3` license.