//! # BJT saturators
//!
//! Provides saturators for BJT transistors in various configurations.
use crate::Saturator;
use numeric_literals::replace_float_literals;
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::math::smooth_clamp;
use valib_core::Scalar;

/// Memoryless nonlinearity for a BJT NPN Transistor in a common-collector configuration.
/// `xbias` and `ybias` are empirical values that can be fit to DC transfer data or to recenter the
/// signal before and after the saturation.
#[derive(Debug, Copy, Clone)]
pub struct CommonCollector<T> {
    /// Positive rail (positive input supply) voltage
    pub vcc: T,
    /// Negative rail (negative input supply) voltage
    pub vee: T,
    /// X-axis bias at the input
    pub xbias: T,
    /// Y-axis bias at the output.
    pub ybias: T,
}

impl<T: Scalar> Default for CommonCollector<T> {
    fn default() -> Self {
        Self {
            vcc: T::from_f64(4.5),
            vee: T::from_f64(-4.5),
            xbias: T::from_f64(0.77),
            ybias: T::from_f64(-0.77),
        }
    }
}

#[profiling::all_functions]
impl<T: Scalar> Saturator<T> for CommonCollector<T> {
    #[replace_float_literals(T::from_f64(literal))]
    fn saturate(&self, x: T) -> T {
        smooth_clamp(0.1, x + self.xbias, self.vee, self.vcc) + self.ybias
    }
}

impl<T: Scalar> DSPMeta for CommonCollector<T> {
    type Sample = T;
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<1, 1> for CommonCollector<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.saturate(x)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::BLUE;
    use valib_core::util::lerp;
    use valib_core::util::tests::{Plot, Series};

    #[test]
    fn test_common_collector() {
        const N: usize = 100;
        const STEP: f32 = 1. / N as f32;
        let sat = CommonCollector {
            vee: -4.5,
            vcc: 4.5,
            xbias: 0.770,
            ybias: -0.770,
        };
        let input: [f32; N] = std::array::from_fn(|i| lerp(i as f32 * STEP, -10., 10.));
        let output = input.map(|x| sat.saturate(x));
        Plot {
            title: "Common Collector saturator",
            bode: false,
            series: &[Series {
                label: "Output",
                samplerate: N as f32,
                color: &BLUE,
                series: &output,
            }],
        }
        .create_svg("plots/saturators/bjt/common_collector.svg");
        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(4)});
    }
}
