//! # Polyphase filter
//!
//! Port of <https://www.musicdsp.org/en/latest/Filters/39-polyphase-filters.html>.

use num_traits::Zero;

use crate::dsp::blocks::Series;
use crate::dsp::{analysis::DspAnalysis, DSPMeta, DSPProcess};
use crate::Scalar;

/// Specialized 2nd-order allpass filter.
#[derive(Debug, Copy, Clone)]
struct Allpass<T> {
    a: T,
    x: [T; 3],
    y: [T; 3],
}

impl<T: Scalar> DSPMeta for Allpass<T> {
    type Sample = T;
    fn latency(&self) -> usize {
        2
    }

    fn reset(&mut self) {
        for s in self.x.iter_mut().chain(self.y.iter_mut()) {
            s.set_zero();
        }
    }
}

#[profiling::all_functions]
impl<T: Scalar> DSPProcess<1, 1> for Allpass<T> {
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let [x0, x1, x2] = self.x;
        let [y0, y1, y2] = self.y;

        self.x = [x, x0, x1];
        let y = x2 + ((x - y2) * self.a);
        self.y = [y, y0, y1];

        [y]
    }
}

impl<T: Zero> Allpass<T> {
    pub fn new(a: T) -> Self {
        Self {
            a,
            x: std::array::from_fn(|_| T::zero()),
            y: std::array::from_fn(|_| T::zero()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HalfbandFilter<T, const ORDER: usize> {
    filter_a: Series<[Allpass<T>; ORDER]>,
    filter_b: Series<[Allpass<T>; ORDER]>,
    y0: T,
}

impl<T: Scalar, const ORDER: usize> DSPMeta for HalfbandFilter<T, ORDER> {
    type Sample = T;

    fn latency(&self) -> usize {
        self.filter_a.latency() + self.filter_b.latency()
    }
}

#[profiling::all_functions]
impl<T: Scalar, const ORDER: usize> DSPProcess<1, 1> for HalfbandFilter<T, ORDER> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let y = (self.filter_a.process(x)[0] + self.y0) * T::from_f64(0.5);
        self.y0 = self.filter_b.process(x)[0];
        [y]
    }
}

impl<T: Scalar, const ORDER: usize> HalfbandFilter<T, ORDER> {
    fn from_coeffs(k_a: [T; ORDER], k_b: [T; ORDER]) -> Self {
        Self {
            filter_a: Series(std::array::from_fn(|i| Allpass::new(k_a[i]))),
            filter_b: Series(std::array::from_fn(|i| Allpass::new(k_b[i]))),
            y0: T::zero(),
        }
    }
}

#[rustfmt::skip]
pub fn steep_order12<T: Scalar>() -> HalfbandFilter<T, 6> {
    HalfbandFilter::from_coeffs(
        [ 0.036681502163648017
        , 0.2746317593794541
        , 0.5610989697879195
        , 0.769741833862266
        , 0.8922608180038789
        , 0.962094548378084
        ].map(T::from_f64),
        [ 0.13654762463195771
        , 0.42313861743656667
        , 0.6775400499741616
        , 0.839889624849638
        , 0.9315419599631839
        , 0.9878163707328971
        ].map(T::from_f64),
    )
}

#[rustfmt::skip]
pub fn steep_order10<T: Scalar>() -> HalfbandFilter<T, 5> {
    HalfbandFilter::from_coeffs(
        [ 0.051457617441190984
		, 0.35978656070567017
		, 0.6725475931034693
		, 0.8590884928249939
		, 0.9540209867860787
		].map(T::from_f64),
        [ 0.18621906251989334
		, 0.529951372847964
		, 0.7810257527489514
		, 0.9141815687605308
		, 0.985475023014907
        ].map(T::from_f64),
    )
}
