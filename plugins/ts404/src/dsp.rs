use numeric_literals::replace_float_literals;
use valib::dsp::DSP;
use valib::filters::statespace::StateSpace;
use valib::saturators::Slew;
use valib::Scalar;

#[derive(Debug, Copy, Clone)]
pub struct Bypass<T> {
    pub inner: T,
    pub active: bool,
}

impl<T, const N: usize> DSP<N, N> for Bypass<T>
where
    T: DSP<N, N>,
{
    type Sample = T::Sample;

    fn process(&mut self, x: [Self::Sample; N]) -> [Self::Sample; N] {
        if self.active {
            self.inner.process(x)
        } else {
            x
        }
    }

    fn latency(&self) -> usize {
        if self.active {
            self.inner.latency()
        } else {
            0
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

impl<T> Bypass<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            active: true,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct InputStage<T: Scalar> {
    pub gain: T,
    state_space: StateSpace<T, 1, 1, 1>,
}

impl<T: Scalar> DSP<1, 1> for InputStage<T> {
    type Sample = T;

    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.state_space.process([x * self.gain])
    }

    fn latency(&self) -> usize {
        self.state_space.latency()
    }

    fn reset(&mut self) {
        self.state_space.reset()
    }
}

impl<T: Scalar> InputStage<T> {
    pub fn new(samplerate: T, gain: T) -> Self {
        Self {
            gain,
            state_space: crate::gen::input(samplerate.simd_recip()),
        }
    }

    pub fn set_samplerate(&mut self, samplerate: T) {
        self.state_space
            .update_matrices(&crate::gen::input(samplerate.simd_recip()));
    }
}

fn crossover_half<T: Scalar>(x: T, a: T, b: T) -> T {
    T::simd_ln(T::simd_exp(b * x) + T::from_f64(10.0).simd_powf(a)) / b
}

fn crossover<T: Scalar>(x: T, a: T, b: T) -> T {
    crossover_half(x, a, b) - crossover_half(-x, a, b)
}

#[derive(Debug, Copy, Clone)]
pub struct ClipperStage<T: Scalar> {
    state_space: StateSpace<T, 1, 3, 1>,
    pub crossover: (T, T),
    pub(crate) slew: Slew<T>,
}

impl<T: Scalar> DSP<1, 1> for ClipperStage<T> {
    type Sample = T;

    #[replace_float_literals(Self::Sample::from_f64(literal))]
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let [y] = self.state_space.process(x);
        let y = y.simd_asinh().simd_clamp(-4.5, 4.5);
        let [y] = self.slew.process([y]);
        let y = crossover(y, self.crossover.0, self.crossover.1);
        [y]
    }

    fn latency(&self) -> usize {
        self.state_space.latency() + self.slew.latency()
    }

    fn reset(&mut self) {
        self.state_space.reset();
        self.slew.reset();
    }
}

impl<T: Scalar> ClipperStage<T> {
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: T, dist: T) -> Self {
        let dt = samplerate.simd_recip();
        Self {
            state_space: crate::gen::clipper(dt, dist),
            crossover: (0.0, 30.0),
            slew: Slew::new(1e4 * dt),
        }
    }

    pub fn set_params(&mut self, samplerate: T, dist: T) {
        let dt = samplerate.simd_recip();
        self.state_space
            .update_matrices(&crate::gen::clipper(dt, dist));
        self.slew.set_max_diff(T::from_f64(1e5), samplerate);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ToneStage<T: Scalar>(StateSpace<T, 1, 4, 1>);

impl<T: Scalar> DSP<1, 1> for ToneStage<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.0.process(x)
    }

    fn latency(&self) -> usize {
        self.0.latency()
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

impl<T: Scalar> ToneStage<T> {
    #[replace_float_literals(T::from_f64(literal))]
    pub fn new(samplerate: T, tone: T) -> Self {
        Self(crate::gen::tone(samplerate.simd_recip(), tone))
    }

    #[replace_float_literals(T::from_f64(literal))]
    pub fn update_params(&mut self, samplerate: T, tone: T) {
        self.0
            .update_matrices(&crate::gen::tone(samplerate.simd_recip(), tone));
    }
}

#[derive(Debug, Copy, Clone)]
pub struct OutputStage<T: Scalar> {
    pub inner: Bypass<StateSpace<T, 1, 2, 1>>,
    pub gain: T,
}

impl<T: Scalar> OutputStage<T> {
    pub fn new(samplerate: T, gain: T) -> Self {
        Self {
            inner: Bypass::new(crate::gen::output(samplerate.simd_recip())),
            gain,
        }
    }
    pub fn set_samplerate(&mut self, samplerate: T) {
        self.inner
            .inner
            .update_matrices(&crate::gen::output(samplerate.simd_recip()));
    }
}

impl<T: Scalar> DSP<1, 1> for OutputStage<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let [y] = self.inner.process(x);
        [y * self.gain]
    }

    fn latency(&self) -> usize {
        DSP::latency(&self.inner)
    }

    fn reset(&mut self) {
        DSP::reset(&mut self.inner)
    }
}
