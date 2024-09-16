use crate::Phasor;
use num_traits::{one, zero, ConstOne, ConstZero};
use std::marker::PhantomData;
use valib_core::dsp::blocks::P1;
use valib_core::dsp::{DSPMeta, DSPProcess};
use valib_core::simd::SimdBool;
use valib_core::util::lerp;
use valib_core::Scalar;

pub struct PolyBLEP<T> {
    pub amplitude: T,
    pub phase: T,
}

impl<T: Scalar> PolyBLEP<T> {
    pub fn eval(&self, dt: T, phase: T) -> T {
        let t = T::simd_fract(phase + self.phase);
        let ret = t.simd_lt(dt).if_else(
            || {
                let t = t / dt;
                t + t - t * t - one()
            },
            || {
                t.simd_gt(one::<T>() - dt).if_else(
                    || {
                        let t = (t - one()) / dt;
                        t * t + t + t + one()
                    },
                    || zero(),
                )
            },
        );
        self.amplitude * ret
    }
}

pub trait PolyBLEPOscillator: DSPMeta {
    fn bleps(&self) -> impl IntoIterator<Item = PolyBLEP<Self::Sample>>;
    fn naive_eval(&mut self, phase: Self::Sample) -> Self::Sample;
}

pub struct PolyBLEPDriver<Osc: PolyBLEPOscillator> {
    pub phasor: Phasor<Osc::Sample>,
    pub blep: Osc,
    samplerate: Osc::Sample,
}

impl<Osc: PolyBLEPOscillator> PolyBLEPDriver<Osc> {
    pub fn new(samplerate: Osc::Sample, frequency: Osc::Sample, blep: Osc) -> Self {
        Self {
            phasor: Phasor::new(samplerate, frequency),
            blep,
            samplerate,
        }
    }

    pub fn set_frequency(&mut self, frequency: Osc::Sample) {
        self.phasor.set_frequency(frequency);
    }
}

impl<Osc: PolyBLEPOscillator> DSPProcess<0, 1> for PolyBLEPDriver<Osc> {
    fn process(&mut self, _: [Self::Sample; 0]) -> [Self::Sample; 1] {
        let [phase] = self.phasor.process([]);
        let mut y = self.blep.naive_eval(phase);
        for blep in self.blep.bleps() {
            y += blep.eval(self.phasor.step, phase);
        }
        [y]
    }
}

impl<Osc: PolyBLEPOscillator> DSPMeta for PolyBLEPDriver<Osc> {
    type Sample = Osc::Sample;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.phasor.set_samplerate(samplerate);
        self.blep.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.blep.latency()
    }

    fn reset(&mut self) {
        self.phasor.reset();
        self.blep.reset();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SawBLEP<T>(PhantomData<T>);

impl<T> Default for SawBLEP<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar> DSPMeta for SawBLEP<T> {
    type Sample = T;
}

impl<T: ConstZero + ConstOne + Scalar> PolyBLEPOscillator for SawBLEP<T> {
    fn bleps(&self) -> impl IntoIterator<Item = PolyBLEP<Self::Sample>> {
        [PolyBLEP {
            amplitude: -T::ONE,
            phase: T::ZERO,
        }]
    }

    fn naive_eval(&mut self, phase: Self::Sample) -> Self::Sample {
        T::from_f64(2.0) * phase - T::one()
    }
}

pub type Sawtooth<T> = PolyBLEPDriver<SawBLEP<T>>;

#[derive(Debug, Copy, Clone)]
pub struct SquareBLEP<T> {
    pw: T,
}

impl<T: Scalar> SquareBLEP<T> {
    pub fn new(pulse_width: T) -> Self {
        Self {
            pw: pulse_width.simd_clamp(zero(), one()),
        }
    }
}

impl<T: Scalar> SquareBLEP<T> {
    pub fn set_pulse_width(&mut self, pw: T) {
        self.pw = pw.simd_clamp(zero(), one());
    }
}

impl<T: Scalar> DSPMeta for SquareBLEP<T> {
    type Sample = T;
}

impl<T: ConstZero + ConstOne + Scalar> PolyBLEPOscillator for SquareBLEP<T> {
    fn bleps(&self) -> impl IntoIterator<Item = PolyBLEP<Self::Sample>> {
        [
            PolyBLEP {
                amplitude: T::ONE,
                phase: T::ZERO,
            },
            PolyBLEP {
                amplitude: -T::ONE,
                phase: T::one() - self.pw,
            },
        ]
    }

    fn naive_eval(&mut self, phase: Self::Sample) -> Self::Sample {
        let dc_offset = lerp(self.pw, -T::ONE, T::ONE);
        phase.simd_gt(self.pw).if_else(T::one, || -T::one()) + dc_offset
    }
}

pub type Square<T> = PolyBLEPDriver<SquareBLEP<T>>;

pub struct Triangle<T: ConstZero + ConstOne + Scalar> {
    square: Square<T>,
    integrator: P1<T>,
}

impl<T: ConstZero + ConstOne + Scalar> DSPMeta for Triangle<T> {
    type Sample = T;
    fn set_samplerate(&mut self, samplerate: f32) {
        self.square.set_samplerate(samplerate);
        self.integrator.set_samplerate(samplerate);
    }
    fn reset(&mut self) {
        self.square.reset();
        self.integrator.reset();
    }
}

impl<T: ConstZero + ConstOne + Scalar> DSPProcess<0, 1> for Triangle<T> {
    fn process(&mut self, []: [Self::Sample; 0]) -> [Self::Sample; 1] {
        self.integrator.process(self.square.process([]))
    }
}

impl<T: ConstZero + ConstOne + Scalar> Triangle<T> {
    pub fn new(samplerate: T, frequency: T, phase: T) -> Self {
        let mut square =
            PolyBLEPDriver::new(samplerate, frequency, SquareBLEP::new(T::from_f64(0.5)));
        square.phasor.phase = phase;
        let integrator = P1::new(samplerate, frequency);
        Self { square, integrator }
    }

    pub fn set_frequency(&mut self, frequency: T) {
        self.square.set_frequency(frequency);
        self.integrator.set_fc(frequency);
    }
}
