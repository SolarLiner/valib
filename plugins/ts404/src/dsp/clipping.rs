use crate::TARGET_SAMPLERATE;
use nih_plug::prelude::AtomicF32;
use nih_plug::util::db_to_gain_fast;
use num_traits::{Float, ToPrimitive};
use numeric_literals::replace_float_literals;
use std::sync::{atomic::Ordering, Arc};
use valib::math::smooth_clamp;
use valib::saturators::clippers::DiodeClipper;
use valib::saturators::{Saturator, Slew};
use valib::simd::SimdValue;
use valib::util::Rms;
use valib::wdf::dsl::*;
use valib::{
    dsp::{DSPMeta, DSPProcess},
    wdf::*,
    Scalar,
};

struct CrossoverDistortion<T> {
    // pub t: T,
    pub drift: T,
}

impl<T: Scalar> CrossoverDistortion<T> {
    pub fn new(drift: T) -> Self {
        Self {
            // t: T::from_f64(1e-1),
            drift,
        }
    }

    pub fn from_age(age: T) -> Self {
        Self::new(Self::drift_from_age(age))
    }

    pub fn set_age(&mut self, age: T) {
        self.drift = Self::drift_from_age(age);
    }

    #[replace_float_literals(T::from_f64(literal))]
    fn drift_from_age(age: T) -> T {
        2.96e-3 * age.simd_powf(1. / 3.)
    }
}

impl<T: Scalar> Saturator<T> for CrossoverDistortion<T> {
    fn saturate(&self, x: T) -> T {
        let l0 = x + self.drift;
        let l1 = x - self.drift;
        l0.simd_min(T::zero().simd_max(l1))
    }
}

type Stage1Module<T> = WdfModule<
    ResistiveVoltageSource<T>,
    Series<Capacitor<T>, Series<Resistor<T>, ResistiveVoltageSource<T>>>,
>;

struct Stage1<T: Scalar> {
    module: Stage1Module<T>,
    vin: Node<ResistiveVoltageSource<T>>,
    vout: Node<ResistiveVoltageSource<T>>,
}

impl<T: Scalar> Stage1<T> {
    pub fn new(samplerate: T) -> Self {
        let vin = rvsource(T::from_f64(4.5), T::one());
        let vout = rvsource(T::from_f64(10e3), T::from_f64(4.5));
        let module = module(
            vin.clone(),
            series(
                capacitor(samplerate, T::from_f64(1e-6)),
                series(resistor(T::from_f64(220.)), vout.clone()),
            ),
        );
        Self { module, vin, vout }
    }
}

impl<T: Scalar> DSPMeta for Stage1<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.module.set_samplerate(samplerate as _);
    }

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.module.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for Stage1<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        node_mut(&self.vin).vs = x[0];
        self.module.process_sample();
        let y = voltage(&self.vout);
        [y]
    }
}

type Stage2Module<T> = WdfModule<IdealVoltageSource<T>, Series<Capacitor<T>, Resistor<T>>>;

struct Stage2<T: Scalar> {
    module: Stage2Module<T>,
    vin: Node<IdealVoltageSource<T>>,
    iout: Node<Resistor<T>>,
}

impl<T: Scalar> Stage2<T> {
    pub fn new(samplerate: T) -> Self {
        let vin = ivsource(T::from_f64(0.));
        let iout = resistor(T::from_f64(4.7e3));
        let module = module(
            vin.clone(),
            series(capacitor(samplerate, T::from_f64(0.047e-6)), iout.clone()),
        );
        Self { module, vin, iout }
    }
}

impl<T: Scalar> DSPMeta for Stage2<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.module.set_samplerate(samplerate as _);
    }

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.module.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for Stage2<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        node_mut(&self.vin).vs = x[0];
        self.module.process_sample();
        let y = current(&self.iout);
        [y]
    }
}

type Stage3Module<T> = WdfModule<DiodeNR<T>, Parallel<ResistiveCurrentSource<T>, Capacitor<T>>>;

struct Stage3<T: Scalar<Element: Float>> {
    module: Stage3Module<T>,
    iin: Node<ResistiveCurrentSource<T>>,
}

impl<T: Scalar<Element: Float>> Stage3<T> {
    pub fn new(samplerate: T) -> Self {
        let iin = rcsource(T::from_f64(51e3), T::from_f64(0.));
        let module = module(
            node({
                let data = DiodeClipper::new_silicon(1, 1, T::zero());
                let mut wdf = DiodeNR::from_data(data);
                #[cfg(debug_assertions)]
                {
                    wdf.max_iter = 10;
                    wdf.max_tolerance = T::from_f64(1e-4);
                }
                #[cfg(not(debug_assertions))]
                {
                    wdf.max_iter = 100;
                    wdf.max_tolerance = T::from_f64(1e-6);
                }
                wdf
            }),
            parallel(iin.clone(), capacitor(samplerate, T::from_f64(51e-12))),
        );
        Self { module, iin }
    }

    pub fn set_dist(&mut self, amt: T) {
        node_mut(&self.iin).r = T::from_f64(51e3) + amt * T::from_f64(500e3);
    }
}

impl<T: Scalar<Element: Float>> DSPMeta for Stage3<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.module.set_samplerate(samplerate as _);
    }

    fn latency(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        self.module.reset();
    }
}

impl<T: Scalar<Element: Float>> DSPProcess<1, 1> for Stage3<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        node_mut(&self.iin).j = x[0];
        self.module.process_sample();
        let y = voltage(&self.module.root);
        [y]
    }
}

pub struct ClippingStage<T: Scalar<Element: Float>> {
    out_slew: Slew<T>,
    stage1: Stage1<T>,
    stage2: Stage2<T>,
    stage3: Stage3<T>,
    pub(crate) led_rms: Rms<f32>,
    pub(crate) led_display: Arc<AtomicF32>,
    samplerate: T,
    crossover: CrossoverDistortion<T>,
}

impl<T: Scalar<Element: Float>> ClippingStage<T> {
    pub fn new(samplerate: T) -> Self {
        let rms_samples = (16e-3 * TARGET_SAMPLERATE) as usize;
        Self {
            out_slew: Slew::new(
                samplerate,
                component_matching_slew_rate(samplerate, T::zero()),
            ),
            stage1: Stage1::new(samplerate),
            stage2: Stage2::new(samplerate),
            stage3: Stage3::new(samplerate),
            led_rms: Rms::new(rms_samples),
            led_display: Arc::new(AtomicF32::new(0.0)),
            crossover: CrossoverDistortion::new(T::zero()),
            samplerate,
        }
    }

    pub fn set_age(&mut self, age: T) {
        self.crossover.set_age(age);
        self.out_slew.max_diff = component_matching_slew_rate(self.samplerate, age);
    }

    pub fn set_dist(&mut self, amt: T) {
        self.stage3.set_dist(amt);
    }
}

impl<T: Scalar<Element: Float>> DSPMeta for ClippingStage<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.out_slew.set_samplerate(samplerate);
        self.stage1.set_samplerate(samplerate);
        self.stage2.set_samplerate(samplerate);
        self.stage3.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.out_slew.latency()
            + self.stage1.latency()
            + self.stage2.latency()
            + self.stage3.latency()
    }

    fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
        self.stage3.reset();
    }
}

impl<T: Scalar<Element: ToPrimitive + Float>> DSPProcess<1, 1> for ClippingStage<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        let [vplus] = self.stage1.process(x);
        let [vf] = self.stage3.process(self.stage2.process([vplus]));
        let led_value = (vf.simd_abs() - T::from_f64(0.7))
            .simd_max(T::from_f64(0.))
            .simd_horizontal_sum()
            .to_f32()
            .unwrap_or_default()
            * 0.5;
        let led_value = led_value.select(led_value.is_finite(), 0.);
        let led_value = self.led_rms.add_element(led_value);
        self.led_display.store(led_value, Ordering::Relaxed);
        let vout = self.crossover.saturate(vplus + vf - T::from_f64(4.5)) + T::from_f64(4.5);
        self.out_slew.process([smooth_clamp(
            T::from_f64(0.1),
            vout,
            T::zero(),
            T::from_f64(9.),
        )])
    }
}

fn component_matching_slew_rate<T: Scalar>(samplerate: T, age: T) -> T {
    let new = T::from_f64(db_to_gain_fast(80.0) as _);
    let rate = T::from_f64(0.05);
    new * (-age * rate).simd_exp() / samplerate
}

#[cfg(test)]
mod tests {
    use super::*;
    use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
    use valib::dsp::{BlockAdapter, DSPProcessBlock};
    use valib::util::lerp;

    #[test]
    fn crossover_dc_sweep() {
        const N: usize = 100;
        let crossover = CrossoverDistortion::new(1.);
        let output: [f32; N] = std::array::from_fn(|i| {
            let x = lerp(i as f32 / N as f32, -5., 5.);
            crossover.saturate(x)
        });
        insta::assert_csv_snapshot!(&output as &[_], { "[]" => insta::rounded_redaction(4) });
    }

    #[test]
    fn drive_test() {
        const SAMPLERATE: f64 = 1024.;
        const N: usize = 4096;
        let mut dsp = BlockAdapter(ClippingStage::new(SAMPLERATE));
        let input = (0..N)
            .map(|i| 100. * i as f64 / SAMPLERATE)
            .map(|x| 4.5 + x.sin() * 10.)
            .collect::<Vec<_>>();
        let mut output = vec![0.; N];
        dsp.process_block(
            AudioBufferRef::new([&input]).unwrap(),
            AudioBufferMut::new([&mut output]).unwrap(),
        );
        let assert: [_; N] = std::array::from_fn(|i| (input[i], output[i]));
        insta::assert_csv_snapshot!(&assert as &[_], { "[][]" => insta::rounded_redaction(4) });
    }
}
