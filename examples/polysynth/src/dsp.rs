use crate::params::{FilterParams, FilterType, OscShape, PolysynthParams};
use crate::{SynthSample, MAX_BUFFER_SIZE, NUM_VOICES, OVERSAMPLE};
use fastrand::Rng;
use fastrand_contrib::RngExt;
use nih_plug::nih_log;
use nih_plug::util::db_to_gain_fast;
use num_traits::{ConstOne, ConstZero};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, SampleAdapter};
use valib::filters::biquad::Biquad;
use valib::filters::ladder::{Ladder, Transistor, OTA};
use valib::filters::specialized::DcBlocker;
use valib::filters::svf::Svf;
use valib::math::interpolation::{sine_interpolation, Interpolate, Sine};
use valib::oscillators::polyblep::{SawBLEP, Sawtooth, Square, SquareBLEP, Triangle};
use valib::oscillators::Phasor;
use valib::saturators::{bjt, Asinh, Clipper, Saturator, Tanh};
use valib::simd::{SimdBool, SimdValue};
use valib::util::{ratio_to_semitone, semitone_to_ratio};
use valib::voice::polyphonic::Polyphonic;
use valib::voice::upsample::UpsampledVoice;
use valib::voice::{NoteData, Voice};
use valib::Scalar;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
enum AdsrState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl AdsrState {
    pub fn next_state(self, gate: bool) -> Self {
        if gate {
            Self::Attack
        } else if !matches!(self, Self::Idle) {
            Self::Release
        } else {
            self
        }
    }
}

struct Adsr {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    samplerate: f32,
    attack_base: f32,
    decay_base: f32,
    release_base: f32,
    attack_rate: f32,
    decay_rate: f32,
    release_rate: f32,
    cur_state: AdsrState,
    cur_value: f32,
    release_coeff: f32,
    decay_coeff: f32,
    attack_coeff: f32,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            samplerate: 0.,
            attack: 0.,
            decay: 0.,
            sustain: 0.,
            release: 0.,
            attack_base: 1. + Self::TARGET_RATIO_ATTACK,
            decay_base: -Self::TARGET_RATIO_RELEASE,
            release_base: -Self::TARGET_RATIO_RELEASE,
            attack_coeff: 0.,
            decay_coeff: 0.,
            release_coeff: 0.,
            attack_rate: 0.,
            decay_rate: 0.,
            release_rate: 0.,
            cur_state: AdsrState::Idle,
            cur_value: 0.,
        }
    }
}

impl Adsr {
    const TARGET_RATIO_ATTACK: f32 = 0.3;
    const TARGET_RATIO_DECAY: f32 = 0.1;
    const TARGET_RATIO_RELEASE: f32 = 1e-3;
    pub fn new(
        samplerate: f32,
        attack: f32,
        decay: f32,
        sustain: f32,
        release: f32,
        gate: bool,
    ) -> Self {
        let mut this = Self {
            samplerate,
            cur_state: AdsrState::Idle.next_state(gate),
            ..Self::default()
        };
        this.set_attack(attack);
        this.set_decay(decay);
        this.set_sustain(sustain);
        this.set_release(release);
        this.cur_state = this.cur_state.next_state(gate);
        this
    }

    pub fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = samplerate;
        self.set_attack(self.attack);
        self.set_decay(self.decay);
        self.set_release(self.release);
    }

    pub fn set_attack(&mut self, attack: f32) {
        if (self.attack - attack).abs() < 1e-6 {
            return;
        }
        self.attack = attack;
        self.attack_rate = self.samplerate * attack;
        self.attack_coeff = Self::calc_coeff(self.attack_rate, Self::TARGET_RATIO_ATTACK);
        self.attack_base = (1. + Self::TARGET_RATIO_ATTACK) * (1.0 - self.attack_coeff);
    }

    pub fn set_decay(&mut self, decay: f32) {
        if (self.decay - decay).abs() < 1e-6 {
            return;
        }
        self.decay = decay;
        self.decay_rate = self.samplerate * decay;
        self.decay_coeff = Self::calc_coeff(self.decay_rate, Self::TARGET_RATIO_DECAY);
        self.decay_base = (self.sustain - Self::TARGET_RATIO_DECAY) * (1. - self.decay_coeff);
    }

    pub fn set_sustain(&mut self, sustain: f32) {
        self.sustain = sustain;
    }

    pub fn set_release(&mut self, release: f32) {
        if (self.release - release).abs() < 1e-6 {
            return;
        }
        self.release = release;
        self.release_rate = self.samplerate * release;
        self.release_coeff = Self::calc_coeff(self.release_rate, Self::TARGET_RATIO_RELEASE);
        self.release_base = -Self::TARGET_RATIO_RELEASE * (1. - self.release_coeff);
    }

    pub fn gate(&mut self, gate: bool) {
        self.cur_state = self.cur_state.next_state(gate);
    }

    pub fn next_sample(&mut self) -> f32 {
        match self.cur_state {
            AdsrState::Attack => {
                self.cur_value = self.attack_base + self.cur_value * self.attack_coeff;
                if self.cur_value >= 1. {
                    self.cur_value = 1.;
                    self.cur_state = AdsrState::Decay;
                }
            }
            AdsrState::Decay => {
                self.cur_value = self.decay_base + self.cur_value * self.decay_coeff;
                if self.cur_value <= self.sustain {
                    self.cur_value = self.sustain;
                    self.cur_state = AdsrState::Sustain;
                }
            }
            AdsrState::Release => {
                self.cur_value = self.release_base + self.cur_value * self.release_coeff;
                if self.cur_value <= 0. {
                    self.cur_value = 0.;
                    self.cur_state = AdsrState::Idle;
                }
            }
            AdsrState::Sustain | AdsrState::Idle => {}
        }
        self.cur_value
    }

    pub fn state(&self) -> AdsrState {
        self.cur_state
    }

    pub fn current_value(&self) -> f32 {
        self.cur_value
    }

    pub fn current_value_as<T: Scalar>(&self) -> T {
        T::from_f64(self.current_value() as _)
    }

    pub fn is_idle(&self) -> bool {
        matches!(self.cur_state, AdsrState::Idle)
    }

    pub fn reset(&mut self) {
        self.cur_state = AdsrState::Idle;
        self.cur_value = 0.;
    }

    fn calc_coeff(rate: f32, ratio: f32) -> f32 {
        if rate <= 0. {
            0.
        } else {
            (-((1.0 + ratio) / ratio).ln() / rate).exp()
        }
    }
}

struct Drift<T> {
    rng: Rng,
    phasor: Phasor<T>,
    last_value: T,
    next_value: T,
    interp: Sine<T>,
}

impl<T: Scalar> Drift<T> {
    pub fn new(mut rng: Rng, samplerate: T, frequency: T) -> Self {
        let phasor = Phasor::new(samplerate, frequency);
        let last_value = T::from_f64(rng.f64_range(-1.0..1.0));
        let next_value = T::from_f64(rng.f64_range(-1.0..1.0));
        Self {
            rng,
            phasor,
            last_value,
            next_value,
            interp: sine_interpolation(),
        }
    }

    pub fn next_sample(&mut self) -> T {
        let reset_mask = self.phasor.next_sample_resets();
        if reset_mask.any() {
            self.last_value = reset_mask.if_else(|| self.next_value, || self.last_value);
            self.next_value = reset_mask.if_else(
                || T::from_f64(self.rng.f64_range(-1.0..1.0)),
                || self.next_value,
            );
        }

        let [t] = self.phasor.process([]);
        self.interp
            .interpolate(t, [self.last_value, self.next_value])
    }
}

pub enum PolyOsc<T: ConstZero + ConstOne + Scalar> {
    Sine(Phasor<T>),
    Triangle(Triangle<T>),
    Square(Square<T>),
    Sawtooth(Sawtooth<T>),
}

impl<T: ConstZero + ConstOne + Scalar> PolyOsc<T> {
    fn new(
        samplerate: T,
        shape: OscShape,
        note_data: NoteData<T>,
        pulse_width: T,
        phase: T,
    ) -> Self {
        match shape {
            OscShape::Sine => {
                Self::Sine(Phasor::new(samplerate, note_data.frequency).with_phase(phase))
            }
            OscShape::Triangle => {
                Self::Triangle(Triangle::new(samplerate, note_data.frequency, phase))
            }
            OscShape::Square => {
                let mut square = Square::new(
                    samplerate,
                    note_data.frequency,
                    SquareBLEP::new(pulse_width),
                );
                square.phasor.set_phase(phase);
                Self::Square(square)
            }
            OscShape::Saw => {
                let mut sawtooth =
                    Sawtooth::new(samplerate, note_data.frequency, SawBLEP::default());
                sawtooth.phasor.set_phase(phase);
                Self::Sawtooth(sawtooth)
            }
        }
    }

    fn is_osc_shape(&self, osc_shape: OscShape) -> bool {
        match self {
            Self::Sine(..) if matches!(osc_shape, OscShape::Sine) => true,
            Self::Triangle(..) if matches!(osc_shape, OscShape::Triangle) => true,
            Self::Square(..) if matches!(osc_shape, OscShape::Square) => true,
            Self::Sawtooth(..) if matches!(osc_shape, OscShape::Saw) => true,
            _ => false,
        }
    }

    pub fn set_pulse_width(&mut self, pw: T) {
        if let Self::Square(sq) = self {
            sq.blep.set_pulse_width(pw)
        }
    }
}

impl<T: ConstZero + ConstOne + Scalar> DSPMeta for PolyOsc<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        match self {
            Self::Sine(p) => p.set_samplerate(samplerate),
            Self::Triangle(tri) => tri.set_samplerate(samplerate),
            Self::Square(sq) => sq.set_samplerate(samplerate),
            Self::Sawtooth(sw) => sw.set_samplerate(samplerate),
        }
    }

    fn reset(&mut self) {
        match self {
            PolyOsc::Sine(p) => p.reset(),
            PolyOsc::Triangle(tri) => tri.reset(),
            PolyOsc::Square(sqr) => sqr.reset(),
            PolyOsc::Sawtooth(saw) => saw.reset(),
        }
    }
}

impl<T: ConstZero + ConstOne + Scalar> DSPProcess<1, 1> for PolyOsc<T> {
    fn process(&mut self, [freq]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        match self {
            Self::Sine(p) => {
                p.set_frequency(freq);
                p.process([]).map(|x| (T::simd_two_pi() * x).simd_sin())
            }
            Self::Triangle(tri) => {
                tri.set_frequency(freq);
                tri.process([])
            }
            Self::Square(sq) => {
                sq.set_frequency(freq);
                sq.process([])
            }
            Self::Sawtooth(sw) => {
                sw.set_frequency(freq);
                sw.process([])
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Noise {
    rng: Rng,
}

impl Noise {
    pub fn from_rng(rng: Rng) -> Self {
        Self { rng }
    }

    pub fn next_value_f32<T: Scalar<Element = f32>>(&mut self) -> T
    where
        [(); <T as SimdValue>::LANES]:,
    {
        T::from_values(std::array::from_fn(|_| self.rng.f32_range(-1.0..1.0)))
    }

    pub fn next_value_f64<T: Scalar<Element = f64>>(&mut self) -> T
    where
        [(); <T as SimdValue>::LANES]:,
    {
        T::from_values(std::array::from_fn(|_| self.rng.f64_range(-1.0..1.0)))
    }
}

#[derive(Debug, Default, Copy, Clone)]
struct Sinh;

impl<T: Scalar> Saturator<T> for Sinh {
    fn saturate(&self, x: T) -> T {
        x.simd_sinh()
    }
}

#[derive(Debug, Copy, Clone)]
enum FilterImpl<T> {
    Transistor(Ladder<T, Transistor<Tanh>>),
    Ota(Ladder<T, OTA<Tanh>>),
    Svf(Svf<T, Sinh>),
    Biquad(Biquad<T, Clipper<T>>),
}

impl<T: Scalar> FilterImpl<T> {
    fn from_type(samplerate: T, ftype: FilterType, cutoff: T, resonance: T) -> FilterImpl<T> {
        match ftype {
            FilterType::TransistorLadder => {
                let mut ladder = Ladder::new(samplerate, cutoff, T::from_f64(4.) * resonance);
                ladder.compensated = true;
                Self::Transistor(ladder)
            }
            FilterType::OTALadder => {
                let mut ladder = Ladder::new(samplerate, cutoff, T::from_f64(4.) * resonance);
                ladder.compensated = true;
                Self::Ota(ladder)
            }
            FilterType::Svf => Self::Svf(Svf::new(samplerate, cutoff, T::one() - resonance)),
            FilterType::Digital => Self::Biquad(
                Biquad::lowpass(
                    cutoff / samplerate,
                    (T::from_f64(3.) * resonance).simd_exp(),
                )
                .with_saturators(Default::default(), Default::default()),
            ),
        }
    }
}

impl<T: Scalar> FilterImpl<T> {
    fn set_params(&mut self, samplerate: T, cutoff: T, resonance: T) {
        match self {
            Self::Transistor(p) => {
                p.set_cutoff(cutoff);
                p.set_resonance(T::from_f64(4.) * resonance);
            }
            Self::Ota(p) => {
                p.set_cutoff(cutoff);
                p.set_resonance(T::from_f64(4.) * resonance);
            }
            Self::Svf(p) => {
                p.set_cutoff(cutoff);
                p.set_r(T::one() - resonance);
            }
            Self::Biquad(p) => {
                p.update_coefficients(&Biquad::lowpass(
                    cutoff / samplerate,
                    (T::from_f64(3.) * resonance).simd_exp(),
                ));
            }
        }
    }
}

impl<T: Scalar> DSPMeta for FilterImpl<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        match self {
            FilterImpl::Transistor(p) => p.set_samplerate(samplerate),
            FilterImpl::Ota(p) => p.set_samplerate(samplerate),
            FilterImpl::Svf(p) => p.set_samplerate(samplerate),
            FilterImpl::Biquad(p) => p.set_samplerate(samplerate),
        }
    }

    fn latency(&self) -> usize {
        match self {
            FilterImpl::Transistor(p) => p.latency(),
            FilterImpl::Ota(p) => p.latency(),
            FilterImpl::Svf(p) => p.latency(),
            FilterImpl::Biquad(p) => p.latency(),
        }
    }

    fn reset(&mut self) {
        match self {
            FilterImpl::Transistor(p) => p.reset(),
            FilterImpl::Ota(p) => p.reset(),
            FilterImpl::Svf(p) => p.reset(),
            FilterImpl::Biquad(p) => p.reset(),
        }
    }
}

impl<T: Scalar> DSPProcess<1, 1> for FilterImpl<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        match self {
            FilterImpl::Transistor(p) => p.process(x),
            FilterImpl::Ota(p) => p.process(x),
            FilterImpl::Svf(p) => [p.process(x)[0]],
            FilterImpl::Biquad(p) => p.process(x),
        }
    }
}

#[derive(Debug, Clone)]
struct Filter<T> {
    fimpl: FilterImpl<T>,
    params: Arc<FilterParams>,
    samplerate: T,
}

impl<T: Scalar> Filter<T> {
    fn new(samplerate: T, params: Arc<FilterParams>) -> Filter<T> {
        let cutoff = T::from_f64(params.cutoff.value() as _);
        let resonance = T::from_f64(params.resonance.value() as _);
        Self {
            fimpl: FilterImpl::from_type(samplerate, params.filter_type.value(), cutoff, resonance),
            params,
            samplerate,
        }
    }
}

impl<T: Scalar> Filter<T> {
    fn update_filter(&mut self, modulation_st: T) {
        let cutoff =
            semitone_to_ratio(modulation_st) * T::from_f64(self.params.cutoff.smoothed.next() as _);
        let resonance = T::from_f64(self.params.resonance.smoothed.next() as _);
        self.fimpl = match self.params.filter_type.value() {
            FilterType::TransistorLadder if !matches!(self.fimpl, FilterImpl::Transistor(..)) => {
                let mut ladder = Ladder::new(self.samplerate, cutoff, T::from_f64(4.) * resonance);
                ladder.compensated = true;
                FilterImpl::Transistor(ladder)
            }
            FilterType::OTALadder if !matches!(self.fimpl, FilterImpl::Ota(..)) => {
                let mut ladder = Ladder::new(self.samplerate, cutoff, T::from_f64(4.) * resonance);
                ladder.compensated = true;
                FilterImpl::Ota(ladder)
            }
            FilterType::Svf if !matches!(self.fimpl, FilterImpl::Svf(..)) => {
                FilterImpl::Svf(Svf::new(self.samplerate, cutoff, T::one() - resonance))
            }
            FilterType::Digital if !matches!(self.fimpl, FilterImpl::Biquad(..)) => {
                FilterImpl::Biquad(
                    Biquad::lowpass(cutoff / self.samplerate, resonance)
                        .with_saturators(Default::default(), Default::default()),
                )
            }
            _ => {
                self.fimpl.set_params(self.samplerate, cutoff, resonance);
                return;
            }
        };
    }
}

impl<T: Scalar> DSPMeta for Filter<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = T::from_f64(samplerate as _);
    }

    fn latency(&self) -> usize {
        self.fimpl.latency()
    }

    fn reset(&mut self) {
        self.fimpl.reset();
    }
}

impl<T: Scalar> DSPProcess<2, 1> for Filter<T> {
    fn process(&mut self, [x, mod_st]: [Self::Sample; 2]) -> [Self::Sample; 1] {
        self.update_filter(mod_st);
        self.fimpl.process([x])
    }
}

pub(crate) const NUM_OSCILLATORS: usize = 2;

pub struct RawVoice<T: ConstZero + ConstOne + Scalar> {
    osc: [PolyOsc<T>; NUM_OSCILLATORS],
    osc_out_sat: bjt::CommonCollector<T>,
    noise: Noise,
    filter: Filter<T>,
    params: Arc<PolysynthParams>,
    vca_env: Adsr,
    vcf_env: Adsr,
    note_data: NoteData<T>,
    drift: [Drift<f32>; NUM_OSCILLATORS],
    samplerate: T,
    rng: Rng,
}

impl<T: ConstZero + ConstOne + Scalar> RawVoice<T> {
    fn new(
        target_samplerate_f64: f64,
        params: Arc<PolysynthParams>,
        note_data: NoteData<T>,
    ) -> Self {
        static VOICE_SEED: AtomicU64 = AtomicU64::new(0);
        let target_samplerate = T::from_f64(target_samplerate_f64);
        let mut rng = Rng::with_seed(VOICE_SEED.fetch_add(1, Ordering::SeqCst));
        RawVoice {
            osc: std::array::from_fn(|i| {
                let osc_param = &params.osc_params[i];
                let pulse_width = T::from_f64(osc_param.pulse_width.value() as _);
                PolyOsc::new(
                    target_samplerate,
                    osc_param.shape.value(),
                    note_data,
                    pulse_width,
                    if osc_param.retrigger.value() {
                        T::zero()
                    } else {
                        T::from_f64(rng.f64_range(0.0..1.0))
                    },
                )
            }),
            filter: Filter::new(target_samplerate, params.filter_params.clone()),
            noise: Noise::from_rng(rng.fork()),
            osc_out_sat: bjt::CommonCollector {
                vee: -T::ONE,
                vcc: T::ONE,
                xbias: T::from_f64(0.1),
                ybias: T::from_f64(-0.1),
            },
            params: params.clone(),
            vca_env: Adsr::new(
                target_samplerate_f64 as _,
                params.vca_env.attack.value(),
                params.vca_env.decay.value(),
                params.vca_env.sustain.value(),
                params.vca_env.release.value(),
                true,
            ),
            vcf_env: Adsr::new(
                target_samplerate_f64 as _,
                params.vcf_env.attack.value(),
                params.vcf_env.decay.value(),
                params.vcf_env.sustain.value(),
                params.vcf_env.release.value(),
                true,
            ),
            note_data,
            drift: std::array::from_fn(|_| Drift::new(rng.fork(), target_samplerate_f64 as _, 0.2)),
            samplerate: target_samplerate,
            rng,
        }
    }

    fn update_osc_types(&mut self) {
        for i in 0..2 {
            let params = &self.params.osc_params[i];
            let shape = params.shape.value();
            let osc = &mut self.osc[i];
            if !osc.is_osc_shape(shape) {
                let pulse_width = T::from_f64(params.pulse_width.value() as _);
                let phase = if params.retrigger.value() {
                    T::zero()
                } else {
                    T::from_f64(self.rng.f64_range(0.0..1.0))
                };
                *osc = PolyOsc::new(self.samplerate, shape, self.note_data, pulse_width, phase);
            }
        }
    }

    fn update_envelopes(&mut self) {
        self.vca_env
            .set_attack(self.params.vca_env.attack.smoothed.next());
        self.vca_env
            .set_decay(self.params.vca_env.decay.smoothed.next());
        self.vca_env
            .set_sustain(self.params.vca_env.sustain.smoothed.next());
        self.vca_env
            .set_release(self.params.vca_env.release.smoothed.next());
        self.vcf_env
            .set_attack(self.params.vcf_env.attack.smoothed.next());
        self.vcf_env
            .set_decay(self.params.vcf_env.decay.smoothed.next());
        self.vcf_env
            .set_sustain(self.params.vcf_env.sustain.smoothed.next());
        self.vcf_env
            .set_release(self.params.vcf_env.release.smoothed.next());
    }
}

impl<T: ConstZero + ConstOne + Scalar> Voice for RawVoice<T> {
    fn active(&self) -> bool {
        !self.vca_env.is_idle()
    }

    fn note_data(&self) -> &NoteData<Self::Sample> {
        &self.note_data
    }

    fn note_data_mut(&mut self) -> &mut NoteData<Self::Sample> {
        &mut self.note_data
    }

    fn release(&mut self, _: f32) {
        nih_log!("RawVoice: release(_)");
        self.vca_env.gate(false);
        self.vcf_env.gate(false);
    }

    fn reuse(&mut self) {
        self.vca_env.gate(true);
        self.vcf_env.gate(true);
    }
}

impl<T: ConstZero + ConstOne + Scalar> DSPMeta for RawVoice<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.samplerate = T::from_f64(samplerate as _);
        for osc in &mut self.osc {
            osc.set_samplerate(samplerate);
        }
        self.filter.set_samplerate(samplerate);
        self.vca_env.set_samplerate(samplerate);
        self.vcf_env.set_samplerate(samplerate);
    }

    fn reset(&mut self) {
        for osc in &mut self.osc {
            osc.reset();
        }
        self.filter.reset();
        self.vca_env.reset();
        self.vcf_env.reset();
    }
}

impl<T: ConstZero + ConstOne + Scalar<Element = f32>> DSPProcess<0, 1> for RawVoice<T>
where
    [(); <T as SimdValue>::LANES]:,
{
    fn process(&mut self, []: [Self::Sample; 0]) -> [Self::Sample; 1] {
        const DRIFT_MAX_ST: f32 = 0.1;
        self.update_osc_types();
        self.update_envelopes();

        // Process oscillators
        let frequency = self.note_data.frequency;
        let filter_params = self.params.filter_params.clone();
        let [osc1, osc2] = std::array::from_fn(|i| {
            let osc = &mut self.osc[i];
            let params = &self.params.osc_params[i];
            let drift = &mut self.drift[i];
            let drift = drift.next_sample() * DRIFT_MAX_ST * params.drift.smoothed.next();
            let osc_freq = frequency
                * T::from_f64(semitone_to_ratio(
                    params.pitch_coarse.value() + params.pitch_fine.value() + drift,
                ) as _);
            osc.set_pulse_width(T::from_f64(params.pulse_width.smoothed.next() as _));
            let [osc] = osc.process([osc_freq]);
            osc
        });
        let noise = self.noise.next_value_f32::<T>();

        // Process filter input
        let mixer_params = &self.params.mixer_params;
        let osc_mixer = osc1 * T::from_f64(mixer_params.osc1_amplitude.smoothed.next() as _)
            + osc2 * T::from_f64(mixer_params.osc2_amplitude.smoothed.next() as _)
            + noise * T::from_f64(mixer_params.noise_amplitude.smoothed.next() as _)
            + osc1 * osc2 * T::from_f64(mixer_params.rm_amplitude.smoothed.next() as _);
        let [filter_in] = self.osc_out_sat.process([osc_mixer]);

        // Process filter
        let freq_mod = T::from_f64(filter_params.keyboard_tracking.smoothed.next() as _)
            * ratio_to_semitone(frequency / T::from_f64(440.))
            + T::from_f64(
                (filter_params.env_amt.smoothed.next() * self.vcf_env.next_sample()) as f64,
            );
        let vca = T::from_f64(self.vca_env.next_sample() as _);
        let static_amp = T::from_f64(self.params.output_level.smoothed.next() as _);
        self.filter
            .process([filter_in, freq_mod])
            .map(|x| static_amp * vca * x)
    }
}

type SynthVoice<T> = SampleAdapter<UpsampledVoice<BlockAdapter<RawVoice<T>>>, 0, 1>;

pub type VoiceManager<T> = Polyphonic<SynthVoice<T>>;

pub fn create_voice_manager<T: ConstZero + ConstOne + Scalar<Element = f32>>(
    samplerate: f32,
    params: Arc<PolysynthParams>,
) -> VoiceManager<T>
where
    [(); <T as SimdValue>::LANES]:,
{
    Polyphonic::new(samplerate, NUM_VOICES, move |samplerate, note_data| {
        let target_samplerate = OVERSAMPLE as f64 * samplerate as f64;
        SampleAdapter::new(UpsampledVoice::new(
            OVERSAMPLE,
            MAX_BUFFER_SIZE,
            BlockAdapter(RawVoice::new(target_samplerate, params.clone(), note_data)),
        ))
    })
}

pub type Voices<T> = VoiceManager<T>;

pub fn create_voices<T: ConstZero + ConstOne + Scalar<Element = f32>>(
    samplerate: f32,
    params: Arc<PolysynthParams>,
) -> Voices<T>
where
    [(); <T as SimdValue>::LANES]:,
{
    create_voice_manager(samplerate, params)
}

#[derive(Debug, Copy, Clone)]
pub struct Effects<T> {
    dc_blocker: DcBlocker<T>,
    bjt: bjt::CommonCollector<T>,
}

impl<T: Scalar> DSPMeta for Effects<T> {
    type Sample = T;

    fn set_samplerate(&mut self, samplerate: f32) {
        self.dc_blocker.set_samplerate(samplerate);
    }

    fn latency(&self) -> usize {
        self.dc_blocker.latency()
    }

    fn reset(&mut self) {
        self.dc_blocker.reset();
    }
}

impl<T: Scalar> DSPProcess<1, 1> for Effects<T> {
    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.bjt.process(self.dc_blocker.process(x))
    }
}

impl<T: Scalar> Effects<T> {
    pub fn new(samplerate: f32) -> Self {
        Self {
            dc_blocker: DcBlocker::new(samplerate),
            bjt: bjt::CommonCollector {
                vee: T::from_f64(-2.5),
                vcc: T::from_f64(2.5),
                xbias: T::from_f64(0.1),
                ybias: T::from_f64(-0.1),
            },
        }
    }
}

pub fn create_effects<T: Scalar>(samplerate: f32) -> Effects<T> {
    Effects::new(samplerate)
}
