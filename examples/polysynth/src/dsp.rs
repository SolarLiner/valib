use crate::params::{FilterParams, OscShape, PolysynthParams};
use crate::{MAX_BUFFER_SIZE, NUM_VOICES, OVERSAMPLE};
use fastrand::Rng;
use fastrand_contrib::RngExt;
use nih_plug::nih_log;
use nih_plug::util::db_to_gain_fast;
use num_traits::{ConstOne, ConstZero};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, SampleAdapter};
use valib::filters::ladder::{Ladder, OTA};
use valib::math::interpolation::{sine_interpolation, Interpolate, Sine};
use valib::oscillators::polyblep::{SawBLEP, Sawtooth, Square, SquareBLEP, Triangle};
use valib::oscillators::Phasor;
use valib::saturators::{bjt, Tanh};
use valib::simd::SimdBool;
use valib::util::semitone_to_ratio;
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
            attack_base: 1. + Self::TARGET_RATIO_A,
            decay_base: -Self::TARGET_RATIO_DR,
            release_base: -Self::TARGET_RATIO_DR,
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
    const TARGET_RATIO_A: f32 = 0.3;
    const TARGET_RATIO_DR: f32 = 1e-4;
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
        self.attack_coeff = Self::calc_coeff(self.attack_rate, Self::TARGET_RATIO_A);
        self.attack_base = (1. + Self::TARGET_RATIO_A) * (1.0 - self.attack_coeff);
    }

    pub fn set_decay(&mut self, decay: f32) {
        if (self.decay - decay).abs() < 1e-6 {
            return;
        }
        self.decay = decay;
        self.decay_rate = self.samplerate * decay;
        self.decay_coeff = Self::calc_coeff(self.decay_rate, Self::TARGET_RATIO_DR);
        self.decay_base = (self.sustain - Self::TARGET_RATIO_DR) * (1. - self.decay_coeff);
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
        self.release_coeff = Self::calc_coeff(self.release_rate, Self::TARGET_RATIO_DR);
        self.release_base = -Self::TARGET_RATIO_DR * (1. - self.release_coeff);
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

pub(crate) const NUM_OSCILLATORS: usize = 2;

pub struct RawVoice<T: ConstZero + ConstOne + Scalar> {
    osc: [PolyOsc<T>; NUM_OSCILLATORS],
    osc_out_sat: bjt::CommonCollector<T>,
    filter: Ladder<T, OTA<Tanh>>,
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
            filter: Ladder::new(
                target_samplerate_f64,
                T::from_f64(params.filter_params.cutoff.value() as _),
                T::from_f64(params.filter_params.resonance.value() as _),
            ),
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

impl<T: ConstZero + ConstOne + Scalar> DSPProcess<0, 1> for RawVoice<T> {
    fn process(&mut self, []: [Self::Sample; 0]) -> [Self::Sample; 1] {
        const DRIFT_MAX_ST: f32 = 0.1;
        self.update_osc_types();
        self.update_envelopes();

        // Process oscillators
        let frequency = self.note_data.frequency;
        let osc_params = self.params.osc_params.clone();
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

        // Process filter input
        let osc_mixer = osc1 * T::from_f64(osc_params[0].amplitude.smoothed.next() as _)
            + osc2 * T::from_f64(osc_params[1].amplitude.smoothed.next() as _);
        let filter_in = self
            .osc_out_sat
            .process([osc_mixer])
            .map(|x| T::from_f64(db_to_gain_fast(9.0) as _) * x);

        let freq_ratio = T::from_f64(filter_params.keyboard_tracking.smoothed.next() as _)
            * frequency
            / T::from_f64(440.)
            + T::from_f64(semitone_to_ratio(
                filter_params.env_amt.smoothed.next() * self.vcf_env.next_sample(),
            ) as _);
        let filter_freq =
            (T::one() + freq_ratio) * T::from_f64(filter_params.cutoff.smoothed.next() as _);

        // Process filter
        self.filter.set_cutoff(filter_freq);
        self.filter.set_resonance(T::from_f64(
            4f64 * filter_params.resonance.smoothed.next() as f64,
        ));
        let vca = T::from_f64(self.vca_env.next_sample() as _);
        let static_amp = T::from_f64(self.params.output_level.smoothed.next() as _);
        self.filter.process(filter_in).map(|x| static_amp * vca * x)
    }
}

type SynthVoice<T> = SampleAdapter<UpsampledVoice<BlockAdapter<RawVoice<T>>>, 0, 1>;

pub type VoiceManager<T> = Polyphonic<SynthVoice<T>>;

pub fn create_voice_manager<T: ConstZero + ConstOne + Scalar>(
    samplerate: f32,
    params: Arc<PolysynthParams>,
) -> VoiceManager<T> {
    let target_samplerate = OVERSAMPLE as f64 * samplerate as f64;
    Polyphonic::new(samplerate, NUM_VOICES, move |_, note_data| {
        SampleAdapter::new(UpsampledVoice::new(
            OVERSAMPLE,
            MAX_BUFFER_SIZE,
            BlockAdapter(RawVoice::new(target_samplerate, params.clone(), note_data)),
        ))
    })
}

pub type Dsp<T> = VoiceManager<T>;

pub fn create<T: ConstZero + ConstOne + Scalar>(
    samplerate: f32,
    params: Arc<PolysynthParams>,
) -> Dsp<T> {
    create_voice_manager(samplerate, params)
}
