use crate::params::{OscShape, PolysynthParams};
use crate::{MAX_BUFFER_SIZE, NUM_VOICES, OVERSAMPLE};
use fastrand::Rng;
use fastrand_contrib::RngExt;
use nih_plug::nih_log;
use nih_plug::util::db_to_gain_fast;
use num_traits::{ConstOne, ConstZero};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use valib::dsp::parameter::SmoothedParam;
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcess, SampleAdapter};
use valib::filters::ladder::{Ladder, OTA};
use valib::oscillators::polyblep::{SawBLEP, Sawtooth, Square, SquareBLEP, Triangle};
use valib::oscillators::Phasor;
use valib::saturators::{bjt, Tanh};
use valib::util::semitone_to_ratio;
use valib::voice::polyphonic::Polyphonic;
use valib::voice::upsample::UpsampledVoice;
use valib::voice::{NoteData, Voice};
use valib::Scalar;

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

pub struct RawVoice<T: ConstZero + ConstOne + Scalar> {
    osc: [PolyOsc<T>; 2],
    osc_out_sat: bjt::CommonCollector<T>,
    filter: Ladder<T, OTA<Tanh>>,
    params: Arc<PolysynthParams>,
    gate: SmoothedParam,
    note_data: NoteData<T>,
    samplerate: T,
    rng: Rng,
}

impl<T: ConstZero + ConstOne + Scalar> RawVoice<T> {
    fn create_voice(
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
            gate: SmoothedParam::exponential(1., target_samplerate_f64 as _, 1.0),
            note_data,
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
}

impl<T: ConstZero + ConstOne + Scalar> Voice for RawVoice<T> {
    fn active(&self) -> bool {
        self.gate.current_value() > 1e-4
    }

    fn note_data(&self) -> &NoteData<Self::Sample> {
        &self.note_data
    }

    fn note_data_mut(&mut self) -> &mut NoteData<Self::Sample> {
        &mut self.note_data
    }

    fn release(&mut self, _: f32) {
        nih_log!("RawVoice: release(_)");
        self.gate.param = 0.;
    }

    fn reuse(&mut self) {
        self.gate.param = 1.;
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
    }

    fn reset(&mut self) {
        for osc in &mut self.osc {
            osc.reset();
        }
        self.filter.reset();
    }
}

impl<T: ConstZero + ConstOne + Scalar> DSPProcess<0, 1> for RawVoice<T> {
    fn process(&mut self, []: [Self::Sample; 0]) -> [Self::Sample; 1] {
        // Process oscillators
        let frequency = self.note_data.frequency;
        let osc_params = self.params.osc_params.clone();
        let filter_params = self.params.filter_params.clone();
        self.update_osc_types();
        let [osc1, osc2] = std::array::from_fn(|i| {
            let osc = &mut self.osc[i];
            let params = &self.params.osc_params[i];
            let osc_freq = frequency
                * T::from_f64(semitone_to_ratio(
                    params.pitch_coarse.value() + params.pitch_fine.value(),
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
            / T::from_f64(440.);
        let filter_freq =
            (T::one() + freq_ratio) * T::from_f64(filter_params.cutoff.smoothed.next() as _);

        // Process filter
        self.filter.set_cutoff(filter_freq);
        self.filter.set_resonance(T::from_f64(
            4f64 * filter_params.resonance.smoothed.next() as f64,
        ));
        let vca = self.gate.next_sample_as::<T>();
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
            BlockAdapter(RawVoice::create_voice(
                target_samplerate,
                params.clone(),
                note_data,
            )),
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
