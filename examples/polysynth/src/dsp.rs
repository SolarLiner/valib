use crate::params::{FilterParams, OscParams, OscShape, PolysynthParams};
use crate::{MAX_BUFFER_SIZE, NUM_VOICES, OVERSAMPLE};
use nih_plug::nih_log;
use nih_plug::util::db_to_gain_fast;
use num_traits::{ConstOne, ConstZero};
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
    fn new(samplerate: T, shape: OscShape, note_data: NoteData<T>, pulse_width: T) -> Self {
        match shape {
            OscShape::Sine => Self::Sine(Phasor::new(samplerate, note_data.frequency)),
            OscShape::Triangle => Self::Triangle(Triangle::new(samplerate, note_data.frequency)),
            OscShape::Square => Self::Square(Square::new(
                samplerate,
                note_data.frequency,
                SquareBLEP::new(pulse_width),
            )),
            OscShape::Saw => Self::Sawtooth(Sawtooth::new(
                samplerate,
                note_data.frequency,
                SawBLEP::default(),
            )),
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
    osc_params: [Arc<OscParams>; 2],
    filter_params: Arc<FilterParams>,
    gate: SmoothedParam,
    note_data: NoteData<T>,
    samplerate: T,
}

impl<T: ConstZero + ConstOne + Scalar> RawVoice<T> {
    pub(crate) fn update_osc_types(&mut self) {
        for i in 0..2 {
            let params = &self.osc_params[i];
            let shape = params.shape.value();
            let osc = &mut self.osc[i];
            if !osc.is_osc_shape(shape) {
                let pulse_width = T::from_f64(params.pulse_width.value() as _);
                *osc = PolyOsc::new(self.samplerate, shape, self.note_data, pulse_width);
            }
        }
    }
}

impl<T: ConstZero + ConstOne + Scalar> Voice for RawVoice<T> {
    fn active(&self) -> bool {
        self.gate.current_value() > 0.5
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
        let frequency = self.note_data.frequency;
        self.update_osc_types();
        let osc1_freq = frequency
            * T::from_f64(semitone_to_ratio(
                self.osc_params[0].pitch_coarse.value() + self.osc_params[0].pitch_fine.value(),
            ) as _);
        let osc2_freq = frequency
            * T::from_f64(semitone_to_ratio(
                self.osc_params[1].pitch_coarse.value() + self.osc_params[1].pitch_fine.value(),
            ) as _);
        let [osc1] = self.osc[0].process([osc1_freq]);
        let [osc2] = self.osc[1].process([osc2_freq]);
        let osc_mixer = osc1 * T::from_f64(self.osc_params[0].amplitude.smoothed.next() as _)
            + osc2 * T::from_f64(self.osc_params[1].amplitude.smoothed.next() as _);
        let filter_in = self
            .osc_out_sat
            .process([osc_mixer])
            .map(|x| T::from_f64(db_to_gain_fast(9.0) as _) * x);
        self.filter
            .set_cutoff(T::from_f64(self.filter_params.cutoff.smoothed.next() as _));
        self.filter.set_resonance(T::from_f64(
            4f64 * self.filter_params.resonance.smoothed.next() as f64,
        ));
        let vca = self.gate.next_sample_as::<T>();
        self.filter.process(filter_in).map(|x| vca * x)
    }
}

type SynthVoice<T> = SampleAdapter<UpsampledVoice<BlockAdapter<RawVoice<T>>>, 0, 1>;

pub type VoiceManager<T> = Polyphonic<SynthVoice<T>>;

pub fn create_voice_manager<T: ConstZero + ConstOne + Scalar>(
    samplerate: f32,
    osc_params: [Arc<OscParams>; 2],
    filter_params: Arc<FilterParams>,
) -> VoiceManager<T> {
    let target_samplerate_f64 = OVERSAMPLE as f64 * samplerate as f64;
    let target_samplerate = T::from_f64(target_samplerate_f64);
    Polyphonic::new(samplerate, NUM_VOICES, move |_, note_data| {
        SampleAdapter::new(UpsampledVoice::new(
            2,
            MAX_BUFFER_SIZE,
            BlockAdapter(RawVoice {
                osc: std::array::from_fn(|i| {
                    let osc_param = &osc_params[i];
                    let pulse_width = T::from_f64(osc_param.pulse_width.value() as _);
                    PolyOsc::new(
                        target_samplerate,
                        osc_param.shape.value(),
                        note_data,
                        pulse_width,
                    )
                }),
                osc_params: osc_params.clone(),
                filter: Ladder::new(
                    target_samplerate_f64,
                    T::from_f64(filter_params.cutoff.value() as _),
                    T::from_f64(filter_params.resonance.value() as _),
                ),
                filter_params: filter_params.clone(),
                osc_out_sat: bjt::CommonCollector {
                    vee: -T::ONE,
                    vcc: T::ONE,
                    xbias: T::from_f64(0.1),
                    ybias: T::from_f64(-0.1),
                },
                gate: SmoothedParam::exponential(1., target_samplerate_f64 as _, 1.0),
                note_data,
                samplerate: target_samplerate,
            }),
        ))
    })
}

pub type Dsp<T> = VoiceManager<T>;

pub fn create<T: ConstZero + ConstOne + Scalar>(
    samplerate: f32,
    params: &PolysynthParams,
) -> Dsp<T> {
    create_voice_manager(
        samplerate,
        params.osc_params.clone(),
        params.filter_params.clone(),
    )
}
