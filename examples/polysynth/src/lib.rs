use crate::params::PolysynthParams;
use nih_plug::audio_setup::{AudioIOLayout, AuxiliaryBuffers};
use nih_plug::buffer::Buffer;
use nih_plug::params::Params;
use nih_plug::plugin::ProcessStatus;
use nih_plug::prelude::*;
use std::cmp::Ordering;
use std::sync::{atomic, Arc};
use valib::dsp::buffer::{AudioBufferMut, AudioBufferRef};
use valib::dsp::{BlockAdapter, DSPMeta, DSPProcessBlock};
use valib::util::Rms;
use valib::voice::{NoteData, VoiceId, VoiceManager};

mod dsp;
mod editor;
mod params;

const NUM_VOICES: usize = 16;
const OVERSAMPLE: usize = 8;
const MAX_BUFFER_SIZE: usize = 64;

const POLYMOD_OSC_AMP: [u32; 2] = [0, 1];
const POLYMOD_OSC_PITCH_COARSE: [u32; 2] = [2, 3];
const POLYMOD_OSC_PITCH_FINE: [u32; 2] = [4, 5];
const POLYMOD_FILTER_CUTOFF: u32 = 6;

#[derive(Debug, Copy, Clone)]
struct VoiceKey {
    voice_id: Option<i32>,
    channel: u8,
    note: u8,
}

impl PartialEq for VoiceKey {
    fn eq(&self, other: &Self) -> bool {
        match (self.voice_id, other.voice_id) {
            (Some(a), Some(b)) => a == b,
            _ => self.channel == other.channel && self.note == other.note,
        }
    }
}

impl Eq for VoiceKey {}

impl Ord for VoiceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.voice_id, other.voice_id) {
            (Some(a), Some(b)) => a.cmp(&b),
            _ => self
                .channel
                .cmp(&other.channel)
                .then(self.note.cmp(&other.note)),
        }
    }
}

impl PartialOrd for VoiceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl VoiceKey {
    fn new(voice_id: Option<i32>, channel: u8, note: u8) -> Self {
        Self {
            voice_id,
            channel,
            note,
        }
    }
}

#[derive(Debug)]
struct VoiceIdMap {
    data: [Option<(VoiceKey, VoiceId<dsp::Dsp<f32>>)>; NUM_VOICES],
}

impl Default for VoiceIdMap {
    fn default() -> Self {
        Self {
            data: [None; NUM_VOICES],
        }
    }
}

impl VoiceIdMap {
    fn add_voice(&mut self, key: VoiceKey, v: VoiceId<dsp::Dsp<f32>>) -> bool {
        let Some(position) = self.data.iter().position(|x| x.is_none()) else {
            return false;
        };
        self.data[position] = Some((key, v));
        true
    }

    fn get_voice(&self, key: VoiceKey) -> Option<VoiceId<dsp::Dsp<f32>>> {
        self.data.iter().find_map(|x| {
            x.as_ref()
                .and_then(|(vkey, id)| (*vkey == key).then_some(*id))
        })
    }

    fn get_voice_by_poly_id(&self, voice_id: i32) -> Option<VoiceId<dsp::Dsp<f32>>> {
        self.data
            .iter()
            .flatten()
            .find_map(|(vkey, id)| (vkey.voice_id == Some(voice_id)).then_some(*id))
    }

    fn remove_voice(&mut self, key: VoiceKey) -> Option<(VoiceKey, VoiceId<dsp::Dsp<f32>>)> {
        let position = self
            .data
            .iter()
            .position(|x| x.as_ref().is_some_and(|(vkey, _)| *vkey == key))?;
        self.data[position].take()
    }
}

type SynthSample = f32;

#[derive(Debug)]
pub struct PolysynthPlugin {
    dsp: BlockAdapter<dsp::Dsp<SynthSample>>,
    params: Arc<PolysynthParams>,
    voice_id_map: VoiceIdMap,
}

impl Default for PolysynthPlugin {
    fn default() -> Self {
        const DEFAULT_SAMPLERATE: f32 = 44100.;
        let params = Arc::new(PolysynthParams::default());
        Self {
            dsp: BlockAdapter(dsp::create(DEFAULT_SAMPLERATE, params.clone())),
            params,
            voice_id_map: VoiceIdMap::default(),
        }
    }
}

impl Plugin for PolysynthPlugin {
    const NAME: &'static str = "Polysynth";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://github.com/SolarLiner/valib";
    const EMAIL: &'static str = "me@solarliner.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(0),
        main_output_channels: NonZeroU32::new(1),
        ..AudioIOLayout::const_default()
    }];
    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.params.editor_state.clone())
    }

    fn initialize(
        &mut self,
        _: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;
        self.dsp.set_samplerate(sample_rate);
        true
    }

    fn reset(&mut self) {
        self.dsp.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        let sample_rate = context.transport().sample_rate;
        let output = buffer.as_slice();

        let mut next_event = context.next_event();
        let mut block_start: usize = 0;
        let mut block_end: usize = MAX_BUFFER_SIZE.min(num_samples);
        while block_start < num_samples {
            'events: loop {
                match next_event {
                    Some(event) if (event.timing() as usize) <= block_start => match event {
                        NoteEvent::NoteOn {
                            voice_id,
                            channel,
                            note,
                            velocity,
                            ..
                        } => {
                            let key = VoiceKey::new(voice_id, channel, note);
                            let note_data = NoteData::from_midi(note, velocity);
                            let id = self.dsp.note_on(note_data);
                            nih_log!("Note on {id} <- {key:?}");
                            self.voice_id_map.add_voice(key, id);
                        }
                        NoteEvent::NoteOff {
                            voice_id,
                            channel,
                            note,
                            velocity,
                            ..
                        } => {
                            let key = VoiceKey::new(voice_id, channel, note);
                            if let Some((_, id)) = self.voice_id_map.remove_voice(key) {
                                nih_log!("Note off {id} <- {key:?}");
                                self.dsp.note_off(id, velocity);
                            } else {
                                nih_log!("Note off {key:?}: ID not found");
                            }
                        }
                        NoteEvent::Choke {
                            voice_id,
                            channel,
                            note,
                            ..
                        } => {
                            let key = VoiceKey::new(voice_id, channel, note);
                            if let Some((_, id)) = self.voice_id_map.remove_voice(key) {
                                self.dsp.choke(id);
                            }
                        }
                        NoteEvent::PolyModulation { voice_id, .. } => {
                            if let Some(id) = self.voice_id_map.get_voice_by_poly_id(voice_id) {
                                nih_log!("TODO: Poly modulation ({id})");
                            }
                        }
                        NoteEvent::MonoAutomation {
                            poly_modulation_id,
                            normalized_value,
                            ..
                        } => match poly_modulation_id {
                            POLYMOD_FILTER_CUTOFF => {
                                let target_plain_value = self
                                    .params
                                    .filter_params
                                    .cutoff
                                    .preview_plain(normalized_value);
                                self.params
                                    .filter_params
                                    .cutoff
                                    .smoothed
                                    .set_target(sample_rate, target_plain_value);
                            }
                            _ => {
                                for i in 0..2 {
                                    match poly_modulation_id {
                                        id if id == POLYMOD_OSC_PITCH_COARSE[i] => {
                                            let target_plain_value = self.params.osc_params[i]
                                                .pitch_coarse
                                                .preview_plain(normalized_value);
                                            self.params.osc_params[i]
                                                .pitch_coarse
                                                .smoothed
                                                .set_target(sample_rate, target_plain_value);
                                        }
                                        id if id == POLYMOD_OSC_PITCH_FINE[i] => {
                                            let target_plain_value = self.params.osc_params[i]
                                                .pitch_fine
                                                .preview_plain(normalized_value);
                                            self.params.osc_params[i]
                                                .pitch_fine
                                                .smoothed
                                                .set_target(sample_rate, target_plain_value);
                                        }
                                        id if id == POLYMOD_OSC_AMP[i] => {
                                            let target_plain_value = self.params.osc_params[i]
                                                .amplitude
                                                .preview_plain(normalized_value);
                                            self.params.osc_params[i]
                                                .amplitude
                                                .smoothed
                                                .set_target(sample_rate, target_plain_value);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        },
                        _ => {}
                    },
                    Some(event) if (event.timing() as usize) < block_end => {
                        block_end = event.timing() as usize;
                        break 'events;
                    }
                    _ => break 'events,
                }
                next_event = context.next_event();
            }
            let dsp_block = AudioBufferMut::from(&mut output[0][block_start..block_end]);
            let input = AudioBufferRef::<SynthSample, 0>::empty(dsp_block.samples());
            self.dsp.process_block(input, dsp_block);

            block_start = block_end;
            block_end = (block_start + MAX_BUFFER_SIZE).min(num_samples);
        }

        self.dsp.0.clean_inactive_voices();
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for PolysynthPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"VaLibPlySynTHSLN";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Synth,
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Mono,
    ];
}

impl ClapPlugin for PolysynthPlugin {
    const CLAP_ID: &'static str = "dev.solarliner.valib.polysynth";
    const CLAP_DESCRIPTION: Option<&'static str> = option_env!("CARGO_PKG_DESCRIPTION");
    const CLAP_MANUAL_URL: Option<&'static str> = option_env!("CARGO_PKG_MANIFEST_URL");
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Synthesizer,
        ClapFeature::Instrument,
        ClapFeature::Mono,
    ];
    const CLAP_POLY_MODULATION_CONFIG: Option<PolyModulationConfig> = Some(PolyModulationConfig {
        // If the plugin's voice capacity changes at runtime (for instance, when switching to a
        // monophonic mode), then the plugin should inform the host in the `initialize()` function
        // as well as in the `process()` function if it changes at runtime using
        // `context.set_current_voice_capacity()`
        max_voice_capacity: NUM_VOICES as _,
        // This enables voice stacking in Bitwig.
        supports_overlapping_voices: true,
    });
}

nih_export_clap!(PolysynthPlugin);
nih_export_vst3!(PolysynthPlugin);
