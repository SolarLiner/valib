use crate::{AbrasiveParams, AbrasiveParamsSerialized};
use nih_plug::nih_error;
use nih_plug::prelude::Params;
use nih_plug_vizia::vizia::prelude::*;
use std::sync::Arc;
use valib::preset_manager as pm;
use valib::preset_manager::data::{PresetMeta, PresetV1};

#[derive(Debug, Clone)]
pub enum PresetManagerEvent {
    Navigate(isize),
    LoadPreset(String, String),
    LoadPresetDirect {
        preset: PresetV1<AbrasiveParamsSerialized>,
        preset_name: String,
    },
    SaveCurrentPreset {
        bank: String,
        preset_meta: PresetMeta,
    },
}

#[derive(Debug)]
pub struct PresetManager {
    preset_manager: Arc<pm::PresetManager<AbrasiveParamsSerialized>>,
    params: Arc<AbrasiveParams>,
}

#[derive(Debug, Clone, Lens)]
struct PresetManagerData {
    current_preset_name: String,
    current_preset: PresetMeta,
}

impl Model for PresetManagerData {
    fn event(&mut self, _: &mut EventContext, event: &mut Event) {
        event.map(|event, _| match event {
            PresetManagerEventInternal::SetCurrentPreset { name, meta } => {
                self.current_preset = meta.clone();
                self.current_preset_name = name.clone();
            }
        });
    }
}

enum PresetManagerEventInternal {
    SetCurrentPreset { name: String, meta: PresetMeta },
}

impl PresetManager {
    pub fn new(cx: &mut Context, params: Arc<AbrasiveParams>) -> Handle<Self> {
        let preset_manager = Arc::new(smol::block_on(pm::PresetManager::new(
            "dev.solarliner",
            "valib",
            "Abrasive",
        )));
        let data = PresetManagerData {
            current_preset_name: "Init".to_string(),
            current_preset: Self::init_meta(),
        };
        Self {
            preset_manager,
            params,
        }
        .build(cx, |cx| {
            data.build(cx);

            Spinbox::new(
                cx,
                PresetManagerData::current_preset.map(|meta| meta.title.clone()),
                SpinboxKind::Horizontal,
                SpinboxIcons::Chevrons,
            )
            .on_decrement(|cx| cx.emit(PresetManagerEvent::Navigate(-1)))
            .on_increment(|cx| cx.emit(PresetManagerEvent::Navigate(1)));
        })
    }

    fn init_meta() -> PresetMeta {
        PresetMeta {
            title: "Init".to_string(),
            author: "User".to_string(),
            tags: Default::default(),
            revision: 0,
            other: Default::default(),
        }
    }
}

impl View for PresetManager {
    fn element(&self) -> Option<&'static str> {
        Some("preset-manager")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(
            |app_event: &PresetManagerEvent, _meta| match app_event.clone() {
                PresetManagerEvent::Navigate(offset) => {
                    let preset_manager = self.preset_manager.clone();
                    let cur_preset_name = cx
                        .data::<PresetManagerData>()
                        .unwrap()
                        .current_preset_name
                        .clone();
                    cx.spawn(move |cx| {
                        smol::block_on(async move {
                            match preset_manager
                                .load_with_offset(cur_preset_name.as_ref(), offset)
                                .await
                            {
                                Ok(Some(preset)) => {
                                    let preset_name = preset.metadata.title.clone();
                                    cx.emit(PresetManagerEvent::LoadPresetDirect {
                                        preset,
                                        preset_name,
                                    })
                                    .unwrap();
                                }
                                Err(err) => nih_error!("Error loading preset: {err}"),
                                Ok(None) => {}
                            }
                        })
                    });
                }
                PresetManagerEvent::LoadPreset(bank, preset_name) => {
                    let bank = bank.clone();
                    let preset_name = preset_name.clone();
                    let preset_manager = self.preset_manager.clone();
                    cx.spawn(move |cx| {
                        smol::block_on(async move {
                            let try_ = async {
                                Some(preset_manager.bank(&bank)?.load_preset(&preset_name).await)
                            };
                            match try_.await {
                                Some(Ok(preset)) => {
                                    let preset_name = preset.metadata.title.clone();
                                    cx.emit(PresetManagerEvent::LoadPresetDirect {
                                        preset,
                                        preset_name,
                                    })
                                    .unwrap();
                                }
                                Some(Err(err)) => {
                                    nih_error!("Cannot load preset: {err}");
                                }
                                None => nih_error!("Bank not found: {bank}"),
                            }
                        });
                    })
                }
                PresetManagerEvent::LoadPresetDirect { preset, .. } => {
                    self.params.deserialize_fields(&preset.data.0);
                    let name = preset.metadata.title.clone();
                    cx.emit(PresetManagerEventInternal::SetCurrentPreset {
                        meta: preset.metadata.clone(),
                        name,
                    })
                }
                PresetManagerEvent::SaveCurrentPreset { bank, preset_meta } => {
                    let param_data = self.params.serialize_fields();
                    let preset_manager = self.preset_manager.clone();
                    cx.spawn(|_| {
                        smol::block_on(async move {
                            match preset_manager
                                .save_into_bank(
                                    &bank,
                                    PresetV1::new(
                                        preset_meta,
                                        AbrasiveParamsSerialized(param_data),
                                    ),
                                )
                                .await
                            {
                                Ok(()) => {}
                                Err(err) => {
                                    nih_error!("Error saving preset: {err}");
                                }
                            }
                        })
                    });
                }
            },
        )
    }
}
