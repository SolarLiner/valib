use rstest::{fixture, rstest};
use smol::stream;
use smol::stream::StreamExt;
use std::path::Path;
use valib_preset_manager::PresetManager;

#[fixture]
fn preset_manager() -> PresetManager<toml::Value> {
    let factory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("presets");
    let user = Path::new(env!("CARGO_TARGET_TMPDIR")).to_path_buf();
    smol::block_on(PresetManager::new_direct([factory.as_ref()], user))
}

#[rstest]
fn list_banks(preset_manager: PresetManager<toml::Value>) {
    let vec = preset_manager.banks().collect::<Vec<_>>();
    insta::assert_debug_snapshot!(&vec);
}

#[rstest]
fn list_presets(preset_manager: PresetManager<toml::Value>) {
    let vec = smol::block_on(
        stream::iter(preset_manager.banks())
            .flat_map(|bank| preset_manager.preset_names(&bank))
            .collect::<Vec<_>>(),
    );
    insta::assert_debug_snapshot!(&vec);
}
