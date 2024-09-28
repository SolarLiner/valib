use rstest::rstest;
use smol_macros::test;
use std::path::PathBuf;
use valib_preset_manager::data::PresetV1;

#[rstest]
fn read_presets_good(#[files("tests/banks/Good/*.preset")] preset_path: PathBuf) {
    let preset = smol::block_on(PresetV1::<toml::Value>::from_file(&preset_path))
        .expect("Preset should load correctly");
    insta::assert_debug_snapshot!(
        preset_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        preset
    );
}

#[rstest]
fn read_presets_bad(#[files("tests/banks/Bad/*.preset")] preset_path: PathBuf) {
    let preset = smol::block_on(PresetV1::<toml::Value>::from_file(&preset_path))
        .expect_err("Preset should not load correctly");
    insta::assert_debug_snapshot!(
        preset_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        preset
    );
}
