use rstest::rstest;
use smol::prelude::*;
use std::path::Path;
use valib_preset_manager::bank::Bank;

const BANKS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/banks");

#[rstest]
#[case("Good")]
#[case("Bad")]
fn list_bank_presets(#[case] bank_name: &str) {
    let bank = Path::new(BANKS_DIR).join(bank_name);
    let bank = Bank::<toml::Value>::new(bank);
    let mut data: Vec<_> = smol::block_on(
        bank.list_files()
            .filter_map(|path| {
                path.file_name()
                    .map(|oss| oss.to_string_lossy().to_string())
            })
            .collect(),
    );
    data.sort_by_cached_key(|p| p.to_string());

    insta::assert_csv_snapshot!(bank_name, &data);
}
