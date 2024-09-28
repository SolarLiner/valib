use crate::data::{PresetData, PresetDeserializeError, PresetSerializeError, PresetV1, EXTENSION};
use futures::lock::Mutex;
use futures::TryStreamExt;
use smol::prelude::*;
use smol::{fs, stream};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug)]
pub struct Bank<Data> {
    base_dir: PathBuf,
    preset_cache: Mutex<HashMap<String, PresetV1<Data>>>,
}

impl<Data> Bank<Data> {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            preset_cache: Mutex::default(),
        }
    }

    pub fn list_files(&self) -> impl '_ + Stream<Item = PathBuf> {
        stream::once_future(fs::read_dir(&self.base_dir))
            .try_flatten_unordered(None)
            .try_filter_map(|entry| async move {
                if entry.metadata().await?.is_file() {
                    Ok(Some(entry.path()))
                } else {
                    Ok(None)
                }
            })
            .filter_map(|res| match res {
                Ok(path) => Some(path),
                Err(err) => {
                    eprintln!("Cannot read bank path: {err}");
                    None
                }
            })
            .filter(|path| {
                path.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(EXTENSION))
            })
    }
}

impl<Data: Clone + PresetData> Bank<Data> {
    pub async fn load_preset(&self, title: &str) -> Result<PresetV1<Data>, PresetDeserializeError> {
        let mut preset_cache = self.preset_cache.lock().await;
        Ok(if preset_cache.contains_key(title) {
            preset_cache[title].clone()
        } else {
            let preset = PresetV1::from_title(&self.base_dir, title).await?;
            preset_cache.insert(title.to_string(), preset.clone());
            preset
        })
    }

    pub async fn save_preset(&self, preset: PresetV1<Data>) -> Result<(), PresetSerializeError> {
        preset.write_to_bank(&self.base_dir, true).await?;

        let mut preset_cache = self.preset_cache.lock().await;
        preset_cache.insert(preset.metadata.title.clone(), preset);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BankManager<Data> {
    banks: HashMap<PathBuf, Arc<Bank<Data>>>,
}

impl<Data> Default for BankManager<Data> {
    fn default() -> Self {
        Self {
            banks: HashMap::default(),
        }
    }
}

impl<Data> BankManager<Data> {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn from_parent_dir(parent_dir: &Path) -> Self {
        let banks = stream::once_future(fs::read_dir(parent_dir))
            .try_flatten_unordered(None)
            .try_filter_map(|entry| async move {
                if entry.metadata().await?.is_dir() {
                    Ok(Some(entry.path()))
                } else {
                    Ok(None)
                }
            })
            .filter_map(|res| match res {
                Ok(path) => Some(path),
                Err(err) => {
                    eprintln!("Cannot read bank directory: {err}");
                    None
                }
            })
            .map(|path| (path.clone(), Arc::new(Bank::new(path))))
            .collect()
            .await;
        Self { banks }
    }

    pub async fn from_parent_dirs(parent_dirs: impl Stream<Item = &Path>) -> Self {
        parent_dirs
            .map(|path| stream::once_future(Self::from_parent_dir(path)))
            .flatten()
            .fold(Self::default(), |mut acc, x| {
                acc.combine(&x);
                acc
            })
            .await
    }

    pub fn combine(&mut self, other: &BankManager<Data>) {
        self.banks
            .extend(other.banks.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
}
