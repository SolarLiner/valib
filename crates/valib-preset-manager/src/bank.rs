//! # Banks
//!
//! Banks are containers of presets, and are represented on disk by folders of preset files.
use crate::data::{PresetData, PresetDeserializeError, PresetSerializeError, PresetV1, EXTENSION};
use dashmap::DashMap;
use futures::TryStreamExt;
use smol::lock::Mutex;
use smol::stream::{Stream, StreamExt as _};
use smol::{fs, pin, stream};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Represents a single bank.
///
/// Presets are not all loaded when the bank is instantiated; instead, only the base directory is
/// stored, and the preset is loaded from disk on demand.
///
/// The bank also manages a cache of loaded presets to speed up repeated loads.
#[derive(Debug)]
pub struct Bank<Data> {
    base_dir: PathBuf,
    preset_cache: Mutex<HashMap<String, PresetV1<Data>>>,
}

impl<Data: 'static> Bank<Data> {
    /// Create a new bank representing the specified bank folder on disk.
    ///
    /// # Arguments
    ///
    /// * `base_dir`: Folder containing the bank's presets on disk.
    ///
    /// returns: Bank<Data>
    pub fn new(base_dir: impl Into<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            base_dir: base_dir.into(),
            preset_cache: Mutex::default(),
        })
    }

    /// Lists all the preset files available in the bank folder
    pub fn list_files(self: &Arc<Self>) -> impl Stream<Item = PathBuf> {
        stream::once_future(fs::read_dir(self.base_dir.clone()))
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

    pub async fn num_presets(self: &Arc<Self>) -> usize {
        self.list_files().count().await
    }

    /// List the preset names (file stems) in this bank.
    pub fn preset_names(self: &Arc<Self>) -> impl Stream<Item = String> {
        self.list_files().filter_map(|path| {
            path.file_stem()
                .map(|oss| oss.to_string_lossy().into_owned())
        })
    }
}

impl<Data: PresetData> Bank<Data> {
    /// Loads a preset from the given preset title
    ///
    /// # Arguments
    ///
    /// * `title`: Preset title
    ///
    /// returns: Result<PresetV1<Data>, PresetDeserializeError>
    pub async fn load_preset(&self, title: &str) -> Result<PresetV1<Data>, PresetDeserializeError>
    where
        Data: Clone,
    {
        let mut preset_cache = self.preset_cache.lock().await;
        Ok(if preset_cache.contains_key(title) {
            preset_cache[title].clone()
        } else {
            let preset = PresetV1::from_title(&self.base_dir, title).await?;
            preset_cache.insert(title.to_string(), preset.clone());
            preset
        })
    }

    /// Load all presets in this bank, returning an unordered stream of loaded presets.
    ///
    /// Failed presets are logged to stderr.
    pub async fn load_all_presets(self: &Arc<Self>) -> impl Stream<Item = PresetV1<Data>> {
        self.list_files()
            .flat_map(|path| stream::once_future(async move { PresetV1::from_file(&path).await }))
            .filter_map(|res| match res {
                Ok(data) => Some(data),
                Err(err) => {
                    eprintln!("Error loading preset: {err}");
                    None
                }
            })
    }

    /// Saves the preset into the bank.
    ///
    /// # Arguments
    ///
    /// * `preset`: Preset to save
    pub async fn save_preset(&self, preset: PresetV1<Data>) -> Result<(), PresetSerializeError> {
        preset.write_to_bank(&self.base_dir, true).await?;

        let mut preset_cache = self.preset_cache.lock().await;
        preset_cache.insert(preset.metadata.title.clone(), preset);
        Ok(())
    }

    /// Retrieve the preset name N position away from the current preset as given by its name.
    ///
    /// # Arguments
    ///
    /// * `current_name`: Name of the "current preset"
    /// * `offset`: Offset to navigate by
    ///
    /// returns: Option<String>
    pub async fn navigate_by_offset(
        self: &Arc<Self>,
        current_name: impl ToString,
        offset: isize,
    ) -> Option<String> {
        let current_name = current_name.to_string();
        let names_stream = self.clone().preset_names();
        pin!(names_stream);
        let pos = names_stream
            .position(move |name| name == current_name)
            .await?;
        self.preset_at_position((pos as isize + offset) as usize)
            .await
    }

    pub async fn preset_at_position(self: &Arc<Self>, pos: usize) -> Option<String> {
        let names_stream = self.preset_names();
        pin!(names_stream);
        names_stream.nth(pos).await
    }

    #[inline]
    pub async fn previous_preset(self: &Arc<Self>, current_name: impl ToString) -> Option<String> {
        self.navigate_by_offset(current_name, -1).await
    }

    #[inline]
    pub async fn next_preset(self: &Arc<Self>, current_name: impl ToString) -> Option<String> {
        self.navigate_by_offset(current_name, 1).await
    }
}

/// The bank group groups banks. (duh)
///
/// Banks being folders of presets, a bank group is a folder of bank folders.
#[derive(Debug, Clone)]
pub struct BankGroup<Data> {
    banks: DashMap<PathBuf, Arc<Bank<Data>>>,
}

impl<Data> Default for BankGroup<Data> {
    fn default() -> Self {
        Self {
            banks: DashMap::default(),
        }
    }
}

impl<Data> BankGroup<Data> {
    /// Create a new bank group.
    ///
    /// This method doesn't do anything. Banks will need to be manually added with
    /// [`add_bank`](Self::add_bank).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a bank group from a given parent dir.
    ///
    /// The method will scan for folders within the given parent directory, and load all of them as
    /// banks.
    ///
    /// # Arguments
    ///
    /// * `parent_dir`: Parent directory to all banks that will be grouped in this bank group
    ///   instance.
    ///
    /// returns: BankGroup<Data>
    pub async fn from_parent_dir(parent_dir: &Path) -> Self
    where
        Data: 'static,
    {
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
            .map(|path| (path.clone(), Bank::new(path)))
            .fold(DashMap::default(), |acc, (path, bank)| {
                acc.insert(path, bank);
                acc
            })
            .await;
        Self { banks }
    }

    /// Loads all banks from all parent directories passed in as a stream.
    ///
    /// Equivalent to calling `bank_group.combine(&BankGroup::from_parent_dir(dir))` for all
    /// directories emitted by the `parent_dirs` stream
    ///
    /// # Arguments
    ///
    /// * `parent_dirs`: Stream of parent directories to group into this instance.
    ///
    /// returns: BankGroup<Data>
    pub async fn from_parent_dirs(parent_dirs: impl Stream<Item = &Path>) -> Self
    where
        Data: 'static,
    {
        parent_dirs
            .map(|path| stream::once_future(Self::from_parent_dir(path)))
            .flatten()
            .fold(Self::default(), |acc, x| {
                acc.combine(&x);
                acc
            })
            .await
    }

    /// Combine two bank groups into one. This merges all the groups banks together, deduplicating
    /// paths.
    ///
    /// # Arguments
    ///
    /// * `other`: Other bank group to merge with.
    ///
    /// returns: ()
    pub fn combine(&self, other: &BankGroup<Data>) {
        for (path, bank) in other
            .banks
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
        {
            self.banks.insert(path, bank);
        }
    }

    /// Add a single bank from its folder on disk.
    ///
    /// # Arguments
    ///
    /// * `bank_dir`: Bank directory on disk
    ///
    /// returns: ()
    pub fn add_bank(&self, bank_dir: &Path) -> Arc<Bank<Data>>
    where
        Data: 'static,
    {
        let bank = Bank::new(bank_dir);
        self.banks.insert(bank_dir.to_owned(), bank.clone());
        bank
    }

    /// Add a new bank group based on the given parent directory.
    ///
    /// Equivalent to calling `group.combine(&BankGroup::from_parent_directory(parent_dir))`.
    ///
    /// # Arguments
    ///
    /// * `parent_dir`: Parent directory for the new group
    ///
    /// returns: ()
    pub async fn add_group(&mut self, parent_dir: &Path)
    where
        Data: 'static,
    {
        self.combine(&Self::from_parent_dir(parent_dir).await);
    }

    /// List all the banks this group knows about.
    pub fn banks(&self) -> impl '_ + Iterator<Item = String> {
        self.banks
            .iter()
            .map(|entry| entry.key().clone())
            .filter_map(|k| k.file_stem().map(|p| p.to_string_lossy().to_string()))
    }

    /// Returns the specified bank, if it exists.
    ///
    /// # Arguments
    ///
    /// * `name`: Name of the bank
    ///
    /// returns: Option<&Arc<Bank<Data>, Global>>
    pub fn get_bank(&self, name: &str) -> Option<Arc<Bank<Data>>> {
        self.banks
            .iter()
            .filter_map(|entry| {
                let key = entry.key();
                let filename = key.file_name().map(|p| p.to_string_lossy().to_string());
                filename
                    .is_some_and(|filename| name == filename)
                    .then_some(entry.value().clone())
            })
            .next()
    }
}
