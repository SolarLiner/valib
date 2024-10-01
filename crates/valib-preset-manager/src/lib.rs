//! # Preset Manager for valib
//!
//! This crate implements preset management based on the following design:
//!
//! - Presets are TOML files, containing a metadata section and a data section.
//!     - Some metadata are predefined as fields in the [`PresetMeta`](data::PresetMeta) struct,
//!       and there is also a catch-all [`other`](data::PresetMeta::other) field which can be used
//!       to save arbitrary metadata into the preset (for forward compatibility)
//!     - The data section is generic over *your* preset data. It is meant to contain parameter data
//!       but is not limited to it; you simply need to have [`serde::Deserialize`] and
//!       [`serde::Serialize`] implemented for your preset data.
//! - Banks are folders of presets. Preset files have the extension `.preset`, and the file name has
//!   no impact on the preset itself (it can be changed without affecting any preset metadata).  
//!   The code sets preset filename to its title, meaning that you cannot have presets with the same
//!   title within a single bank, but this can be overridden by specifying
//!   [`Bank::set_preset_filename_func`](data::Bank::set_preset_filename_func).
//! - A [`BankManager`](data::BankManager) manages banks appearing at a specified base directory,
//!   that is, if the base folder is `<Folder>`, and, on the file system, the structure looks like
//!   this:
//!
//!   ```raw
//!   <Folder>/
//!     Bank 1/
//!       A.preset
//!       B.preset
//!     Bank 2/
//!       C.preset
//!       D.preset
//!   ```
//!
//!   The bank manager will have two banks "Bank 1" and "Bank 2", which will respectively contain
//!   `[A, B]` and `[C, D]` inside.
//!
//! # Standard folders
//!
//! The preset manager (which counterintuitively here, manages [`BankManager`](data::BankManager)s)
//! has the concept of factory vs. user banks. It does this by looking into different standard
//! system and user folders for banks (using the [`directories`](directories)` crate).
//! Specifically, it will look into the following folders:
//! - Factory:
//!   - [Project data dir](ProjectDirs::data_dir) / presets
//!   - [Project local data dir](ProjectDirs::data_local_dir) / presets
//! - User:
//!   - [User documents](UserDirs::document_dir) / <Application Name> / presets
//!
//! The `<Application Name>` is specified when instantiating the preset manager.
//!
//! # Distribution
//!
//! To ease the development of plugins with the preset manager, a `distribution` feature toggles
//! adding the local package root as well as a factory. This is done when the `distribution` feature
//! is **off**, and removed when the `distribution` feature is toggled **on**. This is so that paths
//! to the local build directory do not end up in a distribution release build binary.
#![warn(missing_docs)]

use crate::bank::{Bank, BankGroup};
use crate::data::{PresetData, PresetDeserializeError, PresetSerializeError, PresetV1};
#[cfg(feature = "distribution")]
use directories::ProjectDirs;
use directories::UserDirs;
use futures::future::join;
use smol::stream::{Stream, StreamExt};
use smol::{fs, pin, stream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod bank;
pub mod data;

/// Preset manager, managing presets from standard directories.
#[derive(Debug, Clone)]
pub struct PresetManager<Data> {
    factory_group: BankGroup<Data>,
    user_group: BankGroup<Data>,
    user_parent_dir: PathBuf,
}

impl<Data: 'static> PresetManager<Data> {
    /// Create a new preset manager following standard directories.
    ///
    /// Most OSes now expect applications to have a reverse-domain ID split into 3 parts:
    /// `<qualifier>.<organization>.<project>`. Taking the example of an example plugin from valib,
    /// we have `dev.solarliner.valib.abrasive`. It breaks down to the following:
    ///
    /// - **Qualifier**: `dev.solarliner`
    /// - **Organization**: `valib`
    /// - **Project/Application name**: `abrasive`.
    ///
    /// These parts have an influence on most OSes in construction of "standard paths" into which
    /// the data is stored.
    ///
    /// For presets, the factory banks are stored in a global "data dir" for the application, and
    /// the user banks are stored in a folder `<project>` of their Documents user folder.
    ///
    /// # Arguments
    ///
    /// * `qualifier`: First part of the reverse-domain ID, e.g. `dev.solarliner`
    /// * `organization`: Second part of the reverse-domain ID, e.g. `valib`
    /// * `application_name`: Third part of the reverse-domain ID, e.g. `abrasive`
    ///
    /// returns: PresetManager<Data>
    #[cfg_attr(not(feature = "distribution"), allow(unused_variables))]
    pub async fn new(qualifier: &str, organization: &str, application_name: &str) -> Self {
        #[cfg(feature = "distribution")]
        let project_dirs = ProjectDirs::from(qualifier, organization, application_name)
            .expect("OS configuration mishap: cannot get project directories");
        let user_dirs =
            UserDirs::new().expect("OS configuration mishap: cannot get user directories");
        #[cfg(feature = "distribution")]
        let factory_dirs = [
            project_dirs.data_dir().to_path_buf(),
            project_dirs.data_local_dir().to_path_buf(),
        ];
        #[cfg(not(feature = "distribution"))]
        let factory_dirs = [PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets"
        ))];
        let user_dir = user_dirs
            .document_dir()
            .expect("OS configuration mishap: cannot get documents directory")
            .join(application_name)
            .join("presets");
        Self::new_direct(factory_dirs.iter().map(|p| p.as_ref()), user_dir).await
    }

    /// Create a new preset manager given the factory and user directories. Useful for testing
    /// purposes.
    ///
    /// # Arguments
    ///
    /// * `factory_dirs`: List of factory directories
    /// * `user_dirs`: User directory
    ///
    /// returns: PresetManager<Data>
    pub async fn new_direct(
        factory_dirs: impl IntoIterator<Item = &Path>,
        user_dir: impl Into<PathBuf>,
    ) -> Self {
        let user_dir = user_dir.into();
        let factory_group = BankGroup::from_parent_dirs(stream::iter(factory_dirs));
        let user_group = BankGroup::from_parent_dir(&user_dir);
        let (factory_group, user_group) = join(factory_group, user_group).await;
        Self {
            factory_group,
            user_group,
            user_parent_dir: user_dir,
        }
    }

    /// Return an iterator of all loaded banks
    pub fn banks(&self) -> impl '_ + Iterator<Item = String> {
        self.factory_group.banks().chain(self.user_group.banks())
    }

    /// Get the specified bank if it exists in this preset manager.
    ///
    /// # Arguments
    ///
    /// * `bank`: Name of the bank to retrieve
    ///
    /// returns: Option<&Arc<Bank<Data>, Global>>
    pub fn bank(&self, bank: &str) -> Option<Arc<Bank<Data>>> {
        self.factory_group
            .get_bank(bank)
            .or_else(|| self.user_group.get_bank(bank))
    }

    /// Retrieves the preset names for the current bank, if it exists.
    ///
    /// # Arguments
    ///
    /// * `bank`: Name of the bank to retrieve
    ///
    /// returns: impl Stream<Item=String>+Sized
    pub fn preset_names(&self, bank: &str) -> impl '_ + Stream<Item = String> {
        let bank = self.bank(bank);
        stream::iter(bank).flat_map(|p| p.preset_names())
    }
}

impl<Data: PresetData + Clone> PresetManager<Data> {
    /// Read a preset by its parent bank and its name.
    ///
    /// # Arguments
    ///
    /// * `bank`: Parent bank to search the preset into
    /// * `name`: Name of the preset
    ///
    /// returns: Option<PresetV1<Data>>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub async fn preset_by_name(&self, bank: &str, name: &str) -> Option<PresetV1<Data>> {
        let bank = self.bank(bank)?;
        let preset = bank.load_preset(name).await.ok()?;
        Some(preset)
    }

    /// Save a preset into the given bank, creating the bank as user if it does not exist.
    ///
    /// # Arguments
    ///
    /// * `bank`: Name of the bank to add the preset into
    /// * `preset`: Preset to save
    ///
    /// returns: Result<(), PresetSerializeError>
    pub async fn save_into_bank(
        &self,
        bank: &str,
        preset: PresetV1<Data>,
    ) -> Result<(), PresetSerializeError> {
        let bank = match self.bank(bank) {
            None => {
                let bank_dir = self.user_parent_dir.join(bank);
                fs::create_dir(&bank_dir).await?;
                self.user_group.add_bank(&bank_dir)
            }
            Some(bank) => bank.clone(),
        };
        bank.save_preset(preset).await?;
        Ok(())
    }

    /// Load preset at the offset of the current one.
    ///
    /// # Arguments
    ///
    /// * `current_preset_name`: Name of the current preset
    /// * `offset`: Offset to navigate by
    ///
    /// returns: Result<Option<PresetV1<Data>>, PresetDeserializeError>
    pub async fn load_with_offset(
        &self,
        current_preset_name: &str,
        offset: isize,
    ) -> Result<Option<PresetV1<Data>>, PresetDeserializeError> {
        let flat_map = stream::iter(self.banks()).flat_map(|bank| {
            stream::once_future(async move {
                let bank = self.bank(&bank)?;
                let new_preset = bank.navigate_by_offset(current_preset_name, offset).await?;
                Some((bank.clone(), new_preset))
            })
        });
        pin!(flat_map);
        let Some((bank, preset_name)) = flat_map.next().await.flatten() else {
            return Ok(None);
        };
        match bank.load_preset(&preset_name).await {
            Ok(preset) => Ok(Some(preset)),
            Err(PresetDeserializeError::IoError(err))
                if err.kind() == std::io::ErrorKind::NotFound =>
            {
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    /// Load the preset preceeding the current one.
    ///
    /// # Arguments
    ///
    /// * `current_preset_name`: Name of the current preset
    ///
    /// returns: Result<Option<PresetV1<Data>>, PresetDeserializeError>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    #[inline]
    pub async fn load_previous_preset(
        &self,
        current_preset_name: &str,
    ) -> Result<Option<PresetV1<Data>>, PresetDeserializeError> {
        self.load_with_offset(current_preset_name, -1).await
    }

    /// Load the preset preceeding the current one.
    ///
    /// # Arguments
    ///
    /// * `current_preset_name`: Name of the current preset
    ///
    /// returns: Result<Option<PresetV1<Data>>, PresetDeserializeError>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    #[inline]
    pub async fn load_next_preset(
        &self,
        current_preset_name: &str,
    ) -> Result<Option<PresetV1<Data>>, PresetDeserializeError> {
        self.load_with_offset(current_preset_name, 1).await
    }
}
