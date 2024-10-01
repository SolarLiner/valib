//! # Preset data module
//!
//! Traits and types for a single preset.
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::{fs, io};
use std::any;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// Extension used by preset files
pub const EXTENSION: &str = "preset";

/// Type alias for a raw preset, which can be used in lieu of an actual preset with data type.
pub type RawPresetV1 = PresetV1<toml::Value>;

/// Trait for types which are preset data.
///
/// This is needed to provide an upgrade path from lower revisions; when failing to load with this
/// type, the loader will recursively try with the [`PresetData::PreviousRevision`] until it either
/// succeeds, or fails deserializing a preset data type with [`PresetData::CURRENT_REVISION`] set to
/// 0. **Note**: the [`PresetData::CURRENT_REVISION`] is checked by first loading the preset
/// container without loading the data, and checking that the revision matches. Revision mismatch is
/// also considered a failure.
///
/// If it succeeds, it will then walk the chain back up, incrementally upgrading every `PresetData`
/// until we arrive back at this type, or any of the [`PresetData::upgrade`] returns None.
#[allow(unused_variables)]
pub trait PresetData: 'static + DeserializeOwned + Serialize {
    /// Current revision of this `PresetData` type. Should indicate 0 when there are no previous
    /// versions available.
    const CURRENT_REVISION: u64;

    /// Previous revision type of this `PresetData`. While not checked, that type should have its
    /// current revision set to the preceeding number from this data's revision number.
    type PreviousRevision: PresetData;

    /// Upgrade the preset from a previous version to this one. If the conversion is not possible,
    /// this method can return `None`.
    ///
    /// # Arguments
    ///
    /// * `previous`: Previous preset to upgrade from
    ///
    /// returns: Option<PresetV1<Self>>
    fn upgrade(previous: &PresetV1<Self::PreviousRevision>) -> Option<PresetV1<Self>> {
        None
    }
}

impl PresetData for () {
    const CURRENT_REVISION: u64 = 0;
    type PreviousRevision = ();

    fn upgrade(previous: &PresetV1<Self>) -> Option<PresetV1<Self>> {
        Some(previous.clone())
    }
}

impl PresetData for toml::Value {
    const CURRENT_REVISION: u64 = 0;
    type PreviousRevision = toml::Value;

    fn upgrade(previous: &PresetV1<Self::PreviousRevision>) -> Option<PresetV1<Self>> {
        Some(previous.clone())
    }
}

impl<T: PresetData> PresetData for Arc<T>
where
    Arc<T>: Serialize + DeserializeOwned,
{
    const CURRENT_REVISION: u64 = 0;
    type PreviousRevision = T::PreviousRevision;

    fn upgrade(previous: &PresetV1<Self::PreviousRevision>) -> Option<PresetV1<Self>> {
        let raw = T::upgrade(previous)?;
        Some(raw.map(Arc::new))
    }
}

/// Preset metadata type. Contains data related to the preset itself, and contains both hardcoded
/// fields as well as [`Self::other`] which can be used to store arbitrary data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetMeta {
    /// Preset title
    pub title: String,
    /// Preset author
    pub author: String,
    /// Preset tags
    pub tags: BTreeSet<String>,
    /// Preset revision. This is incremented on each save.
    pub revision: u64,
    /// Other metadata. This field is free to be used (or ignored) by downstream users.
    #[serde(flatten)]
    pub other: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(try_from = "u64")]
struct PresetV1Version;

impl TryFrom<u64> for PresetV1Version {
    type Error = String;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value == 1 {
            Ok(Self)
        } else {
            Err(format!(
                "Presets from PresetV1 can only have version = 1 (not {value})"
            ))
        }
    }
}

/// Preset type. This is the type that is serialized into the preset files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetV1<Data> {
    version: PresetV1Version,
    data_revision: u64,
    /// Preset metadata
    pub metadata: PresetMeta,
    /// Preset data. This contains all your parameter data.
    pub data: Data,
}

impl<Data> PresetV1<Data> {
    /// Map the inner data into a new type.
    ///
    /// The output type needs to have [`PresetData`] implemented so that the data revision can be
    /// updated accordingly.
    ///
    /// # Arguments
    ///
    /// * `func`: Mapping function
    ///
    /// returns: PresetV1<D2>
    pub fn map<D2: PresetData>(self, func: impl FnOnce(Data) -> D2) -> PresetV1<D2> {
        let Self {
            version,
            data,
            metadata,
            ..
        } = self;
        PresetV1 {
            version,
            data_revision: D2::CURRENT_REVISION,
            data: func(data),
            metadata,
        }
    }

    /// Drop the preset data, returning the unwrapped data, and the empty preset container.
    pub fn drop_data(self) -> (PresetV1<()>, Data) {
        let Self {
            version,
            metadata,
            data,
            ..
        } = self;
        let preset = PresetV1 {
            version,
            data_revision: <() as PresetData>::CURRENT_REVISION,
            data: (),
            metadata,
        };
        (preset, data)
    }
}

impl<Data> PresetV1<Option<Data>> {
    /// Transposes the optional data into an optional preset.
    pub fn transpose(self) -> Option<PresetV1<Data>> {
        let Self {
            version,
            data_revision,
            metadata,
            data,
        } = self;
        data.map(|data| PresetV1 {
            version,
            data_revision,
            metadata,
            data,
        })
    }
}

impl<Data, E> PresetV1<Result<Data, E>> {
    /// Transposes the data result into a preset result.
    pub fn transpose(self) -> Result<PresetV1<Data>, E> {
        let Self {
            version,
            data_revision,
            metadata,
            data,
        } = self;
        data.map(|data| PresetV1 {
            version,
            data_revision,
            metadata,
            data,
        })
    }
}

impl<Data: PresetData> PresetV1<Data> {
    /// Creates a new preset.
    ///
    /// # Arguments
    ///
    /// * `metadata`: Preset metadata
    /// * `data`: Preset data
    ///
    /// returns: PresetV1<Data>
    pub fn new(metadata: PresetMeta, data: Data) -> Self {
        Self {
            version: PresetV1Version,
            data_revision: Data::CURRENT_REVISION,
            metadata,
            data,
        }
    }

    /// Transform the preset data into generic toml::Value
    pub fn into_value(self) -> Result<PresetV1<toml::Value>, toml::ser::Error> {
        let Self {
            version,
            data_revision,
            metadata,
            data,
        } = self;
        Ok(PresetV1 {
            version,
            data_revision,
            metadata,
            data: toml::Value::try_from(data)?,
        })
    }
}

/// Type of errors that can occur when deserializing (loading) a preset.
#[derive(Debug, Error)]
pub enum PresetDeserializeError {
    /// Forwarded I/O error
    #[error("Preset I/O Error: {0}")]
    IoError(#[from] io::Error),
    /// Forwarded TOML parse error
    #[error("TOML Parse Error: {0}")]
    TomlError(#[from] toml::de::Error),
    /// Preset was loaded with a type that doesn't match in its revision
    #[error("Cannot load preset: Data revision mismatch: expected {expected} but got {found}")]
    RevisionMismatch {
        /// Expected revision number
        expected: u64,
        /// Found revision number
        found: u64,
    },
    /// Preset was loaded successfully in a previous data type, but could not be upgraded
    #[error("Could not upgrade preset from {previous_data_type}: {source}")]
    UpgradeFailedConversion {
        /// Error that lead to trying to upgrade the preset
        #[source]
        source: Box<Self>,
        /// Name of the previous revision's type
        previous_data_type: &'static str,
    },
}

/// Type of errors that can occur when serializing (saving) a preset.
#[derive(Debug, Error)]
pub enum PresetSerializeError {
    /// Forwarded I/O error
    #[error("Preset I/O Error: {0}")]
    IoError(#[from] io::Error),
    /// Forwarded TOML parse error
    #[error("TOML Parse Error: {0}")]
    TomlError(#[from] toml::ser::Error),
}

impl<Data: PresetData> PresetV1<Data> {
    /// Deserializes a preset from the provided async reader.
    ///
    /// **Warning**: This method does not perform upgrades. You should probably use
    /// [`Self::from_string`] instead.
    ///
    /// # Arguments
    ///
    /// * `reader`: Async reader to read data from.
    ///
    /// returns: Result<PresetV1<Data>, PresetDeserializeError>
    pub fn from_string_no_upgrade(data: &str) -> Result<Self, PresetDeserializeError> {
        // 1. Checks for preset version and data revision matching
        let preset_raw: RawPresetV1 = toml::from_str(data)?;
        if preset_raw.data_revision != Data::CURRENT_REVISION {
            return Err(PresetDeserializeError::RevisionMismatch {
                expected: Data::CURRENT_REVISION,
                found: preset_raw.data_revision,
            });
        }

        // 2. Deserialize preset data type
        let PresetV1 {
            version,
            data_revision,
            metadata,
            data,
        } = preset_raw;
        let data = Data::deserialize(data)?;
        Ok(Self {
            version,
            data_revision,
            metadata,
            data,
        })
    }

    /// Deserialized a preset from a TOML string. Recursively tries to perform conversions.
    ///
    /// # Arguments
    ///
    /// * `data`:
    ///
    /// returns: Result<PresetV1<Data>, PresetDeserializeError>
    pub fn from_string(data: &str) -> Result<Self, PresetDeserializeError> {
        match Self::from_string_no_upgrade(data) {
            Ok(data) => Ok(data),
            Err(err) => {
                if Data::CURRENT_REVISION == 0 {
                    return Err(err);
                }
                let previous: PresetV1<Data::PreviousRevision> = PresetV1::from_string(data)?;
                match Data::upgrade(&previous) {
                    Some(previous) => Ok(previous),
                    None => Err(PresetDeserializeError::UpgradeFailedConversion {
                        previous_data_type: any::type_name::<Data::PreviousRevision>(),
                        source: Box::new(err),
                    }),
                }
            }
        }
    }

    /// Deserializes a preset from the given async reader.
    ///
    /// # Arguments
    ///
    /// * `reader`: Async reader containing the preset file contents.
    ///
    /// returns: Result<PresetV1<Data>, PresetDeserializeError>
    pub async fn from_reader(
        mut reader: impl io::AsyncBufRead + Unpin,
    ) -> Result<Self, PresetDeserializeError> {
        let data = {
            let mut s = String::new();
            reader.read_to_string(&mut s).await?;
            s
        };
        Self::from_string(&data)
    }

    /// Deserialize a preset from a file at the given path.
    ///
    /// # Arguments
    ///
    /// * `path`: Path to the preset file
    ///
    /// returns: Result<PresetV1<Data>, PresetDeserializeError>
    pub async fn from_file(path: &Path) -> Result<Self, PresetDeserializeError> {
        let file = fs::File::open(path).await?;
        let file = io::BufReader::new(file);
        Self::from_reader(file).await
    }

    /// Read a preset file from the given preset title. This assumes that the preset filename is the
    /// same as the preset title, and that the extensions is the same as the one defined by
    /// [`EXTENSION`].
    ///
    /// # Arguments
    ///
    /// * `base_dir`: Directory where the preset file is
    /// * `title`: Preset title
    ///
    /// returns: Result<PresetV1<Data>, PresetDeserializeError>
    pub async fn from_title(base_dir: &Path, title: &str) -> Result<Self, PresetDeserializeError> {
        Self::from_file(&base_dir.join(Self::get_preset_filename(title))).await
    }

    /// Serialize the preset to the given async writer.
    ///
    /// # Arguments
    ///
    /// * `writer`: Writer to serialize into
    /// * `pretty`: Whether to use TOML's pretty formatting, or the condensed formatting methods.
    ///
    /// returns: Result<(), PresetSerializeError>
    pub async fn write_to(
        &self,
        mut writer: impl io::AsyncWrite + Unpin,
        pretty: bool,
    ) -> Result<(), PresetSerializeError> {
        let data = if pretty {
            toml::to_string_pretty(self)
        } else {
            toml::to_string(self)
        }?;
        writer.write_all(data.as_bytes()).await?;
        Ok(())
    }

    /// Serializes this preset to the given file.
    ///
    /// # Arguments
    ///
    /// * `path`: Path to the preset file
    /// * `pretty`: Whether to use TOML's pretty formatting, or the condensed formatting methods.
    ///
    /// returns: Result<(), PresetSerializeError>
    pub async fn write_to_file(
        &self,
        path: &Path,
        pretty: bool,
    ) -> Result<(), PresetSerializeError> {
        let file = fs::File::create(path).await?;
        let file = io::BufWriter::new(file);
        self.write_to(file, pretty).await
    }

    /// Serialize this preset to bank at the given path.
    ///
    /// # Arguments
    ///
    /// * `bank_dir`: Path to the bank on disk
    /// * `pretty`: Whether to use TOML's pretty formatting, or the condensed formatting methods.
    ///
    /// returns: Result<(), PresetSerializeError>
    pub async fn write_to_bank(
        &self,
        bank_dir: &Path,
        pretty: bool,
    ) -> Result<(), PresetSerializeError> {
        self.write_to_file(&bank_dir.join(self.preset_filename()), pretty)
            .await
    }

    /// Get the preset filename, given the title.
    ///
    /// # Arguments
    ///
    /// * `title`: Preset title
    ///
    /// returns: PathBuf
    pub fn get_preset_filename(title: &str) -> PathBuf {
        Path::new(title).with_extension(EXTENSION)
    }

    /// Get the preset filename corresponding to this preset, following standard preset naming
    /// conventions.
    pub fn preset_filename(&self) -> PathBuf {
        Self::get_preset_filename(&self.metadata.title)
    }
}
