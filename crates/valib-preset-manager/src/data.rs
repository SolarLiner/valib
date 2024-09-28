use futures::TryStreamExt;
use serde::de::{DeserializeOwned, IntoDeserializer};
use serde::{Deserialize, Serialize};
use smol::io::{AsyncReadExt, AsyncWriteExt};
use smol::stream::{Stream, StreamExt};
use smol::{fs, io, stream};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const EXTENSION: &'static str = "preset";

#[allow(unused_variables)]
pub trait PresetData: DeserializeOwned + Serialize {
    const CURRENT_REVISION: u64;

    type PreviousRevision: PresetData;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetMeta {
    pub title: String,
    pub author: String,
    pub tags: BTreeSet<String>,
    pub revision: u64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetV1<Data> {
    version: PresetV1Version,
    pub metadata: PresetMeta,
    pub data: Data,
}

impl<Data> PresetV1<Data> {
    pub fn new(metadata: PresetMeta, data: Data) -> Self {
        Self {
            version: PresetV1Version,
            metadata,
            data,
        }
    }
}

#[derive(Debug, Error)]
pub enum PresetDeserializeError {
    #[error("Preset I/O Error: {0}")]
    IoError(#[from] io::Error),
    #[error("TOML Parse Error: {0}")]
    TomlError(#[from] toml::de::Error),
}

#[derive(Debug, Error)]
pub enum PresetSerializeError {
    #[error("Preset I/O Error: {0}")]
    IoError(#[from] io::Error),
    #[error("TOML Parse Error: {0}")]
    TomlError(#[from] toml::ser::Error),
}

impl<Data: PresetData> PresetV1<Data> {
    pub async fn from_reader(
        mut reader: impl io::AsyncBufRead + Unpin,
    ) -> Result<Self, PresetDeserializeError> {
        let data = {
            let mut s = String::new();
            reader.read_to_string(&mut s).await?;
            s
        };
        Ok(toml::from_str(&data)?)
    }

    pub async fn from_file(path: &Path) -> Result<Self, PresetDeserializeError> {
        let file = fs::File::open(path).await?;
        let file = io::BufReader::new(file);
        Self::from_reader(file).await
    }

    pub async fn from_title(base_dir: &Path, title: &str) -> Result<Self, PresetDeserializeError> {
        Self::from_file(&base_dir.join(Self::get_preset_filename(title))).await
    }

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

    pub async fn write_to_file(
        &self,
        path: &Path,
        pretty: bool,
    ) -> Result<(), PresetSerializeError> {
        let file = fs::File::create(path).await?;
        let file = io::BufWriter::new(file);
        self.write_to(file, pretty).await
    }

    pub async fn write_to_bank(
        &self,
        base_dir: &Path,
        pretty: bool,
    ) -> Result<(), PresetSerializeError> {
        self.write_to_file(&base_dir.join(self.preset_filename()), pretty)
            .await
    }

    pub fn get_preset_filename(title: &str) -> PathBuf {
        Path::new(title).with_extension(EXTENSION)
    }

    pub fn preset_filename(&self) -> PathBuf {
        Self::get_preset_filename(&self.metadata.title)
    }
}

pub fn read_folder<Data: PresetData>(
    dir: &Path,
) -> impl Stream<Item = Result<PresetV1<Data>, PresetDeserializeError>> + '_ {
    stream::once_future(fs::read_dir(dir))
        .try_flatten_unordered(None)
        .try_filter_map(|entry| async move {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                Ok(Some(entry.path()))
            } else {
                Ok(None)
            }
        })
        .map(|res| res.map_err(PresetDeserializeError::IoError))
        .try_filter_map(|path| async move {
            let preset = PresetV1::from_file(&path).await?;
            Ok(Some(preset))
        })
}
