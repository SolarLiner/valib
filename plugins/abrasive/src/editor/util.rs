use directories::ProjectDirs;
use nih_plug::nih_log;
use std::env;
use std::path::{Path, PathBuf};

pub fn resolve_asset_file(file: &Path) -> PathBuf {
    search_dirs()
        .map(|p| p.join(file))
        .inspect(|p| nih_log!("Getting asset: trying {}", p.display()))
        .find(|p| p.is_file())
        .expect(&format!("Cannot find assets directory {}", file.display()))
}

pub fn resolve_asset_dir(dir: &Path) -> PathBuf {
    search_dirs()
        .map(|p| p.join(dir))
        .inspect(|p| nih_log!("Getting assets directory: trying {}", p.display()))
        .find(|p| p.is_dir())
        .expect(&format!("Cannot find assets directory {}", dir.display()))
}

fn search_dirs() -> impl Iterator<Item = PathBuf> {
    [
        installed_assets_dir(),
        dev_workspace_dir(),
        plugin_package_workspace_dir(),
        current_dir(),
        current_exe(),
    ]
    .into_iter()
    .flatten()
}

fn installed_assets_dir() -> Option<PathBuf> {
    let dirs = ProjectDirs::from("dev.solarliner", "valib", env!("CARGO_PKG_NAME"))?;
    Some(dirs.data_dir().join("assets"))
}

fn dev_workspace() -> Option<PathBuf> {
    if cfg!(feature = "distribution") {
        None
    } else {
        // Plugin project is at plugins/abrasive, so we need to climb up twice to get the workspace
        Some(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()?
                .parent()?
                .to_path_buf(),
        )
    }
}

fn dev_workspace_dir() -> Option<PathBuf> {
    dev_workspace().map(|p| p.join("assets"))
}

fn plugin_package_workspace_dir() -> Option<PathBuf> {
    dev_workspace().map(|dir| {
        dir.join("plugins")
            .join(env!("CARGO_PKG_NAME"))
            .join("assets")
    })
}

fn current_dir() -> Option<PathBuf> {
    env::current_dir().ok().map(|p| p.join("assets"))
}

fn current_exe() -> Option<PathBuf> {
    env::current_exe()
        .ok()
        .and_then(|p| Some(p.canonicalize().ok()?.parent()?.join("assets")))
}
