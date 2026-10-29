pub mod bert;
pub mod classifier;
pub mod memory;
pub mod value;
use crate::Parameter;
use sejong_core::{Result, ensure, json};
use sejong_gpu::Engine;
use std::path::Path;

pub fn signature(
    engine: &mut Engine,
    parameters: &[Parameter],
    metadata: &impl serde::Serialize,
) -> Result<String> {
    let mut h = blake3::Hasher::new();
    h.update(&json::to_vec(metadata)?);
    for p in parameters {
        h.update(p.name.as_bytes());
        h.update(&json::to_vec(&p.value.shape)?);
        h.update(&json::to_vec(&p.value.dtype)?);
        for v in engine.to_f64(&p.value)? {
            h.update(&v.to_le_bytes());
        }
    }
    Ok(h.finalize().to_hex().to_string())
}
pub fn atomic_directory(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    ensure(
        !path.exists(),
        format!("output already exists: {}", path.display()),
    )?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".{}.{}.staging",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    std::fs::create_dir(&staging)?;
    let result = write(&staging).and_then(|()| {
        std::fs::rename(&staging, path)?;
        Ok(())
    });
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}
