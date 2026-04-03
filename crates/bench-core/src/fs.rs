use std::path::Path;

use anyhow::Result;
use walkdir::WalkDir;

pub fn dir_size_bytes(path: &Path) -> Result<u64> {
    if path.is_file() {
        return Ok(std::fs::metadata(path)?.len());
    }
    let mut total = 0u64;
    for entry in WalkDir::new(path).follow_links(false) {
        let entry = entry?;
        if entry.file_type().is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}
