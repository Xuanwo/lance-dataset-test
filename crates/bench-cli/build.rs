use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str;

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let git_commit = compute_git_commit(&manifest_dir).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BENCH_GIT_COMMIT={git_commit}");

    let rustc_version = compute_rustc_version().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BENCH_RUSTC_VERSION={rustc_version}");
}

fn compute_rustc_version() -> Option<String> {
    let rustc = env::var("RUSTC")
        .ok()
        .unwrap_or_else(|| "rustc".to_string());
    let output = Command::new(rustc).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = str::from_utf8(&output.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn compute_git_commit(start_dir: &Path) -> Option<String> {
    let git_dir = find_git_dir(start_dir)?;
    let head_path = git_dir.join("HEAD");
    let head = fs::read_to_string(&head_path).ok()?;
    let head = head.trim();
    if let Some(ref_name) = head.strip_prefix("ref: ").map(str::trim) {
        if let Some(commit) = read_ref_commit(&git_dir, ref_name) {
            return Some(commit);
        }
        return read_packed_refs_commit(&git_dir, ref_name);
    }
    if head.is_empty() {
        None
    } else {
        Some(head.to_string())
    }
}

fn find_git_dir(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = Some(start_dir);
    while let Some(current) = dir {
        let dot_git = current.join(".git");
        if dot_git.is_dir() {
            return Some(dot_git);
        }
        if dot_git.is_file() {
            let content = fs::read_to_string(&dot_git).ok()?;
            let content = content.trim();
            let gitdir = content.strip_prefix("gitdir:")?.trim();
            let gitdir = PathBuf::from(gitdir);
            return Some(if gitdir.is_relative() {
                current.join(gitdir)
            } else {
                gitdir
            });
        }
        dir = current.parent();
    }
    None
}

fn read_ref_commit(git_dir: &Path, ref_name: &str) -> Option<String> {
    let p = git_dir.join(ref_name);
    let s = fs::read_to_string(p).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn read_packed_refs_commit(git_dir: &Path, ref_name: &str) -> Option<String> {
    let p = git_dir.join("packed-refs");
    let content = fs::read_to_string(p).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
            continue;
        }
        let mut it = line.split_whitespace();
        let commit = it.next()?;
        let name = it.next()?;
        if name == ref_name {
            return Some(commit.to_string());
        }
    }
    None
}
