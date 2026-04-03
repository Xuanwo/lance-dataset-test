use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::metrics::{LatencySummary, Timing};
use crate::workload::{DatasetName, EngineName, Workload};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub bench_version: String,
    pub git_commit: String,
    pub os: String,
    pub arch: String,
    pub cpu_count: u64,
    pub engine_version: String,
    pub engine: EngineName,
    pub dataset: DatasetName,
    pub workload: Workload,
    pub seed: u64,
    pub started_at_unix_ms: u128,
    pub rustc: String,
    pub params: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub meta: RunMetadata,
    pub timing: Timing,
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
    pub latency: Option<LatencySummary>,
    pub notes: Vec<String>,
}
