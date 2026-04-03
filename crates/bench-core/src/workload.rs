use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineName {
    Unknown,
    Lance,
    LanceFragment,
    Parquet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetName {
    Unknown,
    Laion10m,
    OpenVid,
    FineWeb,
    LeRobotPushT,
    LeRobotPushTImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    Ingest,
    ScanFull,
    ScanProject,
    ScanFilterLow,
    ScanFilterHigh,
    RandomTake,
    RandomBlob,
    EvolutionBackfill,
    Size,
}
