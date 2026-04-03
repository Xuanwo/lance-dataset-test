use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timing {
    #[serde(default)]
    pub wall_time_us: u128,
    pub wall_time_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_wall_time_us: Option<Vec<u128>>,
}

impl Timing {
    pub fn from_duration(duration: Duration) -> Self {
        Self {
            wall_time_us: duration.as_micros(),
            wall_time_ms: duration.as_millis(),
            repeat_wall_time_us: None,
        }
    }
}

pub struct WallTimer {
    start: Instant,
}

impl WallTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn stop(self) -> Timing {
        Timing::from_duration(self.start.elapsed())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

impl LatencySummary {
    pub fn from_histogram(histogram: &Histogram<u64>) -> Self {
        Self {
            p50_us: histogram.value_at_quantile(0.50),
            p95_us: histogram.value_at_quantile(0.95),
            p99_us: histogram.value_at_quantile(0.99),
        }
    }
}
