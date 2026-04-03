use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    LtU64 { column: String, value: u64 },
}

impl Filter {
    pub fn lt_u64(column: impl Into<String>, value: u64) -> Self {
        Self::LtU64 {
            column: column.into(),
            value,
        }
    }
}
