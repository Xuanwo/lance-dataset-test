use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;
use lance::dataset::fragment::FileFragment;
use lance::dataset::NewColumnTransform;
pub use lance::dataset::ProjectionRequest;
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
pub use lance::Dataset;
use lance_datafusion::exec::{new_session_context, LanceExecutionOptions};
use lance_file::version::LanceFileVersion;
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;

use arrow_array::RecordBatchReader;
use bench_core::metrics::{LatencySummary, Timing, WallTimer};
use bench_core::query::Filter;
use hdrhistogram::Histogram;

pub const ENGINE_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const LANCE_DEP_VERSION: &str = "09174bc";

pub struct LanceEngine;

mod binary_compression_none_input;
mod blob_input;

const FILE_SCAN_BATCH_SIZE: u32 = 8192;

fn bench_target_partition() -> Option<usize> {
    match std::env::var("BENCH_LANCE_TARGET_PARTITION") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(0) => None,
            Ok(v) => Some(v),
            Err(_) => Some(1),
        },
        Err(_) => Some(1),
    }
}

#[derive(Debug, Clone, Default)]
pub struct IngestOptions {
    pub max_rows_per_file: Option<usize>,
    pub max_bytes_per_file: Option<usize>,
    pub data_storage_version: Option<String>,
    pub index_columns: Vec<String>,
    pub write_mode: Option<String>,
}

impl LanceEngine {
    pub fn new() -> Self {
        Self
    }

    async fn ensure_scalar_btree_indices(
        &self,
        dataset: &mut Dataset,
        columns: &[String],
    ) -> Result<()> {
        if columns.is_empty() {
            return Ok(());
        }

        let field_names: Vec<String> = {
            let schema = dataset.schema();
            schema.fields.iter().map(|f| f.name.clone()).collect()
        };
        let existing = dataset.load_indices().await?;

        for column in columns {
            let exists = field_names.iter().any(|name| name == column);
            if !exists {
                anyhow::bail!("index column not found in schema: {column}");
            }

            let index_name = format!("{column}_btree");
            if existing.iter().any(|idx| idx.name == index_name) {
                continue;
            }

            let params = ScalarIndexParams::default();
            dataset
                .create_index(
                    &[column.as_str()],
                    IndexType::BTree,
                    Some(index_name),
                    &params,
                    true,
                )
                .await?;
        }

        Ok(())
    }

    pub async fn open_dataset(&self, dataset_path: &Path) -> Result<Arc<Dataset>> {
        let uri = dataset_path.to_string_lossy().to_string();
        Ok(Arc::new(Dataset::open(&uri).await?))
    }

    pub async fn schema_field_names(&self, dataset_path: &Path) -> Result<Vec<String>> {
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Dataset::open(&uri).await?;
        Ok(dataset
            .schema()
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect())
    }

    pub async fn smoke_check(&self) -> Result<()> {
        Ok(())
    }

    pub async fn ingest(
        &self,
        reader: Box<dyn RecordBatchReader + Send>,
        out: &Path,
        options: IngestOptions,
    ) -> Result<()> {
        tokio::fs::create_dir_all(out).await?;
        let uri = out.to_string_lossy().to_string();
        let mut params = WriteParams::default();
        if let Some(v) = options.max_rows_per_file {
            params.max_rows_per_file = v;
        }
        if let Some(v) = options.max_bytes_per_file {
            params.max_bytes_per_file = v;
        }
        if let Some(v) = options.write_mode.as_deref() {
            params.mode = v.try_into()?;
        }
        let storage_version = options
            .data_storage_version
            .as_deref()
            .map(|v| v.parse::<LanceFileVersion>())
            .transpose()?
            .unwrap_or_default();
        if options.data_storage_version.is_some() {
            params.data_storage_version = Some(storage_version);
        }

        let reader = binary_compression_none_input::wrap_binary_compression_none_input(reader)?;
        let reader = blob_input::maybe_wrap_blob_v2_input(reader, storage_version)?;

        let _dataset = if matches!(params.mode, WriteMode::Append) {
            let mut dataset = Dataset::open(&uri).await?;
            dataset.append(reader, Some(params)).await?;
            dataset
        } else {
            Dataset::write(reader, &uri, Some(params)).await?
        };
        if !options.index_columns.is_empty() {
            let mut dataset = Dataset::open(&uri).await?;
            self.ensure_scalar_btree_indices(&mut dataset, &options.index_columns)
                .await?;
        }

        Ok(())
    }

    pub async fn create_scalar_btree_indices(
        &self,
        dataset_path: &Path,
        columns: &[String],
    ) -> Result<()> {
        let uri = dataset_path.to_string_lossy().to_string();
        let mut dataset = Dataset::open(&uri).await?;
        self.ensure_scalar_btree_indices(&mut dataset, columns)
            .await
    }

    pub async fn fragment_count(&self, dataset_path: &Path) -> Result<u64> {
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Dataset::open(&uri).await?;
        Ok(dataset.fragments().len() as u64)
    }

    pub async fn data_storage_version(&self, dataset_path: &Path) -> Result<String> {
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Dataset::open(&uri).await?;
        Ok(dataset
            .manifest
            .data_storage_format
            .lance_file_version()?
            .to_string())
    }

    pub fn data_file_count(&self, dataset_path: &Path) -> Result<u64> {
        let data_dir = dataset_path.join("data");
        let mut count = 0u64;
        for entry in std::fs::read_dir(&data_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                count += 1;
            }
        }
        Ok(count)
    }

    pub async fn scan_count_rows(
        &self,
        dataset_path: &Path,
        projection: Option<&[String]>,
        filter: Option<&Filter>,
        limit_rows: Option<u64>,
        row_offset: Option<u64>,
        repeats: u32,
    ) -> Result<(u64, u64)> {
        let repeats = repeats.max(1);
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Arc::new(Dataset::open(&uri).await?);

        let exec_options = LanceExecutionOptions {
            use_spilling: false,
            mem_pool_size: None,
            batch_size: Some(FILE_SCAN_BATCH_SIZE as usize),
            target_partition: bench_target_partition(),
            execution_stats_callback: None,
            max_temp_directory_size: None,
            skip_logging: true,
        };
        let session_ctx = new_session_context(&exec_options);

        let mut scanner = dataset.scan();
        scanner.batch_size(FILE_SCAN_BATCH_SIZE as usize);
        if let Some(filter) = filter {
            match filter {
                Filter::LtU64 { column, value } => {
                    scanner.filter(&format!("{column} < {value}"))?;
                }
            }
        }
        if let Some(cols) = projection {
            scanner.project(cols)?;
        }
        if limit_rows.is_some() || row_offset.is_some() {
            scanner.limit(
                limit_rows.map(|v| v.min(i64::MAX as u64) as i64),
                row_offset.map(|v| v.min(i64::MAX as u64) as i64),
            )?;
        }

        let mut rows = 0u64;
        let mut bytes = 0u64;
        for _ in 0..repeats {
            // NOTE: Lance's DataFusion execution plan is not safely re-executable (subsequent
            // executions can yield an empty stream). To preserve the intended `--repeats N`
            // semantics ("run the same scan N times"), rebuild the plan each time while reusing
            // the already-opened dataset and configured scanner.
            let plan = scanner.create_plan().await?;
            let partitions = plan.properties().partitioning.partition_count();
            let task_ctx = session_ctx.task_ctx();
            for partition in 0..partitions {
                let mut stream = plan.execute(partition, task_ctx.clone())?;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    rows += batch.num_rows() as u64;
                    bytes = bytes.saturating_add(batch.get_array_memory_size() as u64);
                }
            }
        }
        Ok((rows, bytes))
    }

    pub async fn scan_count_rows_measured(
        &self,
        dataset_path: &Path,
        projection: Option<&[String]>,
        filter: Option<&Filter>,
        limit_rows: Option<u64>,
        row_offset: Option<u64>,
        repeats: u32,
    ) -> Result<(Timing, LatencySummary, u64, u64)> {
        let repeats = repeats.max(1);
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Arc::new(Dataset::open(&uri).await?);

        let exec_options = LanceExecutionOptions {
            use_spilling: false,
            mem_pool_size: None,
            batch_size: Some(FILE_SCAN_BATCH_SIZE as usize),
            target_partition: bench_target_partition(),
            execution_stats_callback: None,
            max_temp_directory_size: None,
            skip_logging: true,
        };
        let session_ctx = new_session_context(&exec_options);
        let task_ctx = session_ctx.task_ctx();

        let mut scanner = dataset.scan();
        scanner.batch_size(FILE_SCAN_BATCH_SIZE as usize);
        if let Some(filter) = filter {
            match filter {
                Filter::LtU64 { column, value } => {
                    scanner.filter(&format!("{column} < {value}"))?;
                }
            }
        }
        if let Some(cols) = projection {
            scanner.project(cols)?;
        }
        if limit_rows.is_some() || row_offset.is_some() {
            scanner.limit(
                limit_rows.map(|v| v.min(i64::MAX as u64) as i64),
                row_offset.map(|v| v.min(i64::MAX as u64) as i64),
            )?;
        }

        let execute_plan = |plan: Arc<dyn ExecutionPlan>| {
            let task_ctx = task_ctx.clone();
            let mut rows = 0u64;
            let mut bytes = 0u64;
            async move {
                let partitions = plan.properties().partitioning.partition_count();
                for partition in 0..partitions {
                    let mut stream = plan.execute(partition, task_ctx.clone())?;
                    while let Some(batch) = stream.next().await {
                        let batch = batch?;
                        rows += batch.num_rows() as u64;
                        bytes = bytes.saturating_add(batch.get_array_memory_size() as u64);
                    }
                }
                Ok::<(u64, u64), anyhow::Error>((rows, bytes))
            }
        };

        // Warm up OS page cache and engine-internal caches with the same query shape.
        let warmup_plan = scanner.create_plan().await?;
        let _ = execute_plan(warmup_plan).await?;

        let mut histogram = Histogram::<u64>::new(3)?;
        let mut repeat_wall_time_us = Vec::with_capacity(repeats as usize);
        let timer = WallTimer::start();
        let mut rows = 0u64;
        let mut bytes = 0u64;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            let plan = scanner.create_plan().await?;
            let (r, b) = execute_plan(plan).await?;
            let elapsed_us_u128 = start.elapsed().as_micros();
            let elapsed_us_u64 = elapsed_us_u128.min(u128::from(u64::MAX)) as u64;
            histogram.record(elapsed_us_u64)?;
            repeat_wall_time_us.push(elapsed_us_u128);
            rows += r;
            bytes += b;
        }
        let mut timing = timer.stop();
        timing.repeat_wall_time_us = Some(repeat_wall_time_us);
        Ok((
            timing,
            LatencySummary::from_histogram(&histogram),
            rows,
            bytes,
        ))
    }

    pub async fn scan_count_rows_fragments(
        &self,
        dataset_path: &Path,
        projection: Option<&[String]>,
        filter: Option<&Filter>,
        repeats: u32,
    ) -> Result<(u64, u64)> {
        let repeats = repeats.max(1);
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Arc::new(Dataset::open(&uri).await?);

        let exec_options = LanceExecutionOptions {
            use_spilling: false,
            mem_pool_size: None,
            batch_size: Some(FILE_SCAN_BATCH_SIZE as usize),
            target_partition: bench_target_partition(),
            execution_stats_callback: None,
            max_temp_directory_size: None,
            skip_logging: true,
        };
        let session_ctx = new_session_context(&exec_options);
        let task_ctx = session_ctx.task_ctx();

        let fragments = dataset.fragments().as_ref();
        if fragments.is_empty() {
            return Ok((0, 0));
        }

        let mut fragment_indices: Vec<usize> = (0..fragments.len()).collect();
        if !fragments.windows(2).all(|w| w[0].id <= w[1].id) {
            fragment_indices.sort_by_key(|&idx| fragments[idx].id);
        }

        if let (Some(filter), Some(projection)) = (filter, projection) {
            let Filter::LtU64 { column, .. } = filter;
            if !projection.iter().any(|c| c == column) {
                anyhow::bail!("projection must include filter column {column}");
            }
        }

        let mut scanners = Vec::with_capacity(fragment_indices.len());
        for idx in fragment_indices {
            let fragment = fragments[idx].clone();
            let fragment = FileFragment::new(dataset.clone(), fragment);

            let mut scanner = fragment.scan();
            scanner.batch_size(FILE_SCAN_BATCH_SIZE as usize);
            if let Some(filter) = filter {
                match filter {
                    Filter::LtU64 { column, value } => {
                        scanner.filter(&format!("{column} < {value}"))?;
                    }
                }
            }
            if let Some(cols) = projection {
                scanner.project(cols)?;
            }
            scanners.push(scanner);
        }

        let mut rows = 0u64;
        let mut bytes = 0u64;
        for _ in 0..repeats {
            for scanner in &mut scanners {
                let plan = scanner.create_plan().await?;
                let partitions = plan.properties().partitioning.partition_count();
                for partition in 0..partitions {
                    let mut stream = plan.execute(partition, task_ctx.clone())?;
                    while let Some(batch) = stream.next().await {
                        let batch = batch?;
                        rows += batch.num_rows() as u64;
                        bytes = bytes.saturating_add(batch.get_array_memory_size() as u64);
                    }
                }
            }
        }

        Ok((rows, bytes))
    }

    pub async fn scan_count_rows_fragments_measured(
        &self,
        dataset_path: &Path,
        projection: Option<&[String]>,
        filter: Option<&Filter>,
        repeats: u32,
    ) -> Result<(Timing, LatencySummary, u64, u64)> {
        let repeats = repeats.max(1);
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Arc::new(Dataset::open(&uri).await?);

        let exec_options = LanceExecutionOptions {
            use_spilling: false,
            mem_pool_size: None,
            batch_size: Some(FILE_SCAN_BATCH_SIZE as usize),
            target_partition: bench_target_partition(),
            execution_stats_callback: None,
            max_temp_directory_size: None,
            skip_logging: true,
        };
        let session_ctx = new_session_context(&exec_options);
        let task_ctx = session_ctx.task_ctx();

        let fragments = dataset.fragments().as_ref();
        if fragments.is_empty() {
            let mut timing = Timing::from_duration(std::time::Duration::from_millis(0));
            timing.repeat_wall_time_us = Some(vec![0; repeats as usize]);
            return Ok((
                timing,
                LatencySummary {
                    p50_us: 0,
                    p95_us: 0,
                    p99_us: 0,
                },
                0,
                0,
            ));
        }

        let mut fragment_indices: Vec<usize> = (0..fragments.len()).collect();
        if !fragments.windows(2).all(|w| w[0].id <= w[1].id) {
            fragment_indices.sort_by_key(|&idx| fragments[idx].id);
        }

        if let (Some(filter), Some(projection)) = (filter, projection) {
            let Filter::LtU64 { column, .. } = filter;
            if !projection.iter().any(|c| c == column) {
                anyhow::bail!("projection must include filter column {column}");
            }
        }

        let mut scanners = Vec::with_capacity(fragment_indices.len());
        for idx in fragment_indices {
            let fragment = fragments[idx].clone();
            let fragment = FileFragment::new(dataset.clone(), fragment);

            let mut scanner = fragment.scan();
            scanner.batch_size(FILE_SCAN_BATCH_SIZE as usize);
            if let Some(filter) = filter {
                match filter {
                    Filter::LtU64 { column, value } => {
                        scanner.filter(&format!("{column} < {value}"))?;
                    }
                }
            }
            if let Some(cols) = projection {
                scanner.project(cols)?;
            }
            scanners.push(scanner);
        }

        let execute_plan = |plan: Arc<dyn ExecutionPlan>| {
            let task_ctx = task_ctx.clone();
            let mut rows = 0u64;
            let mut bytes = 0u64;
            async move {
                let partitions = plan.properties().partitioning.partition_count();
                for partition in 0..partitions {
                    let mut stream = plan.execute(partition, task_ctx.clone())?;
                    while let Some(batch) = stream.next().await {
                        let batch = batch?;
                        rows += batch.num_rows() as u64;
                        bytes = bytes.saturating_add(batch.get_array_memory_size() as u64);
                    }
                }
                Ok::<(u64, u64), anyhow::Error>((rows, bytes))
            }
        };

        // Warm up OS page cache and engine-internal caches with the same query shape.
        for scanner in scanners.iter_mut() {
            let plan = scanner.create_plan().await?;
            let _ = execute_plan(plan).await?;
        }

        let mut histogram = Histogram::<u64>::new(3)?;
        let mut repeat_wall_time_us = Vec::with_capacity(repeats as usize);
        let timer = WallTimer::start();
        let mut rows = 0u64;
        let mut bytes = 0u64;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            for scanner in scanners.iter_mut() {
                let plan = scanner.create_plan().await?;
                let (r, b) = execute_plan(plan).await?;
                rows += r;
                bytes += b;
            }
            let elapsed_us_u128 = start.elapsed().as_micros();
            let elapsed_us_u64 = elapsed_us_u128.min(u128::from(u64::MAX)) as u64;
            histogram.record(elapsed_us_u64)?;
            repeat_wall_time_us.push(elapsed_us_u128);
        }
        let mut timing = timer.stop();
        timing.repeat_wall_time_us = Some(repeat_wall_time_us);
        Ok((
            timing,
            LatencySummary::from_histogram(&histogram),
            rows,
            bytes,
        ))
    }

    pub async fn take_one(
        &self,
        dataset_path: &Path,
        row_offset: u64,
        projection: Option<&[String]>,
    ) -> Result<()> {
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Dataset::open(&uri).await?;

        let projection = match projection {
            Some(cols) => ProjectionRequest::from_columns(cols.iter(), dataset.schema()),
            None => ProjectionRequest::Schema(dataset.schema().clone().into()),
        };
        let _ = dataset.take(&[row_offset], projection).await?;
        Ok(())
    }

    pub async fn take_blob_one(
        &self,
        dataset_path: &Path,
        row_offset: u64,
        column: &str,
    ) -> Result<usize> {
        let uri = dataset_path.to_string_lossy().to_string();
        let dataset = Arc::new(Dataset::open(&uri).await?);
        self.take_blob_one_opened(&dataset, row_offset, column)
            .await
    }

    pub async fn take_blob_one_opened(
        &self,
        dataset: &Arc<Dataset>,
        row_offset: u64,
        column: &str,
    ) -> Result<usize> {
        let blobs = dataset.take_blobs_by_indices(&[row_offset], column).await?;
        let Some(blob) = blobs.first() else {
            return Ok(0);
        };
        Ok(blob.read().await?.len())
    }

    pub async fn evolution_add_column_sql(
        &self,
        dataset_path: &Path,
        name: &str,
        expr: &str,
    ) -> Result<()> {
        let uri = dataset_path.to_string_lossy().to_string();
        let mut dataset = Dataset::open(&uri).await?;
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(name.to_string(), expr.to_string())]),
                None,
                None,
            )
            .await?;
        Ok(())
    }
}
