use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::builder::{BooleanBuilder, UInt64Builder};
use arrow_array::cast::as_primitive_array;
use arrow_array::{Array, LargeBinaryArray, RecordBatch, RecordBatchReader, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::filter::filter_record_batch;
use bench_core::metrics::{LatencySummary, Timing, WallTimer};
use bench_core::query::Filter;
use hdrhistogram::Histogram;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelectionPolicy,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::metadata::{KeyValue, PageIndexPolicy};
use parquet::file::properties::{WriterProperties, DEFAULT_PAGE_SIZE};

pub const ENGINE_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PARQUET_DEP_VERSION: &str = "58.3.0";
pub const RANDOM_BLOB_DATA_PAGE_SIZE_LIMIT: usize = DEFAULT_PAGE_SIZE;
pub const RANDOM_BLOB_WRITE_BATCH_SIZE: usize = 1;
pub const RANDOM_BLOB_MAX_ROW_GROUP_BYTES: usize = 128 * 1024 * 1024;
const WRITER_PROFILE_METADATA_KEY: &str = "lance-dataset-test.parquet-writer-profile";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParquetReadMode {
    #[default]
    Sequential,
    RowSelection,
}

impl ParquetReadMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::RowSelection => "row-selection",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParquetWriterProfile {
    #[default]
    Default,
    RandomBlob,
}

impl ParquetWriterProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::RandomBlob => "random-blob",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParquetWriterOptions {
    pub profile: ParquetWriterProfile,
    pub blob_columns: Vec<String>,
    pub data_page_size_limit: Option<usize>,
    pub write_batch_size: Option<usize>,
    pub max_row_group_bytes: Option<usize>,
}

impl Default for ParquetWriterOptions {
    fn default() -> Self {
        Self {
            profile: ParquetWriterProfile::Default,
            blob_columns: Vec::new(),
            data_page_size_limit: None,
            write_batch_size: None,
            max_row_group_bytes: None,
        }
    }
}

impl ParquetWriterOptions {
    pub fn random_blob(blob_columns: Vec<String>) -> Self {
        Self {
            profile: ParquetWriterProfile::RandomBlob,
            blob_columns,
            data_page_size_limit: Some(RANDOM_BLOB_DATA_PAGE_SIZE_LIMIT),
            write_batch_size: Some(RANDOM_BLOB_WRITE_BATCH_SIZE),
            max_row_group_bytes: Some(RANDOM_BLOB_MAX_ROW_GROUP_BYTES),
        }
    }

    fn writer_properties(&self) -> Result<Option<WriterProperties>> {
        if self.profile == ParquetWriterProfile::Default {
            return Ok(None);
        }

        let data_page_size_limit = self
            .data_page_size_limit
            .context("random-blob profile requires data_page_size_limit")?;
        let write_batch_size = self
            .write_batch_size
            .context("random-blob profile requires write_batch_size")?;
        let max_row_group_bytes = self
            .max_row_group_bytes
            .context("random-blob profile requires max_row_group_bytes")?;
        if data_page_size_limit == 0 || write_batch_size == 0 || max_row_group_bytes == 0 {
            anyhow::bail!("random-blob writer limits must be greater than zero");
        }
        if self.blob_columns.is_empty() {
            anyhow::bail!("random-blob profile requires at least one blob column");
        }

        let builder = WriterProperties::builder()
            .set_data_page_size_limit(data_page_size_limit)
            .set_write_batch_size(write_batch_size)
            .set_max_row_group_bytes(Some(max_row_group_bytes))
            .set_offset_index_disabled(false)
            .set_key_value_metadata(Some(vec![KeyValue::new(
                WRITER_PROFILE_METADATA_KEY.to_string(),
                self.profile.as_str().to_string(),
            )]));
        Ok(Some(builder.build()))
    }
}

pub struct ParquetEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobBatchRead {
    pub selected_blobs: usize,
    pub materialized_blobs: usize,
    pub total_bytes: usize,
}

pub struct OpenedParquetFile {
    file: File,
    metadata: ArrowReaderMetadata,
    row_group_start_rows: Vec<u64>,
    root_name_to_index: HashMap<String, usize>,
    read_mode: ParquetReadMode,
    writer_profile: ParquetWriterProfile,
}

impl ParquetEngine {
    pub fn new() -> Self {
        Self
    }

    pub async fn open_file(&self, path: &Path) -> Result<OpenedParquetFile> {
        self.open_file_with_mode(path, ParquetReadMode::Sequential)
            .await
    }

    pub async fn open_file_with_mode(
        &self,
        path: &Path,
        read_mode: ParquetReadMode,
    ) -> Result<OpenedParquetFile> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<OpenedParquetFile> {
            let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
            let options = match read_mode {
                ParquetReadMode::Sequential => ArrowReaderOptions::new(),
                ParquetReadMode::RowSelection => {
                    ArrowReaderOptions::new().with_offset_index_policy(PageIndexPolicy::Required)
                }
            };
            let metadata = ArrowReaderMetadata::load(&file, options)
                .with_context(|| format!("load parquet metadata {}", path.display()))?;
            let writer_profile = metadata
                .metadata()
                .file_metadata()
                .key_value_metadata()
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|entry| entry.key == WRITER_PROFILE_METADATA_KEY)
                })
                .and_then(|entry| entry.value.as_deref())
                .map(|value| match value {
                    "default" => Ok(ParquetWriterProfile::Default),
                    "random-blob" => Ok(ParquetWriterProfile::RandomBlob),
                    other => anyhow::bail!("unknown parquet writer profile in metadata: {other}"),
                })
                .transpose()?
                .unwrap_or_default();

            let mut row_group_start_rows = Vec::with_capacity(metadata.metadata().num_row_groups());
            let mut start = 0u64;
            for rg in metadata.metadata().row_groups() {
                row_group_start_rows.push(start);
                start = start.saturating_add(rg.num_rows() as u64);
            }

            let mut root_name_to_index = HashMap::new();
            for (idx, field) in metadata.schema().fields().iter().enumerate() {
                root_name_to_index.insert(field.name().to_string(), idx);
            }

            Ok(OpenedParquetFile {
                file,
                metadata,
                row_group_start_rows,
                root_name_to_index,
                read_mode,
                writer_profile,
            })
        })
        .await?
    }

    pub async fn schema_field_names(&self, file_path: &Path) -> Result<Vec<String>> {
        let opened = self.open_file(file_path).await?;
        Ok(opened
            .metadata
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect())
    }

    pub async fn smoke_check(&self) -> Result<()> {
        Ok(())
    }

    pub async fn ingest(
        &self,
        reader: Box<dyn RecordBatchReader + Send>,
        out: &Path,
        options: ParquetWriterOptions,
    ) -> Result<()> {
        let properties = options.writer_properties()?;
        let out = out.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create {}", parent.display()))?;
                }
            }

            let file = File::create(&out).with_context(|| format!("create {}", out.display()))?;
            let mut writer = ArrowWriter::try_new(file, reader.schema(), properties)
                .with_context(|| format!("create parquet writer {}", out.display()))?;

            for batch in reader {
                let batch = batch?;
                writer.write(&batch)?;
            }

            writer.close()?;
            Ok(())
        })
        .await?
    }

    pub async fn scan_count_rows_measured(
        &self,
        file_path: &Path,
        projection: Option<&[String]>,
        filter: Option<&Filter>,
        limit_rows: Option<u64>,
        row_offset: Option<u64>,
        repeats: u32,
    ) -> Result<(Timing, LatencySummary, u64, u64)> {
        let repeats = repeats.max(1);
        let opened = self.open_file(file_path).await?;

        let projection_mask = build_projection_mask(&opened, projection)?;
        let filter_spec = filter.cloned();
        let opened_ref = &opened;

        let scan_once = || async {
            let file = opened_ref.file.try_clone().context("clone file")?;
            let metadata = opened_ref.metadata.clone();
            let mut reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
                .with_projection(projection_mask.clone())
                .with_batch_size(8192)
                .build()?;

            let mut rows = 0u64;
            let mut bytes = 0u64;
            let mut remaining_offset = row_offset.unwrap_or(0);
            let mut remaining_limit = limit_rows.unwrap_or(u64::MAX);
            while let Some(batch) = reader.next() {
                let batch = batch?;
                let batch = match &filter_spec {
                    None => batch,
                    Some(Filter::LtU64 { column, value }) => {
                        filter_record_batch_lt_u64(&batch, column, *value)?
                    }
                };
                if batch.num_rows() == 0 {
                    continue;
                }
                if remaining_offset >= batch.num_rows() as u64 {
                    remaining_offset -= batch.num_rows() as u64;
                    continue;
                }
                let start = remaining_offset as usize;
                let available = batch.num_rows() - start;
                let take = available.min(remaining_limit.min(usize::MAX as u64) as usize);
                let batch = batch.slice(start, take);
                rows += batch.num_rows() as u64;
                bytes = bytes.saturating_add(batch.get_array_memory_size() as u64);
                remaining_offset = 0;
                remaining_limit = remaining_limit.saturating_sub(batch.num_rows() as u64);
                if remaining_limit == 0 {
                    break;
                }
            }
            Ok::<(u64, u64), anyhow::Error>((rows, bytes))
        };

        // Warm up OS page cache and Parquet metadata path.
        let _ = scan_once().await?;

        let mut histogram = Histogram::<u64>::new(3)?;
        let mut repeat_wall_time_us = Vec::with_capacity(repeats as usize);
        let timer = WallTimer::start();
        let mut rows = 0u64;
        let mut bytes = 0u64;
        for _ in 0..repeats {
            let start = std::time::Instant::now();
            let (r, b) = scan_once().await?;
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

    pub async fn take_one(
        &self,
        file_path: &Path,
        row_offset: u64,
        projection: Option<&[String]>,
    ) -> Result<()> {
        let opened = self.open_file(file_path).await?;
        self.take_one_opened(&opened, row_offset, projection).await
    }

    pub async fn take_one_opened(
        &self,
        opened: &OpenedParquetFile,
        row_offset: u64,
        projection: Option<&[String]>,
    ) -> Result<()> {
        let (rg_idx, in_rg_offset) =
            locate_row_group(&opened.row_group_start_rows, row_offset).unwrap_or((0, 0));

        let file = opened.file.try_clone().context("clone file")?;
        let metadata = opened.metadata.clone();
        let projection_mask = build_projection_mask(opened, projection)?;

        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
                .with_row_groups(vec![rg_idx])
                .with_projection(projection_mask)
                .with_batch_size(1024)
                .build()?;

            let mut seen = 0u64;
            while let Some(batch) = reader.next() {
                let batch = batch?;
                let batch_rows = batch.num_rows() as u64;
                if in_rg_offset < seen.saturating_add(batch_rows) {
                    let idx = (in_rg_offset - seen) as usize;
                    if let Some(col) = batch.columns().first() {
                        if col.is_null(idx) {
                            return Ok(());
                        }
                        match col.data_type() {
                            DataType::UInt64 => {
                                let arr = as_primitive_array::<arrow_array::types::UInt64Type>(
                                    col.as_ref(),
                                );
                                let _ = arr.value(idx);
                            }
                            DataType::LargeBinary => {
                                let arr = col
                                    .as_any()
                                    .downcast_ref::<LargeBinaryArray>()
                                    .context("expected LargeBinaryArray")?;
                                let _ = arr.value(idx).len();
                            }
                            _ => {}
                        }
                    }
                    return Ok(());
                }
                seen = seen.saturating_add(batch_rows);
            }
            Ok(())
        })
        .await?
    }

    pub async fn take_binary_one(
        &self,
        file_path: &Path,
        row_offset: u64,
        column: &str,
    ) -> Result<usize> {
        self.take_binary_one_with_mode(file_path, row_offset, column, ParquetReadMode::Sequential)
            .await
    }

    pub async fn take_binary_one_with_mode(
        &self,
        file_path: &Path,
        row_offset: u64,
        column: &str,
        read_mode: ParquetReadMode,
    ) -> Result<usize> {
        let opened = self.open_file_with_mode(file_path, read_mode).await?;
        self.take_binary_one_opened(&opened, row_offset, column)
            .await
    }

    pub async fn take_binary_one_opened(
        &self,
        opened: &OpenedParquetFile,
        row_offset: u64,
        column: &str,
    ) -> Result<usize> {
        match self
            .read_binary_one_impl(opened, row_offset, column, false)
            .await?
        {
            BinaryRead::Null => Ok(0),
            BinaryRead::Length(len) => Ok(len),
            BinaryRead::Value(value) => Ok(value.len()),
        }
    }

    pub async fn read_binary_one_opened(
        &self,
        opened: &OpenedParquetFile,
        row_offset: u64,
        column: &str,
    ) -> Result<Option<Vec<u8>>> {
        match self
            .read_binary_one_impl(opened, row_offset, column, true)
            .await?
        {
            BinaryRead::Null => Ok(None),
            BinaryRead::Value(value) => Ok(Some(value)),
            BinaryRead::Length(_) => unreachable!("materialized reads return a value"),
        }
    }

    pub async fn read_binary_batch_opened(
        &self,
        opened: &OpenedParquetFile,
        row_offsets: &[u64],
        column: &str,
        preserve_order: bool,
    ) -> Result<BlobBatchRead> {
        if opened.read_mode != ParquetReadMode::RowSelection {
            anyhow::bail!("batched binary reads require parquet row-selection mode");
        }
        if row_offsets.is_empty() {
            anyhow::bail!("batched binary reads require at least one row");
        }

        let file = opened.file.try_clone().context("clone file")?;
        let metadata = opened.metadata.clone();
        let total_rows =
            usize::try_from(opened.row_count()).context("parquet row count does not fit usize")?;
        let Some(&root_idx) = opened.root_name_to_index.get(column) else {
            anyhow::bail!("column not found: {column}");
        };
        let projection_mask = ProjectionMask::roots(metadata.parquet_schema(), [root_idx]);
        let row_offsets = row_offsets.to_vec();
        let column = column.to_string();

        tokio::task::spawn_blocking(move || -> Result<BlobBatchRead> {
            let mut sorted_rows = row_offsets.clone();
            sorted_rows.sort_unstable();
            if sorted_rows.windows(2).any(|pair| pair[0] == pair[1]) {
                anyhow::bail!("a parquet RowSelection request cannot contain duplicate rows");
            }
            if let Some(&last) = sorted_rows.last() {
                if last >= total_rows as u64 {
                    anyhow::bail!(
                        "row offset {last} is outside parquet file with {total_rows} rows"
                    );
                }
            }

            let selection = RowSelection::from_consecutive_ranges(
                consecutive_row_ranges(&sorted_rows)?.into_iter(),
                total_rows,
            );
            let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
                .with_projection(projection_mask)
                .with_row_selection(selection)
                .with_row_selection_policy(RowSelectionPolicy::Selectors)
                .with_batch_size(sorted_rows.len())
                .build()?;

            let mut lengths = Vec::with_capacity(sorted_rows.len());
            for batch in reader {
                let batch = batch?;
                let values = batch
                    .column_by_name(&column)
                    .with_context(|| format!("column missing from projected batch: {column}"))?;
                if let Some(binary) = values.as_any().downcast_ref::<LargeBinaryArray>() {
                    for index in 0..binary.len() {
                        lengths.push((!binary.is_null(index)).then(|| binary.value(index).len()));
                    }
                } else if let Some(binary) =
                    values.as_any().downcast_ref::<arrow_array::BinaryArray>()
                {
                    for index in 0..binary.len() {
                        lengths.push((!binary.is_null(index)).then(|| binary.value(index).len()));
                    }
                } else {
                    anyhow::bail!("column {column} is not Binary or LargeBinary");
                }
            }
            if lengths.len() != sorted_rows.len() {
                anyhow::bail!(
                    "parquet RowSelection returned {} rows for {} requested rows",
                    lengths.len(),
                    sorted_rows.len()
                );
            }

            let ordered_lengths = if preserve_order {
                let positions: HashMap<u64, usize> = sorted_rows
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(position, row)| (row, position))
                    .collect();
                row_offsets
                    .iter()
                    .map(|row| {
                        let position = positions
                            .get(row)
                            .copied()
                            .with_context(|| format!("missing selected parquet row {row}"))?;
                        Ok(lengths[position])
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                lengths
            };
            let materialized_blobs = ordered_lengths
                .iter()
                .filter(|value| value.is_some())
                .count();
            let total_bytes = ordered_lengths.iter().flatten().copied().sum();
            Ok(BlobBatchRead {
                selected_blobs: row_offsets.len(),
                materialized_blobs,
                total_bytes,
            })
        })
        .await?
    }

    async fn read_binary_one_impl(
        &self,
        opened: &OpenedParquetFile,
        row_offset: u64,
        column: &str,
        materialize: bool,
    ) -> Result<BinaryRead> {
        let (rg_idx, in_rg_offset) =
            locate_row_group(&opened.row_group_start_rows, row_offset).unwrap_or((0, 0));

        let file = opened.file.try_clone().context("clone file")?;
        let metadata = opened.metadata.clone();
        let read_mode = opened.read_mode;

        let Some(&root_idx) = opened.root_name_to_index.get(column) else {
            anyhow::bail!("column not found: {column}");
        };
        let projection_mask = ProjectionMask::roots(metadata.parquet_schema(), [root_idx]);

        let column = column.to_string();
        tokio::task::spawn_blocking(move || -> Result<BinaryRead> {
            let row_group_rows = usize::try_from(metadata.metadata().row_group(rg_idx).num_rows())
                .context("row group row count does not fit usize")?;
            let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata)
                .with_row_groups(vec![rg_idx])
                .with_projection(projection_mask);

            let (target_offset, selection) = match read_mode {
                ParquetReadMode::Sequential => (in_rg_offset, None),
                ParquetReadMode::RowSelection => {
                    let start = usize::try_from(in_rg_offset)
                        .context("row offset within row group does not fit usize")?;
                    if start >= row_group_rows {
                        anyhow::bail!(
                            "row offset {row_offset} is outside row group {rg_idx} with {row_group_rows} rows"
                        );
                    }
                    let selection = RowSelection::from_consecutive_ranges(
                        std::iter::once(start..start + 1),
                        row_group_rows,
                    );
                    (0, Some(selection))
                }
            };

            let mut reader = match read_mode {
                ParquetReadMode::Sequential => builder.with_batch_size(1024).build()?,
                ParquetReadMode::RowSelection => builder
                    .with_row_selection(selection.expect("row-selection mode has a selection"))
                    .with_row_selection_policy(RowSelectionPolicy::Selectors)
                    .with_batch_size(1)
                    .build()?,
            };

            let mut seen = 0u64;
            while let Some(batch) = reader.next() {
                let batch = batch?;
                let batch_rows = batch.num_rows() as u64;
                if target_offset < seen.saturating_add(batch_rows) {
                    let idx = (target_offset - seen) as usize;
                    let col = batch
                        .column_by_name(&column)
                        .with_context(|| format!("column missing from projected batch: {column}"))?;
                    if col.is_null(idx) {
                        return Ok(BinaryRead::Null);
                    }
                    if let Some(bin) = col.as_any().downcast_ref::<LargeBinaryArray>() {
                        let value = bin.value(idx);
                        return Ok(if materialize {
                            BinaryRead::Value(value.to_vec())
                        } else {
                            BinaryRead::Length(value.len())
                        });
                    }
                    if let Some(bin) = col.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                        let value = bin.value(idx);
                        return Ok(if materialize {
                            BinaryRead::Value(value.to_vec())
                        } else {
                            BinaryRead::Length(value.len())
                        });
                    }
                    anyhow::bail!("column {column} is not Binary or LargeBinary");
                }
                seen = seen.saturating_add(batch_rows);
            }
            anyhow::bail!("row offset {row_offset} was not returned by parquet reader")
        })
        .await?
    }

    pub async fn evolution_add_column_rewrite(
        &self,
        in_path: &Path,
        out_path: &Path,
        new_column: &str,
    ) -> Result<()> {
        let in_path = in_path.to_path_buf();
        let new_column = new_column.to_string();
        let (tx, rx) = tokio::sync::mpsc::channel::<
            std::result::Result<RecordBatch, arrow_schema::ArrowError>,
        >(4);
        let (schema_tx, schema_rx) = tokio::sync::oneshot::channel::<arrow_schema::SchemaRef>();

        tokio::spawn(async move {
            let mut schema_tx = Some(schema_tx);
            let opened = match ParquetEngine::new().open_file(&in_path).await {
                Ok(v) => v,
                Err(err) => {
                    let err = std::io::Error::new(std::io::ErrorKind::Other, err.to_string());
                    let _ = tx
                        .send(Err(arrow_schema::ArrowError::ExternalError(Box::new(err))))
                        .await;
                    return;
                }
            };

            let file = match opened.file.try_clone() {
                Ok(f) => f,
                Err(err) => {
                    let err = std::io::Error::new(std::io::ErrorKind::Other, err.to_string());
                    let _ = tx
                        .send(Err(arrow_schema::ArrowError::ExternalError(Box::new(err))))
                        .await;
                    return;
                }
            };
            let metadata = opened.metadata.clone();
            let mut reader =
                match ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata).build() {
                    Ok(r) => r,
                    Err(err) => {
                        let err = std::io::Error::new(std::io::ErrorKind::Other, err.to_string());
                        let _ = tx
                            .send(Err(arrow_schema::ArrowError::ExternalError(Box::new(err))))
                            .await;
                        return;
                    }
                };

            let mut out_schema: Option<arrow_schema::SchemaRef> = None;
            while let Some(batch) = reader.next() {
                let batch = match batch {
                    Ok(b) => b,
                    Err(err) => {
                        let err = std::io::Error::new(std::io::ErrorKind::Other, err.to_string());
                        let _ = tx
                            .send(Err(arrow_schema::ArrowError::ExternalError(Box::new(err))))
                            .await;
                        return;
                    }
                };

                let schema = match out_schema.as_ref() {
                    Some(s) => s.clone(),
                    None => {
                        let mut fields = batch.schema().fields().to_vec();
                        fields.push(Field::new(&new_column, DataType::UInt64, false).into());
                        let schema = Arc::new(Schema::new(fields));
                        if let Some(tx_schema) = schema_tx.take() {
                            let _ = tx_schema.send(schema.clone());
                        }
                        out_schema = Some(schema.clone());
                        schema
                    }
                };

                let row_id = batch
                    .column_by_name(bench_core::dataset::BENCH_ROW_ID)
                    .and_then(|col| col.as_any().downcast_ref::<UInt64Array>());
                let mut builder = UInt64Builder::new();
                if let Some(row_id) = row_id {
                    for i in 0..row_id.len() {
                        builder.append_value(row_id.value(i).wrapping_add(1));
                    }
                } else {
                    for _ in 0..batch.num_rows() {
                        builder.append_value(1);
                    }
                }
                let new_arr = Arc::new(builder.finish());

                let mut cols = batch.columns().to_vec();
                cols.push(new_arr);

                let out_batch = match RecordBatch::try_new(schema, cols) {
                    Ok(b) => b,
                    Err(err) => {
                        let _ = tx.send(Err(err)).await;
                        return;
                    }
                };
                if tx.send(Ok(out_batch)).await.is_err() {
                    return;
                }
            }
        });

        let schema = schema_rx
            .await
            .unwrap_or_else(|_| Arc::new(Schema::new(Vec::<Field>::new())));
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(TokioReceiverRecordBatchReader::new(schema, rx));
        self.ingest(reader, out_path, ParquetWriterOptions::default())
            .await
    }
}

enum BinaryRead {
    Null,
    Length(usize),
    Value(Vec<u8>),
}

impl OpenedParquetFile {
    pub fn row_count(&self) -> u64 {
        self.metadata.metadata().file_metadata().num_rows() as u64
    }

    pub fn row_group_count(&self) -> usize {
        self.metadata.metadata().num_row_groups()
    }

    pub fn has_offset_index(&self) -> bool {
        self.metadata.metadata().offset_index().is_some()
    }

    pub fn writer_profile(&self) -> ParquetWriterProfile {
        self.writer_profile
    }
}

fn build_projection_mask(
    opened: &OpenedParquetFile,
    projection: Option<&[String]>,
) -> Result<ProjectionMask> {
    let Some(cols) = projection else {
        return Ok(ProjectionMask::all());
    };
    if cols.is_empty() {
        return Ok(ProjectionMask::all());
    }

    let mut root_indices = Vec::with_capacity(cols.len());
    for col in cols {
        let Some(&idx) = opened.root_name_to_index.get(col) else {
            anyhow::bail!("projection column not found: {col}");
        };
        root_indices.push(idx);
    }
    Ok(ProjectionMask::roots(
        opened.metadata.parquet_schema(),
        root_indices,
    ))
}

fn locate_row_group(row_group_start_rows: &[u64], row_offset: u64) -> Option<(usize, u64)> {
    if row_group_start_rows.is_empty() {
        return None;
    }
    let idx = match row_group_start_rows.binary_search(&row_offset) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    let start = row_group_start_rows[idx];
    Some((idx, row_offset.saturating_sub(start)))
}

fn consecutive_row_ranges(rows: &[u64]) -> Result<Vec<Range<usize>>> {
    let mut ranges = Vec::new();
    let Some(&first) = rows.first() else {
        return Ok(ranges);
    };
    let mut start = usize::try_from(first).context("row offset does not fit usize")?;
    let mut end = start + 1;
    for &row in rows.iter().skip(1) {
        let row = usize::try_from(row).context("row offset does not fit usize")?;
        if row == end {
            end += 1;
        } else {
            ranges.push(start..end);
            start = row;
            end = row + 1;
        }
    }
    ranges.push(start..end);
    Ok(ranges)
}

fn filter_record_batch_lt_u64(
    batch: &RecordBatch,
    column: &str,
    threshold: u64,
) -> Result<RecordBatch> {
    let idx = batch.schema().index_of(column)?;
    let col = batch.column(idx);
    let values = as_primitive_array::<arrow_array::types::UInt64Type>(col.as_ref());

    let mut mask_builder = BooleanBuilder::new();
    for i in 0..values.len() {
        if values.is_null(i) {
            mask_builder.append_value(false);
        } else {
            mask_builder.append_value(values.value(i) < threshold);
        }
    }
    let mask = mask_builder.finish();
    Ok(filter_record_batch(batch, &mask)?)
}

struct TokioReceiverRecordBatchReader {
    schema: Arc<Schema>,
    rx: tokio::sync::mpsc::Receiver<std::result::Result<RecordBatch, arrow_schema::ArrowError>>,
}

impl TokioReceiverRecordBatchReader {
    fn new(
        schema: Arc<Schema>,
        rx: tokio::sync::mpsc::Receiver<std::result::Result<RecordBatch, arrow_schema::ArrowError>>,
    ) -> Self {
        Self { schema, rx }
    }
}

impl RecordBatchReader for TokioReceiverRecordBatchReader {
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for TokioReceiverRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.blocking_recv() {
            Some(v) => Some(v),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use parquet::schema::types::ColumnPath;

    use super::*;

    #[test]
    fn random_blob_layout_keeps_default_column_encoding() {
        let defaults = WriterProperties::default();
        let tuned = ParquetWriterOptions::random_blob(vec!["blob".to_string()])
            .writer_properties()
            .unwrap()
            .unwrap();
        let column = ColumnPath::from("blob");

        assert_eq!(tuned.encoding(&column), defaults.encoding(&column));
        assert_eq!(tuned.compression(&column), defaults.compression(&column));
        assert_eq!(
            tuned.dictionary_enabled(&column),
            defaults.dictionary_enabled(&column)
        );
        assert_eq!(
            tuned.statistics_enabled(&column),
            defaults.statistics_enabled(&column)
        );
        assert_eq!(tuned.data_page_size_limit(), DEFAULT_PAGE_SIZE);
        assert_eq!(tuned.write_batch_size(), RANDOM_BLOB_WRITE_BATCH_SIZE);
    }
}
