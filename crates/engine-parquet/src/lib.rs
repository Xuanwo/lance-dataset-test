use std::collections::HashMap;
use std::fs::File;
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
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};

pub const ENGINE_ADAPTER_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PARQUET_DEP_VERSION: &str = "57.2.0";

pub struct ParquetEngine;

pub struct OpenedParquetFile {
    file: File,
    metadata: ArrowReaderMetadata,
    row_group_start_rows: Vec<u64>,
    root_name_to_index: HashMap<String, usize>,
}

impl ParquetEngine {
    pub fn new() -> Self {
        Self
    }

    pub async fn open_file(&self, path: &Path) -> Result<OpenedParquetFile> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<OpenedParquetFile> {
            let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
            let metadata = ArrowReaderMetadata::load(&file, Default::default())
                .with_context(|| format!("load parquet metadata {}", path.display()))?;

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
    ) -> Result<()> {
        let out = out.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create {}", parent.display()))?;
                }
            }

            let file = File::create(&out).with_context(|| format!("create {}", out.display()))?;
            let mut writer = ArrowWriter::try_new(file, reader.schema(), None)
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
        let opened = self.open_file(file_path).await?;
        self.take_binary_one_opened(&opened, row_offset, column)
            .await
    }

    pub async fn take_binary_one_opened(
        &self,
        opened: &OpenedParquetFile,
        row_offset: u64,
        column: &str,
    ) -> Result<usize> {
        let (rg_idx, in_rg_offset) =
            locate_row_group(&opened.row_group_start_rows, row_offset).unwrap_or((0, 0));

        let file = opened.file.try_clone().context("clone file")?;
        let metadata = opened.metadata.clone();

        let Some(&root_idx) = opened.root_name_to_index.get(column) else {
            anyhow::bail!("column not found: {column}");
        };
        let projection_mask = ProjectionMask::roots(metadata.parquet_schema(), [root_idx]);

        let column = column.to_string();
        tokio::task::spawn_blocking(move || -> Result<usize> {
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
                    let Some(col) = batch.column_by_name(&column) else {
                        return Ok(0);
                    };
                    if col.is_null(idx) {
                        return Ok(0);
                    }
                    if let Some(bin) = col.as_any().downcast_ref::<LargeBinaryArray>() {
                        return Ok(bin.value(idx).len());
                    }
                    if let Some(bin) = col.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                        return Ok(bin.value(idx).len());
                    }
                    return Ok(0);
                }
                seen = seen.saturating_add(batch_rows);
            }
            Ok(0)
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
        self.ingest(reader, out_path).await
    }
}

impl OpenedParquetFile {
    pub fn row_count(&self) -> u64 {
        self.metadata.metadata().file_metadata().num_rows() as u64
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
