use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use arrow_array::builder::LargeBinaryBuilder;
use arrow_array::builder::StringBuilder;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::UInt64Array;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use flate2::read::GzDecoder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use tar::Archive;
use walkdir::WalkDir;

use crate::workload::DatasetName;

pub const BENCH_ROW_ID: &str = "bench_row_id";
pub const BENCH_SMALL_U64: &str = "bench_small_u64";

#[derive(Debug, Clone)]
pub struct DatasetReadOptions {
    pub batch_size: usize,
    pub limit_rows: Option<u64>,
    pub row_id_offset: u64,
}

impl Default for DatasetReadOptions {
    fn default() -> Self {
        Self {
            batch_size: 8192,
            limit_rows: None,
            row_id_offset: 0,
        }
    }
}

pub struct OpenDataset {
    pub schema: SchemaRef,
    pub reader: Box<dyn RecordBatchReader + Send>,
}

pub async fn open_dataset(
    name: DatasetName,
    input: &Path,
    options: DatasetReadOptions,
) -> Result<OpenDataset> {
    match name {
        DatasetName::LeRobotPushTImage => open_lerobot_dir(input, options),
        DatasetName::LeRobotPushT => open_lerobot_dir(input, options),
        DatasetName::FineWeb => open_parquet_dir(input, options),
        DatasetName::OpenVid => open_openvid_dir(input, options),
        DatasetName::Laion10m => open_webdataset_dir(input, options).await,
        DatasetName::Unknown => bail!("dataset adapter is not implemented yet for {name:?}"),
    }
}

fn open_lerobot_dir(input: &Path, options: DatasetReadOptions) -> Result<OpenDataset> {
    let data_dir = input.join("data");
    let mut files: Vec<PathBuf> = WalkDir::new(&data_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .map(|e| e.into_path())
        .collect();
    files.sort();
    let first = files
        .into_iter()
        .next()
        .with_context(|| format!("no parquet files found under {}", data_dir.display()))?;
    open_parquet_dir(&first, options)
}

fn open_openvid_dir(input: &Path, options: DatasetReadOptions) -> Result<OpenDataset> {
    let opened = open_parquet_dir(input, options)?;
    let blob_bytes = std::env::var("OPENVID_FAKE_BLOB_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    if blob_bytes == 0 {
        return Ok(opened);
    }
    add_fake_blob_column(opened, "video_blob", blob_bytes)
}

fn open_parquet_dir(input: &Path, options: DatasetReadOptions) -> Result<OpenDataset> {
    let mut files: Vec<PathBuf> = WalkDir::new(input)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .map(|e| e.into_path())
        .collect();
    files.sort();

    if files.is_empty() {
        bail!("no parquet files found under input directory");
    }

    // Read schema from the first parquet file and assume all parquet files under the same input
    // directory share a compatible schema.
    let schema = {
        let file = File::open(&files[0])
            .with_context(|| format!("open parquet file: {}", files[0].display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        builder.schema().clone()
    };

    let reader: Box<dyn RecordBatchReader + Send> = if let Some(limit_rows) = options.limit_rows {
        Box::new(RepeatingParquetDirRecordBatchReader::new(
            schema.clone(),
            files,
            options.batch_size,
            limit_rows,
        ))
    } else {
        let mut readers: VecDeque<Box<dyn RecordBatchReader + Send>> = VecDeque::new();
        for file in files {
            let file = File::open(&file)
                .with_context(|| format!("open parquet file: {}", file.display()))?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.with_batch_size(options.batch_size).build()?;
            readers.push_back(Box::new(reader));
        }
        Box::new(ChainedRecordBatchReader::new(schema.clone(), readers, None))
    };

    let schema = sanitize_schema(schema);
    let reader: Box<dyn RecordBatchReader + Send> =
        Box::new(RenameSchemaRecordBatchReader::new(schema.clone(), reader));
    wrap_with_row_id(schema, reader, options.row_id_offset)
}

struct RepeatingParquetDirRecordBatchReader {
    schema: SchemaRef,
    files: Vec<PathBuf>,
    batch_size: usize,
    remaining_rows: u64,
    next_file_idx: usize,
    current_reader: Option<Box<dyn RecordBatchReader + Send>>,
    current_file_had_rows: bool,
    empty_file_steps: usize,
}

impl RepeatingParquetDirRecordBatchReader {
    fn new(schema: SchemaRef, files: Vec<PathBuf>, batch_size: usize, limit_rows: u64) -> Self {
        Self {
            schema,
            files,
            batch_size,
            remaining_rows: limit_rows,
            next_file_idx: 0,
            current_reader: None,
            current_file_had_rows: false,
            empty_file_steps: 0,
        }
    }

    fn open_next_file(&mut self) -> Option<std::result::Result<(), arrow_schema::ArrowError>> {
        if self.files.is_empty() {
            return Some(Err(arrow_schema::ArrowError::InvalidArgumentError(
                "no parquet files found under input directory".to_string(),
            )));
        }
        if self.remaining_rows == 0 {
            return None;
        }

        let path = self.files[self.next_file_idx].clone();
        self.next_file_idx += 1;
        if self.next_file_idx >= self.files.len() {
            self.next_file_idx = 0;
        }

        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                return Some(Err(arrow_schema::ArrowError::IoError(
                    format!("open parquet file {}: {e}", path.display()),
                    e,
                )));
            }
        };
        let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => b,
            Err(e) => return Some(Err(arrow_schema::ArrowError::ParquetError(e.to_string()))),
        };
        let reader = match builder.with_batch_size(self.batch_size).build() {
            Ok(r) => r,
            Err(e) => return Some(Err(arrow_schema::ArrowError::ParquetError(e.to_string()))),
        };
        self.current_reader = Some(Box::new(reader));
        self.current_file_had_rows = false;
        Some(Ok(()))
    }
}

impl RecordBatchReader for RepeatingParquetDirRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for RepeatingParquetDirRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_rows == 0 {
            return None;
        }

        loop {
            if self.current_reader.is_none() {
                match self.open_next_file()? {
                    Ok(_) => {}
                    Err(e) => return Some(Err(e)),
                }
            }

            let Some(reader) = self.current_reader.as_mut() else {
                continue;
            };

            match reader.next() {
                None => {
                    self.current_reader = None;
                    if self.current_file_had_rows {
                        self.empty_file_steps = 0;
                    } else {
                        self.empty_file_steps += 1;
                        if self.empty_file_steps >= self.files.len() {
                            return None;
                        }
                    }
                    continue;
                }
                Some(batch) => {
                    self.empty_file_steps = 0;
                    self.current_file_had_rows = true;
                    let batch = match batch {
                        Ok(batch) => batch,
                        Err(err) => return Some(Err(err)),
                    };
                    let take = (self.remaining_rows as usize).min(batch.num_rows());
                    self.remaining_rows = self.remaining_rows.saturating_sub(take as u64);
                    let batch = batch.slice(0, take);
                    return Some(Ok(batch));
                }
            }
        }
    }
}

async fn open_webdataset_dir(input: &Path, options: DatasetReadOptions) -> Result<OpenDataset> {
    let mut files: Vec<PathBuf> = WalkDir::new(input)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            let name = e.file_name().to_string_lossy();
            name.ends_with(".tar") || name.ends_with(".tar.gz") || name.ends_with(".tgz")
        })
        .map(|e| e.into_path())
        .collect();
    files.sort();

    let mut image_field = Field::new("image", DataType::LargeBinary, true);
    image_field = image_field.with_metadata(HashMap::from([(
        lance_arrow::BLOB_META_KEY.to_string(),
        "true".to_string(),
    )]));

    let fields: Vec<Arc<Field>> = vec![
        Field::new("key", DataType::Utf8, false).into(),
        Field::new("caption", DataType::Utf8, true).into(),
        image_field.into(),
        Field::new("image_ext", DataType::Utf8, true).into(),
    ];
    let schema = Arc::new(Schema::new(fields));

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(WebDatasetRecordBatchReader::new(
        schema.clone(),
        files,
        options.batch_size,
        options.limit_rows,
    ));
    wrap_with_row_id(schema, reader, options.row_id_offset)
}

fn sanitize_schema(schema: SchemaRef) -> SchemaRef {
    let mut seen: HashMap<String, u32> = HashMap::new();
    let fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .map(|field| {
            let mut name = sanitize_field_name(field.name());
            let counter = seen.entry(name.clone()).or_insert(0);
            if *counter > 0 {
                name = format!("{name}_{}", *counter);
            }
            *counter += 1;

            let mut new_field = Field::new(&name, field.data_type().clone(), field.is_nullable());
            new_field = new_field.with_metadata(field.metadata().clone());
            Arc::new(new_field)
        })
        .collect();
    Arc::new(Schema::new(fields))
}

fn sanitize_field_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        let keep = matches!(ch, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_');
        out.push(if keep { ch } else { '_' });
    }
    if out.is_empty() {
        "col".to_string()
    } else {
        out
    }
}

struct RenameSchemaRecordBatchReader {
    schema: SchemaRef,
    inner: Box<dyn RecordBatchReader + Send>,
}

impl RenameSchemaRecordBatchReader {
    fn new(schema: SchemaRef, inner: Box<dyn RecordBatchReader + Send>) -> Self {
        Self { schema, inner }
    }
}

impl RecordBatchReader for RenameSchemaRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for RenameSchemaRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        let batch = match batch {
            Ok(batch) => batch,
            Err(err) => return Some(Err(err)),
        };
        Some(RecordBatch::try_new(
            self.schema.clone(),
            batch.columns().to_vec(),
        ))
    }
}

fn add_fake_blob_column(
    opened: OpenDataset,
    name: &str,
    bytes_per_row: usize,
) -> Result<OpenDataset> {
    let mut field = Field::new(name, DataType::LargeBinary, true);
    let mut metadata = field.metadata().clone();
    metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
    field = field.with_metadata(metadata);

    let mut fields: Vec<Arc<Field>> = opened.schema.fields().iter().cloned().collect();
    fields.push(field.into());
    let schema = Arc::new(Schema::new(fields));

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(FakeBlobRecordBatchReader::new(
        schema.clone(),
        opened.reader,
        bytes_per_row,
    ));
    Ok(OpenDataset { schema, reader })
}

struct FakeBlobRecordBatchReader {
    schema: SchemaRef,
    inner: Box<dyn RecordBatchReader + Send>,
    bytes_per_row: usize,
}

impl FakeBlobRecordBatchReader {
    fn new(
        schema: SchemaRef,
        inner: Box<dyn RecordBatchReader + Send>,
        bytes_per_row: usize,
    ) -> Self {
        Self {
            schema,
            inner,
            bytes_per_row,
        }
    }
}

impl RecordBatchReader for FakeBlobRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for FakeBlobRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        let batch = match batch {
            Ok(b) => b,
            Err(err) => return Some(Err(err)),
        };

        let id_idx = match batch.schema().index_of(BENCH_ROW_ID) {
            Ok(i) => i,
            Err(err) => return Some(Err(err)),
        };
        let ids = match batch.column(id_idx).as_any().downcast_ref::<UInt64Array>() {
            Some(a) => a,
            None => {
                let err =
                    std::io::Error::new(std::io::ErrorKind::Other, "bench_row_id must be UInt64");
                return Some(Err(arrow_schema::ArrowError::ExternalError(Box::new(err))));
            }
        };

        let mut buf = vec![0u8; self.bytes_per_row];
        let mut builder = LargeBinaryBuilder::new();
        for i in 0..batch.num_rows() {
            let id = ids.value(i);
            fill_bytes_from_u64(id, &mut buf);
            builder.append_value(&buf);
        }
        let blob = Arc::new(builder.finish());

        let mut cols = batch.columns().to_vec();
        cols.push(blob);
        Some(RecordBatch::try_new(self.schema.clone(), cols))
    }
}

fn fill_bytes_from_u64(value: u64, buf: &mut [u8]) {
    let pat = value.to_le_bytes();
    for (idx, b) in buf.iter_mut().enumerate() {
        *b = pat[idx % pat.len()];
    }
}

#[derive(Debug)]
struct WebRow {
    key: String,
    caption: Option<String>,
    image: Option<Vec<u8>>,
    image_ext: Option<String>,
}

struct WebDatasetRecordBatchReader {
    schema: SchemaRef,
    batch_size: usize,
    rx: std::sync::mpsc::Receiver<std::result::Result<WebRow, arrow_schema::ArrowError>>,
}

impl WebDatasetRecordBatchReader {
    fn new(
        schema: SchemaRef,
        files: Vec<PathBuf>,
        batch_size: usize,
        remaining_rows: Option<u64>,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<
            std::result::Result<WebRow, arrow_schema::ArrowError>,
        >(8);
        std::thread::spawn(move || {
            let tx_err = tx.clone();
            if let Err(err) = produce_webdataset_rows(files, remaining_rows, tx) {
                let io_err = std::io::Error::new(std::io::ErrorKind::Other, err.to_string());
                let _ = tx_err.send(Err(arrow_schema::ArrowError::ExternalError(Box::new(
                    io_err,
                ))));
            }
        });

        Self {
            schema,
            batch_size,
            rx,
        }
    }
}

impl RecordBatchReader for WebDatasetRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for WebDatasetRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut key_builder = StringBuilder::new();
        let mut caption_builder = StringBuilder::new();
        let mut image_builder = LargeBinaryBuilder::new();
        let mut ext_builder = StringBuilder::new();

        let mut rows = 0usize;
        while rows < self.batch_size {
            let row = match self.rx.recv() {
                Ok(Ok(row)) => row,
                Ok(Err(err)) => return Some(Err(err)),
                Err(_) => break,
            };

            key_builder.append_value(row.key);
            match row.caption {
                Some(v) => caption_builder.append_value(v),
                None => caption_builder.append_null(),
            }
            match row.image {
                Some(v) => image_builder.append_value(v),
                None => image_builder.append_null(),
            }
            match row.image_ext {
                Some(v) => ext_builder.append_value(v),
                None => ext_builder.append_null(),
            }

            rows += 1;
        }

        if rows == 0 {
            return None;
        }

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(key_builder.finish()),
                Arc::new(caption_builder.finish()),
                Arc::new(image_builder.finish()),
                Arc::new(ext_builder.finish()),
            ],
        );
        Some(batch)
    }
}

fn parse_webdataset_entry_name(name: &str) -> Option<(String, String)> {
    let (base, ext) = name.rsplit_once('.')?;
    if base.is_empty() || ext.is_empty() {
        return None;
    }
    Some((base.to_string(), ext.to_ascii_lowercase()))
}

fn produce_webdataset_rows(
    files: Vec<PathBuf>,
    mut remaining_rows: Option<u64>,
    tx: std::sync::mpsc::SyncSender<std::result::Result<WebRow, arrow_schema::ArrowError>>,
) -> Result<()> {
    let mut tx = tx;
    let mut current_key: Option<String> = None;
    let mut current_caption: Option<String> = None;
    let mut current_image: Option<Vec<u8>> = None;
    let mut current_image_ext: Option<String> = None;

    fn flush_row(
        tx: &mut std::sync::mpsc::SyncSender<std::result::Result<WebRow, arrow_schema::ArrowError>>,
        remaining_rows: &mut Option<u64>,
        current_key: &mut Option<String>,
        current_caption: &mut Option<String>,
        current_image: &mut Option<Vec<u8>>,
        current_image_ext: &mut Option<String>,
    ) -> Result<()> {
        if matches!(remaining_rows, Some(0)) {
            return Ok(());
        }

        let Some(key) = current_key.take() else {
            return Ok(());
        };
        let row = WebRow {
            key,
            caption: current_caption.take(),
            image: current_image.take(),
            image_ext: current_image_ext.take(),
        };

        if let Some(r) = remaining_rows.as_mut() {
            *r = r.saturating_sub(1);
        }

        let _ = tx.send(Ok(row));
        Ok(())
    }

    for path in files {
        if matches!(remaining_rows, Some(0)) {
            break;
        }

        let file =
            File::open(&path).with_context(|| format!("open tar shard: {}", path.display()))?;
        let reader: Box<dyn std::io::Read + Send> = match path.extension().and_then(|s| s.to_str())
        {
            Some("gz") => Box::new(GzDecoder::new(file)),
            Some("tgz") => Box::new(GzDecoder::new(file)),
            _ => Box::new(file),
        };

        let mut archive = Archive::new(reader);
        for entry in archive.entries()? {
            if matches!(remaining_rows, Some(0)) {
                break;
            }
            let mut entry = entry?;
            let path = entry.path()?;
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let Some((key, ext)) = parse_webdataset_entry_name(name) else {
                continue;
            };

            if current_key.as_deref() != Some(&key) {
                flush_row(
                    &mut tx,
                    &mut remaining_rows,
                    &mut current_key,
                    &mut current_caption,
                    &mut current_image,
                    &mut current_image_ext,
                )?;
                current_key = Some(key);
            }

            let mut data = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut data)?;
            match ext.as_str() {
                "txt" => current_caption = Some(String::from_utf8(data).unwrap_or_default()),
                "jpg" | "jpeg" | "png" | "webp" => {
                    current_image = Some(data);
                    current_image_ext = Some(ext);
                }
                _ => {}
            }
        }
    }

    flush_row(
        &mut tx,
        &mut remaining_rows,
        &mut current_key,
        &mut current_caption,
        &mut current_image,
        &mut current_image_ext,
    )?;
    Ok(())
}

struct ChainedRecordBatchReader {
    schema: SchemaRef,
    readers: VecDeque<Box<dyn RecordBatchReader + Send>>,
    remaining_rows: Option<u64>,
}

impl ChainedRecordBatchReader {
    fn new(
        schema: SchemaRef,
        readers: VecDeque<Box<dyn RecordBatchReader + Send>>,
        remaining_rows: Option<u64>,
    ) -> Self {
        Self {
            schema,
            readers,
            remaining_rows,
        }
    }
}

impl RecordBatchReader for ChainedRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for ChainedRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if matches!(self.remaining_rows, Some(0)) {
            return None;
        }

        loop {
            let front = self.readers.front_mut()?;
            match front.next() {
                None => {
                    self.readers.pop_front();
                    continue;
                }
                Some(batch) => {
                    let batch = match batch {
                        Ok(batch) => batch,
                        Err(err) => return Some(Err(err)),
                    };
                    if let Some(remaining) = &mut self.remaining_rows {
                        let take = (*remaining as usize).min(batch.num_rows());
                        *remaining = remaining.saturating_sub(take as u64);
                        let batch = batch.slice(0, take);
                        return Some(Ok(batch));
                    }
                    return Some(Ok(batch));
                }
            }
        }
    }
}

fn wrap_with_row_id(
    schema: SchemaRef,
    reader: Box<dyn RecordBatchReader + Send>,
    row_id_offset: u64,
) -> Result<OpenDataset> {
    let has_row_id = schema.index_of(BENCH_ROW_ID).is_ok();
    let has_small_u64 = schema.index_of(BENCH_SMALL_U64).is_ok();
    if has_row_id && has_small_u64 {
        // Input already contains benchmark helper columns; keep the existing schema/values.
        return Ok(OpenDataset { schema, reader });
    }
    if has_row_id || has_small_u64 {
        bail!(
            "input schema must contain both helper columns or none, found {}={}, {}={}",
            BENCH_ROW_ID,
            has_row_id,
            BENCH_SMALL_U64,
            has_small_u64
        );
    }

    let mut fields = schema.fields().to_vec();
    fields.push(Field::new(BENCH_ROW_ID, DataType::UInt64, false).into());
    fields.push(Field::new(BENCH_SMALL_U64, DataType::UInt64, false).into());
    let schema = Arc::new(Schema::new(fields));

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RowIdRecordBatchReader::new(
        schema.clone(),
        reader,
        row_id_offset,
    ));
    Ok(OpenDataset { schema, reader })
}

struct RowIdRecordBatchReader {
    schema: SchemaRef,
    inner: Box<dyn RecordBatchReader + Send>,
    next_row_id: u64,
}

impl RowIdRecordBatchReader {
    fn new(
        schema: SchemaRef,
        inner: Box<dyn RecordBatchReader + Send>,
        row_id_offset: u64,
    ) -> Self {
        Self {
            schema,
            inner,
            next_row_id: row_id_offset,
        }
    }
}

impl RecordBatchReader for RowIdRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for RowIdRecordBatchReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        let batch = match batch {
            Ok(batch) => batch,
            Err(err) => return Some(Err(err)),
        };

        let num_rows = batch.num_rows();
        let row_ids = (self.next_row_id..self.next_row_id + num_rows as u64).collect::<Vec<_>>();
        self.next_row_id += num_rows as u64;
        let row_id_array = Arc::new(UInt64Array::from(row_ids));
        let small_u64 = Arc::new(UInt64Array::from(
            (self.next_row_id - num_rows as u64..self.next_row_id)
                .map(splitmix64)
                .collect::<Vec<_>>(),
        ));

        let mut columns = batch.columns().to_vec();
        columns.push(row_id_array);
        columns.push(small_u64);

        Some(RecordBatch::try_new(self.schema.clone(), columns))
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
