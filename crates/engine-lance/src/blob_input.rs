use std::sync::Arc;

use anyhow::Result;
use arrow_array::Array;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use lance::blob::BlobArrayBuilder;
use lance_file::version::LanceFileVersion;

pub fn maybe_wrap_blob_v2_input(
    reader: Box<dyn RecordBatchReader + Send>,
    file_version: LanceFileVersion,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    // Blob v2 requires Lance file version >= 2.2. For older file versions, keep the legacy
    // blob marking (LargeBinary + metadata) so older writers can handle it.
    if file_version.resolve() < LanceFileVersion::V2_2 {
        return Ok(reader);
    }

    let schema = reader.schema();
    let blob_columns: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| {
            if is_legacy_blob_field(field) {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    if blob_columns.is_empty() {
        return Ok(reader);
    }

    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    for &idx in &blob_columns {
        let old = fields[idx].as_ref();
        fields[idx] = Arc::new(lance::blob::blob_field(old.name(), old.is_nullable()));
    }
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));

    Ok(Box::new(BlobV2InputRecordBatchReader::new(
        schema,
        reader,
        blob_columns,
    )))
}

fn is_legacy_blob_field(field: &Field) -> bool {
    let is_marked = field
        .metadata()
        .get(lance_arrow::BLOB_META_KEY)
        .map(String::as_str)
        == Some("true")
        || field
            .metadata()
            .get("lance-encoding:blob")
            .map(String::as_str)
            == Some("true");

    is_marked && matches!(field.data_type(), DataType::LargeBinary)
}

struct BlobV2InputRecordBatchReader {
    schema: SchemaRef,
    inner: Box<dyn RecordBatchReader + Send>,
    blob_columns: Vec<usize>,
}

impl BlobV2InputRecordBatchReader {
    fn new(
        schema: SchemaRef,
        inner: Box<dyn RecordBatchReader + Send>,
        blob_columns: Vec<usize>,
    ) -> Self {
        Self {
            schema,
            inner,
            blob_columns,
        }
    }
}

impl RecordBatchReader for BlobV2InputRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for BlobV2InputRecordBatchReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        let batch = match batch {
            Ok(b) => b,
            Err(err) => return Some(Err(err)),
        };

        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        for &idx in &self.blob_columns {
            let arr = cols.get(idx)?.clone();
            match convert_large_binary_to_blob_v2(&arr) {
                Ok(v2) => cols[idx] = v2,
                Err(err) => return Some(Err(err)),
            }
        }

        Some(RecordBatch::try_new(self.schema.clone(), cols))
    }
}

fn convert_large_binary_to_blob_v2(arr: &ArrayRef) -> std::result::Result<ArrayRef, ArrowError> {
    let binary = arr
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError(
                "expected LargeBinaryArray for legacy blob input column".to_string(),
            )
        })?;

    let mut builder = BlobArrayBuilder::new(binary.len());
    for row in 0..binary.len() {
        if binary.is_null(row) {
            builder
                .push_null()
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
            continue;
        }

        let bytes = binary.value(row);
        if bytes.is_empty() {
            builder
                .push_empty()
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        } else {
            builder
                .push_bytes(bytes)
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        }
    }

    builder
        .finish()
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))
}
