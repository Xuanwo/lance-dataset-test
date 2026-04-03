use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

const COMPRESSION_META_KEY: &str = "lance-encoding:compression";
const COMPRESSION_NONE: &str = "none";

pub fn wrap_binary_compression_none_input(
    reader: Box<dyn RecordBatchReader + Send>,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let schema = reader.schema();
    let mut changed = false;
    let mut fields: Vec<Arc<Field>> = Vec::with_capacity(schema.fields().len());
    for f in schema.fields() {
        let (next, field_changed) = set_binary_compression_none(f.as_ref());
        changed |= field_changed;
        fields.push(Arc::new(next));
    }
    if !changed {
        return Ok(reader);
    }
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    Ok(Box::new(RewriteSchemaRecordBatchReader::new(
        schema, reader,
    )))
}

fn set_binary_compression_none(field: &Field) -> (Field, bool) {
    let is_binary = matches!(
        field.data_type(),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    );
    if !is_binary {
        return (field.clone(), false);
    }

    let mut metadata: HashMap<String, String> = field.metadata().clone();
    let changed = metadata
        .get(COMPRESSION_META_KEY)
        .map(String::as_str)
        .map(|v| v != COMPRESSION_NONE)
        .unwrap_or(true);
    metadata.insert(
        COMPRESSION_META_KEY.to_string(),
        COMPRESSION_NONE.to_string(),
    );

    (
        Field::new(field.name(), field.data_type().clone(), field.is_nullable())
            .with_metadata(metadata),
        changed,
    )
}

struct RewriteSchemaRecordBatchReader {
    schema: SchemaRef,
    inner: Box<dyn RecordBatchReader + Send>,
}

impl RewriteSchemaRecordBatchReader {
    fn new(schema: SchemaRef, inner: Box<dyn RecordBatchReader + Send>) -> Self {
        Self { schema, inner }
    }
}

impl RecordBatchReader for RewriteSchemaRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for RewriteSchemaRecordBatchReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        let batch = match batch {
            Ok(b) => b,
            Err(err) => return Some(Err(err)),
        };
        Some(RecordBatch::try_new(
            self.schema.clone(),
            batch.columns().to_vec(),
        ))
    }
}
