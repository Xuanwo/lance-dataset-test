use std::sync::Arc;

use arrow_array::{
    Int32Array, LargeBinaryArray, RecordBatch, RecordBatchIterator, RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
use tempfile::tempdir;

#[tokio::test]
async fn lance_ingest_and_scan_smoke() {
    let schema = Arc::new(Schema::new(vec![Arc::new(Field::new(
        "x",
        DataType::Int32,
        false,
    ))]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema.clone(),
    ));

    let dir = tempdir().unwrap();
    let out = dir.path().join("lance");

    let engine = engine_lance::LanceEngine::new();
    engine
        .ingest(reader, &out, engine_lance::IngestOptions::default())
        .await
        .unwrap();
    let (rows, _bytes) = engine.scan_count_rows(&out, None, None, 1).await.unwrap();
    assert_eq!(rows, 3);
}

#[tokio::test]
async fn lance_v2_2_blob_ingest_and_take_smoke() {
    let mut blob_field = Field::new("blob", DataType::LargeBinary, true);
    blob_field = blob_field.with_metadata(std::collections::HashMap::from([(
        "lance-encoding:blob".to_string(),
        "true".to_string(),
    )]));

    let schema = Arc::new(Schema::new(vec![
        Arc::new(Field::new("x", DataType::Int32, false)),
        Arc::new(blob_field),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(LargeBinaryArray::from_iter([
                Some(&b"abc"[..]),
                Some(&b""[..]),
                Some(&b"xy"[..]),
            ])),
        ],
    )
    .unwrap();
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema.clone(),
    ));

    let dir = tempdir().unwrap();
    let out = dir.path().join("lance-v2_2-blob");

    let engine = engine_lance::LanceEngine::new();
    engine
        .ingest(
            reader,
            &out,
            engine_lance::IngestOptions {
                data_storage_version: Some("2.2".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let ds = Dataset::open(&out.to_string_lossy().to_string())
        .await
        .unwrap();
    let blob = ds.schema().field("blob").unwrap();
    assert_eq!(
        blob.metadata.get(ARROW_EXT_NAME_KEY).map(String::as_str),
        Some(BLOB_V2_EXT_NAME)
    );

    let len0 = engine.take_blob_one(&out, 0, "blob").await.unwrap();
    assert_eq!(len0, 3);
    let len2 = engine.take_blob_one(&out, 2, "blob").await.unwrap();
    assert_eq!(len2, 2);
}
