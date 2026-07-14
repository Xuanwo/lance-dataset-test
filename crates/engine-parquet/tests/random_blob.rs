use std::sync::Arc;

use arrow_array::{
    Int32Array, LargeBinaryArray, RecordBatch, RecordBatchIterator, RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema};
use engine_parquet::{ParquetEngine, ParquetReadMode, ParquetWriterOptions, ParquetWriterProfile};
use tempfile::tempdir;

fn reader() -> Box<dyn RecordBatchReader + Send> {
    let schema = Arc::new(Schema::new(vec![
        Arc::new(Field::new("id", DataType::Int32, false)),
        Arc::new(Field::new("blob", DataType::LargeBinary, true)),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
            Arc::new(LargeBinaryArray::from_iter([
                Some(&b"zero"[..]),
                None,
                Some(&b"two-two"[..]),
                Some(&b"three"[..]),
            ])),
        ],
    )
    .unwrap();
    Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema,
    ))
}

#[tokio::test]
async fn row_selection_reads_exact_binary_value_from_default_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("default.parquet");
    let engine = ParquetEngine::new();
    engine
        .ingest(reader(), &path, ParquetWriterOptions::default())
        .await
        .unwrap();

    let opened = engine
        .open_file_with_mode(&path, ParquetReadMode::RowSelection)
        .await
        .unwrap();
    assert!(opened.has_offset_index());
    assert_eq!(opened.writer_profile(), ParquetWriterProfile::Default);
    assert_eq!(opened.row_count(), 4);
    assert_eq!(
        engine
            .read_binary_one_opened(&opened, 2, "blob")
            .await
            .unwrap(),
        Some(b"two-two".to_vec())
    );
    assert_eq!(
        engine
            .read_binary_one_opened(&opened, 1, "blob")
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn random_blob_writer_remains_native_parquet_and_readable() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("random-blob.parquet");
    let engine = ParquetEngine::new();
    let mut options = ParquetWriterOptions::random_blob(vec!["blob".to_string()]);
    options.data_page_size_limit = Some(16);
    options.max_row_group_bytes = Some(64);
    engine.ingest(reader(), &path, options).await.unwrap();

    let opened = engine
        .open_file_with_mode(&path, ParquetReadMode::RowSelection)
        .await
        .unwrap();
    assert!(opened.has_offset_index());
    assert_eq!(opened.writer_profile(), ParquetWriterProfile::RandomBlob);
    assert_eq!(
        engine
            .read_binary_one_opened(&opened, 3, "blob")
            .await
            .unwrap(),
        Some(b"three".to_vec())
    );

    let default_opened = engine.open_file(&path).await.unwrap();
    assert_eq!(
        engine
            .take_binary_one_opened(&default_opened, 0, "blob")
            .await
            .unwrap(),
        4
    );
    assert_eq!(ParquetWriterProfile::RandomBlob.as_str(), "random-blob");
}
