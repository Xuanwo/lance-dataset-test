use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::tempdir;

use bench_core::dataset::{open_dataset, DatasetReadOptions, BENCH_ROW_ID};
use bench_core::workload::DatasetName;

#[tokio::test]
async fn parquet_dir_gets_row_id() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    let batch =
        RecordBatch::try_from_iter(vec![("x", Arc::new(Int32Array::from(vec![1, 2, 3])) as _)])
            .unwrap();
    let parquet_path = data_dir.join("part-00000.parquet");
    let file = File::create(&parquet_path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let opened = open_dataset(
        DatasetName::FineWeb,
        &data_dir,
        DatasetReadOptions {
            batch_size: 1024,
            limit_rows: None,
        },
    )
    .await
    .unwrap();

    assert!(opened
        .schema
        .fields()
        .iter()
        .any(|f| f.name() == BENCH_ROW_ID));

    let mut total = 0usize;
    for batch in opened.reader {
        let batch = batch.unwrap();
        total += batch.num_rows();
        assert!(batch.schema().field_with_name(BENCH_ROW_ID).is_ok());
    }
    assert_eq!(total, 3);
}

#[tokio::test]
async fn openvid_adds_fake_blob_column() {
    std::env::set_var("OPENVID_FAKE_BLOB_BYTES", "8");

    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    let batch =
        RecordBatch::try_from_iter(vec![("x", Arc::new(Int32Array::from(vec![1, 2, 3])) as _)])
            .unwrap();
    let parquet_path = data_dir.join("part-00000.parquet");
    let file = File::create(&parquet_path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let opened = open_dataset(
        DatasetName::OpenVid,
        &data_dir,
        DatasetReadOptions {
            batch_size: 1024,
            limit_rows: None,
        },
    )
    .await
    .unwrap();

    let blob_field = opened.schema.field_with_name("video_blob").unwrap();
    assert_eq!(
        blob_field
            .metadata()
            .get(lance_arrow::BLOB_META_KEY)
            .map(|v| v.as_str()),
        Some("true")
    );

    let mut total = 0usize;
    for batch in opened.reader {
        let batch = batch.unwrap();
        total += batch.num_rows();
        let blob_idx = batch.schema().index_of("video_blob").unwrap();
        let blob = batch.column(blob_idx);
        assert_eq!(blob.len(), batch.num_rows());
    }
    assert_eq!(total, 3);
}

#[tokio::test]
async fn webdataset_tar_smoke() {
    let dir = tempdir().unwrap();
    let tar_path = dir.path().join("shard-00000.tar");
    write_test_webdataset_tar(&tar_path);
    assert_eq!(count_tar_entries(&tar_path), 2);

    let opened = open_dataset(
        DatasetName::Laion10m,
        dir.path(),
        DatasetReadOptions {
            batch_size: 4,
            limit_rows: None,
        },
    )
    .await
    .unwrap();

    let mut batches = opened.reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.len(), 1);
    let batch = batches.pop().unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert!(batch.schema().field_with_name(BENCH_ROW_ID).is_ok());
}

fn count_tar_entries(path: &Path) -> usize {
    let file = File::open(path).unwrap();
    let mut archive = tar::Archive::new(file);
    archive.entries().unwrap().count()
}

fn write_test_webdataset_tar(path: &Path) {
    let file = File::create(path).unwrap();
    let mut builder = tar::Builder::new(file);

    let caption = b"hello";
    let image = b"\x89PNG\r\n";

    let mut header = tar::Header::new_gnu();
    header.set_mode(0o644);
    header.set_mtime(0);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_size(caption.len() as u64);
    header.set_cksum();
    builder
        .append_data(&mut header, "00000000.txt", &caption[..])
        .unwrap();

    let mut header = tar::Header::new_gnu();
    header.set_mode(0o644);
    header.set_mtime(0);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_size(image.len() as u64);
    header.set_cksum();
    builder
        .append_data(&mut header, "00000000.png", &image[..])
        .unwrap();

    builder.finish().unwrap();
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    assert!(!buf.is_empty());
}
