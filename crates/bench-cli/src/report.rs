use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs as stdfs, io};

use anyhow::{Context, Result};
use bench_core::fs;
use bench_core::metrics::Timing;
use bench_core::result::RunResult;
use bench_core::workload::{DatasetName, EngineName, Workload};

#[derive(Debug, Clone)]
struct RunKey {
    dataset: DatasetName,
    engine: EngineName,
    lance_file_version: Option<String>,
    lance_disable_compression: bool,
    lance_index_columns: Option<String>,
    workload: Workload,
}

#[derive(Debug, Clone)]
struct RunRecord {
    key: RunKey,
    result: RunResult,
    source: PathBuf,
}

pub fn run_report(results_dir: &Path, data_dir: &Path, out_path: &Path) -> Result<()> {
    let records = load_results(results_dir)?;
    let env = pick_env(&records)?;
    let results_commit_summary = summarize_result_commits(&records);
    let repo_head_commit = try_repo_head_commit();
    let dataset_rows = build_dataset_rows(data_dir)?;
    let results_root = results_dir.parent().unwrap_or(results_dir);
    let lance_layout_note = build_lance_layout_note(&records, results_root)?;

    let mut report = String::new();
    report.push_str("# Benchmark Report\n\n");

    report.push_str("## Environment\n");
    report.push_str(&format!("- OS: {} {}\n", env.os, env.arch));
    report.push_str(&format!("- CPU cores: {}\n", env.cpu_count));
    report.push_str(&format!("- Rust: {}\n", env.rustc));
    report.push_str(&format!(
        "- Commit (results): {}\n",
        results_commit_summary.primary
    ));
    if results_commit_summary.all.len() > 1 {
        report.push_str(&format!(
            "- Commits (results): {}\n",
            format_commit_summary(&results_commit_summary.all)
        ));
    }
    if let Some(head) = repo_head_commit {
        report.push_str(&format!(
            "- Commit (repo HEAD at report generation): {}\n",
            head
        ));
    }
    report.push('\n');

    report.push_str("## Datasets (Local Inputs)\n\n");
    report.push_str("| Dataset | Input path | Input size | Notes |\n");
    report.push_str("|---|---|---:|---|\n");
    for row in dataset_rows {
        report.push_str(&format!(
            "| {} | `{}` | {} | {} |\n",
            row.display_name, row.input_path, row.input_size, row.notes
        ));
    }
    report.push('\n');

    write_results_section(&mut report, &records)?;

    report.push_str("## Artifacts\n\n");
    report.push_str(&format!(
        "- JSON results: `{}`\n",
        results_dir.to_string_lossy()
    ));
    report.push_str(&format!(
        "- Engine outputs: `{}`\n\n",
        results_root.join("datasets").to_string_lossy()
    ));

    report.push_str("## Notes\n\n");
    report
        .push_str("- Some results are below 1 ms wall time; these are displayed as `0.xxx ms`.\n");
    report.push_str("- Scan workloads may be executed multiple times via `--repeats`; scan times are reported as p50/p95/p99 across repeats, while row/byte counts are reported per scan as an average across repeats.\n");
    report.push_str("- Scan workloads exclude blob columns for blob-bearing datasets; blob throughput/latency is reported under `Random Blob`.\n");
    report.push_str("- Projection and filtered scans use a stored numeric column (`bench_small_u64`) instead of a synthetic row index to avoid column-virtualization artifacts.\n");
    report.push_str("- LeRobot datasets use only the first parquet file under the `data/` subdirectory to avoid schema drift across parquet shards and meta parquet files.\n");
    report.push_str("- OpenVid uses a synthetic blob column (`video_blob`) materialized during ingestion (default `OPENVID_FAKE_BLOB_BYTES=256`).\n");
    if let Some(note) = lance_layout_note {
        report.push_str(&format!("- {note}\n"));
    }

    stdfs::write(out_path, report).with_context(|| format!("write {}", out_path.display()))?;
    Ok(())
}

fn build_lance_layout_note(records: &[RunRecord], _results_root: &Path) -> Result<Option<String>> {
    let mut parts = Vec::new();
    for dataset in dataset_iter() {
        let mut versions = lance_versions_for_layout(records, dataset, Workload::ScanFull);
        for w in [
            Workload::ScanProject,
            Workload::ScanFilterLow,
            Workload::ScanFilterHigh,
        ] {
            for v in lance_versions_for_layout(records, dataset, w) {
                if !versions.iter().any(|x| x == &v) {
                    versions.push(v);
                }
            }
        }
        versions.sort_by_key(|v| lance_file_version_order(Some(v.as_str())));

        for version in versions {
            let rec = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&version),
                false,
                None,
                Workload::ScanFull,
            )
            .or_else(|| {
                find_record(
                    records,
                    dataset,
                    EngineName::Lance,
                    Some(&version),
                    false,
                    None,
                    Workload::ScanProject,
                )
            })
            .or_else(|| {
                find_record(
                    records,
                    dataset,
                    EngineName::Lance,
                    Some(&version),
                    false,
                    None,
                    Workload::ScanFilterLow,
                )
            })
            .or_else(|| {
                find_record(
                    records,
                    dataset,
                    EngineName::Lance,
                    Some(&version),
                    false,
                    None,
                    Workload::ScanFilterHigh,
                )
            });

            let Some(rec) = rec else { continue };
            let fragments = parse_u64_param(&rec.result.meta.params, "lance_fragment_count");
            let data_files = parse_u64_param(&rec.result.meta.params, "lance_data_file_count");

            if fragments.is_none() && data_files.is_none() {
                continue;
            }

            let fragments = fragments
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".to_string());
            let data_files = data_files
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".to_string());
            parts.push(format!(
                "{}@{} fragments={} data_files={}",
                dataset_display(dataset),
                version,
                fragments,
                data_files
            ));
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    let mut note = String::new();
    note.push_str("Lance output layout (from scan runs): ");
    note.push_str(&parts.join("; "));
    Ok(Some(note))
}

fn parse_u64_param(params: &BTreeMap<String, String>, key: &str) -> Option<u64> {
    params.get(key).and_then(|v| v.parse::<u64>().ok())
}

fn failure_tag(notes: &[String]) -> Option<&'static str> {
    if notes.iter().any(|n| n.starts_with("panic:")) {
        return Some("PANIC");
    }
    if notes.iter().any(|n| n.starts_with("error:")) {
        return Some("ERROR");
    }
    None
}

fn write_results_section(out: &mut String, records: &[RunRecord]) -> Result<()> {
    out.push_str("## Results\n\n");
    out.push_str("- Primary comparison: `parquet` vs `lance`.\n\n");
    write_ingest_section(out, records)?;
    write_scan_sections(out, records)?;
    write_take_section(out, records)?;
    write_blob_section(out, records)?;
    write_evolve_section(out, records)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct EnvSummary {
    os: String,
    arch: String,
    cpu_count: u64,
    rustc: String,
}

fn pick_env(records: &[RunRecord]) -> Result<EnvSummary> {
    let record = if records.is_empty() {
        return Err(anyhow::anyhow!("no results found"));
    } else {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for r in records {
            if is_unknown_commit(&r.result.meta.git_commit) {
                continue;
            }
            *counts.entry(r.result.meta.git_commit.clone()).or_default() += 1;
        }

        if let Some((picked, _)) = counts.into_iter().max_by_key(|(_k, v)| *v) {
            records
                .iter()
                .find(|r| r.result.meta.git_commit == picked)
                .unwrap_or(&records[0])
        } else {
            &records[0]
        }
    };
    Ok(EnvSummary {
        os: record.result.meta.os.clone(),
        arch: record.result.meta.arch.clone(),
        cpu_count: record.result.meta.cpu_count,
        rustc: record.result.meta.rustc.clone(),
    })
}

#[derive(Debug, Clone)]
struct CommitSummary {
    primary: String,
    all: Vec<(String, u64)>,
}

fn summarize_result_commits(records: &[RunRecord]) -> CommitSummary {
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    for r in records {
        *counts.entry(r.result.meta.git_commit.clone()).or_default() += 1;
    }

    let mut all: Vec<(String, u64)> = counts.into_iter().collect();
    all.sort_by(|(a_commit, a_count), (b_commit, b_count)| {
        b_count.cmp(a_count).then_with(|| a_commit.cmp(b_commit))
    });

    let primary = all
        .iter()
        .find(|(c, _)| !is_unknown_commit(c))
        .map(|(c, _)| c.clone())
        .unwrap_or_else(|| "unknown".to_string());

    CommitSummary { primary, all }
}

fn format_commit_summary(all: &[(String, u64)]) -> String {
    let mut parts = Vec::new();
    for (i, (commit, count)) in all.iter().enumerate() {
        if i >= 3 {
            break;
        }
        parts.push(format!("{commit} ({count})"));
    }
    if all.len() > 3 {
        parts.push(format!("+ {} more", all.len() - 3));
    }
    parts.join(", ")
}

fn is_unknown_commit(commit: &str) -> bool {
    commit == "unknown" || commit == "no-commits"
}

fn try_repo_head_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[derive(Debug, Clone)]
struct DatasetRow {
    display_name: &'static str,
    input_path: String,
    input_size: String,
    notes: &'static str,
}

fn build_dataset_rows(data_dir: &Path) -> Result<Vec<DatasetRow>> {
    let rows = vec![
        (
            "FineWeb",
            "fineweb",
            "Single parquet: `sample/10BT/000_00000.parquet` (ingest capped by `--limit-rows 10000000`)",
        ),
        (
            "OpenVid",
            "openvid",
            "Local parquet generated from `nkp37/OpenVid-1M` CSV; fake blob column `video_blob` during ingest (`OPENVID_FAKE_BLOB_BYTES=256`)",
        ),
        ("LAION", "laion10m", "WebDataset shard prefix: `00000.tar` only"),
        (
            "LeRobot PushT",
            "lerobot-pusht",
            "Uses first parquet under `data/` subdir to avoid schema drift",
        ),
        (
            "LeRobot PushT Image",
            "lerobot-pusht_image",
            "Uses first parquet under `data/` subdir to avoid schema drift",
        ),
    ];

    let mut out = Vec::with_capacity(rows.len());
    for (display_name, dir_name, notes) in rows {
        let size = fs::dir_size_bytes(&data_dir.join(dir_name))
            .with_context(|| format!("stat input size: {}/{}", data_dir.display(), dir_name))?;
        out.push(DatasetRow {
            display_name,
            input_path: format!("data/{dir_name}"),
            input_size: fmt_bytes(size),
            notes,
        });
    }

    Ok(out)
}

fn load_results(results_dir: &Path) -> Result<Vec<RunRecord>> {
    // Keep only the newest result for each (dataset, engine, workload) tuple.
    //
    // Rationale: `results/json/` may contain multiple historical runs for the same case (including
    // renamed engines such as `lance-fragment` vs `lance_fragment`). The report must pick the
    // newest run deterministically.
    let mut best_by_key: BTreeMap<String, RunRecord> = BTreeMap::new();
    for entry in
        stdfs::read_dir(results_dir).with_context(|| format!("read {}", results_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }

        let file = stdfs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let reader = io::BufReader::new(file);
        let result: RunResult =
            serde_json::from_reader(reader).with_context(|| format!("parse {}", path.display()))?;

        let inferred = infer_key_from_filename(&path).unwrap_or_else(|| RunKey {
            dataset: result.meta.dataset,
            engine: result.meta.engine,
            lance_file_version: None,
            lance_disable_compression: false,
            lance_index_columns: None,
            workload: result.meta.workload,
        });

        let dataset = if result.meta.dataset == DatasetName::Unknown {
            inferred.dataset
        } else {
            result.meta.dataset
        };
        let engine = if result.meta.engine == EngineName::Unknown {
            inferred.engine
        } else {
            result.meta.engine
        };
        let lance_file_version = if engine == EngineName::Lance {
            Some(
                result
                    .meta
                    .params
                    .get("lance_file_version")
                    .cloned()
                    .unwrap_or_else(|| "2.0".to_string()),
            )
        } else {
            None
        };
        let lance_disable_compression = if engine == EngineName::Lance {
            result
                .meta
                .params
                .get("lance_disable_compression")
                .map(|v| v == "true")
                .unwrap_or(false)
        } else {
            false
        };
        let lance_index_columns = if engine == EngineName::Lance {
            result.meta.params.get("lance_index_columns").cloned()
        } else {
            None
        };
        let key = RunKey {
            dataset,
            engine,
            lance_file_version,
            lance_disable_compression,
            lance_index_columns,
            workload: result.meta.workload,
        };

        let rec = RunRecord {
            key,
            result,
            source: path,
        };
        if rec.key.engine == EngineName::LanceFragment {
            continue;
        }

        let key_str = format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            rec.key.dataset,
            rec.key.engine,
            rec.key.lance_file_version,
            rec.key.lance_disable_compression,
            rec.key.lance_index_columns,
            rec.key.workload
        );

        match best_by_key.get(&key_str) {
            None => {
                best_by_key.insert(key_str, rec);
            }
            Some(existing) => {
                let a = rec.result.meta.started_at_unix_ms;
                let b = existing.result.meta.started_at_unix_ms;
                let pick_new = a > b || (a == b && rec.source > existing.source);
                if pick_new {
                    best_by_key.insert(key_str, rec);
                }
            }
        }
    }

    let mut records: Vec<RunRecord> = best_by_key.into_values().collect();
    records.sort_by_key(|r| {
        (
            dataset_order(r.key.dataset),
            engine_order(r.key.engine),
            lance_file_version_order(r.key.lance_file_version.as_deref()),
            lance_disable_compression_order(r.key.lance_disable_compression),
            lance_index_columns_order(r.key.lance_index_columns.as_deref()),
            r.key.lance_index_columns.clone().unwrap_or_default(),
            workload_order(r.key.workload),
            r.source.clone(),
        )
    });
    Ok(records)
}

fn infer_key_from_filename(path: &Path) -> Option<RunKey> {
    let stem = path.file_stem()?.to_str()?;
    let mut parts = stem.split('.').collect::<Vec<_>>();
    if parts.len() < 3 {
        return None;
    }
    let engine = match parts.pop()? {
        "lance" => EngineName::Lance,
        "lance_fragment" | "lance-fragment" => EngineName::LanceFragment,
        "parquet" => EngineName::Parquet,
        _ => EngineName::Unknown,
    };
    let workload = match parts.pop()? {
        "ingest" => Workload::Ingest,
        "scan-full" => Workload::ScanFull,
        "scan-project" => Workload::ScanProject,
        "scan-filter-low" => Workload::ScanFilterLow,
        "scan-filter-high" => Workload::ScanFilterHigh,
        "take" => Workload::RandomTake,
        "blob" => Workload::RandomBlob,
        "evolve" => Workload::EvolutionBackfill,
        "size" => Workload::Size,
        _ => Workload::Size,
    };
    // Prefer the last remaining segment as dataset id when stems are longer than 3 parts.
    let dataset = match parts.pop()? {
        "laion10m" => DatasetName::Laion10m,
        "openvid" => DatasetName::OpenVid,
        "fineweb" => DatasetName::FineWeb,
        "lerobot_pusht" => DatasetName::LeRobotPushT,
        "lerobot_pusht_image" => DatasetName::LeRobotPushTImage,
        "lerobot-pusht" => DatasetName::LeRobotPushT,
        "lerobot-pusht_image" => DatasetName::LeRobotPushTImage,
        _ => DatasetName::Unknown,
    };
    Some(RunKey {
        dataset,
        engine,
        lance_file_version: None,
        lance_disable_compression: false,
        lance_index_columns: None,
        workload,
    })
}

fn write_ingest_section(out: &mut String, records: &[RunRecord]) -> Result<()> {
    out.push_str("### Ingest\n\n");
    out.push_str("- Query shape: read Arrow `RecordBatch` stream from dataset adapter; write into engine-native storage.\n\n");
    out.push_str("| Dataset | Engine | Rows | Time | Rows/s | Bytes | Bytes/s | Output size |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");

    for dataset in dataset_iter() {
        if let Some(ingest) = find_record(
            records,
            dataset,
            EngineName::Parquet,
            None,
            false,
            None,
            Workload::Ingest,
        ) {
            let size = find_record(
                records,
                dataset,
                EngineName::Parquet,
                None,
                false,
                None,
                Workload::Size,
            );
            write_ingest_row(out, dataset, ingest, size)?;
        }

        for case in lance_cases_for_workload(records, dataset, Workload::Ingest) {
            let ingest = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&case.file_version),
                case.disable_compression,
                case.index_columns.as_deref(),
                Workload::Ingest,
            );
            let size = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&case.file_version),
                case.disable_compression,
                case.index_columns.as_deref(),
                Workload::Size,
            );
            let Some(ingest) = ingest else {
                continue;
            };
            write_ingest_row(out, dataset, ingest, size)?;
        }
    }
    out.push('\n');
    Ok(())
}

fn write_ingest_row(
    out: &mut String,
    dataset: DatasetName,
    ingest: &RunRecord,
    size: Option<&RunRecord>,
) -> Result<()> {
    let rows = ingest.result.rows.unwrap_or_default();
    let bytes = ingest.result.bytes.unwrap_or_default();
    let time_us = timing_us(&ingest.result.timing);
    if let Some(tag) = failure_tag(&ingest.result.notes) {
        out.push_str(&format!(
            "| {} | {} | {} | {tag} | - | {} | - | - |\n",
            dataset_display(dataset),
            engine_display(
                ingest.key.engine,
                ingest.key.lance_file_version.as_deref(),
                ingest.key.lance_disable_compression,
                ingest.key.lance_index_columns.as_deref(),
            ),
            rows,
            fmt_bytes(bytes),
        ));
        return Ok(());
    }
    let (rows_s, bytes_s) = throughput(rows, bytes, time_us);
    let output_size = size
        .and_then(|r| r.result.bytes)
        .map(fmt_bytes)
        .unwrap_or_else(|| "-".to_string());

    out.push_str(&format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
        dataset_display(dataset),
        engine_display(
            ingest.key.engine,
            ingest.key.lance_file_version.as_deref(),
            ingest.key.lance_disable_compression,
            ingest.key.lance_index_columns.as_deref(),
        ),
        rows,
        fmt_time_ms(time_us),
        fmt_rows_rate(rows_s),
        fmt_bytes(bytes),
        fmt_bytes_rate(bytes_s),
        output_size
    ));
    Ok(())
}

fn write_scan_sections(out: &mut String, records: &[RunRecord]) -> Result<()> {
    out.push_str("### Scan\n\n");

    write_scan_table(out, records, "Full scan", Workload::ScanFull)?;
    write_scan_table(out, records, "Projection scan", Workload::ScanProject)?;
    write_scan_table(
        out,
        records,
        "Filtered scan (low selectivity)",
        Workload::ScanFilterLow,
    )?;
    write_scan_table(
        out,
        records,
        "Filtered scan (high selectivity)",
        Workload::ScanFilterHigh,
    )?;

    Ok(())
}

fn write_scan_table(
    out: &mut String,
    records: &[RunRecord],
    title: &str,
    workload: Workload,
) -> Result<()> {
    out.push_str(&format!("#### {title}\n\n"));
    out.push_str(&format!(
        "- Query shape: {}\n\n",
        query_shape_for_workload(workload)
    ));
    out.push_str("| Dataset | Engine | Rows | p50 | p95 | p99 | Rows/s | Bytes | Bytes/s |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");

    for dataset in dataset_iter() {
        if let Some(rec) = find_record(
            records,
            dataset,
            EngineName::Parquet,
            None,
            false,
            None,
            workload,
        ) {
            write_scan_row(out, dataset, rec)?;
        }

        for case in lance_cases_for_workload(records, dataset, workload) {
            let Some(rec) = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&case.file_version),
                case.disable_compression,
                case.index_columns.as_deref(),
                workload,
            ) else {
                continue;
            };
            write_scan_row(out, dataset, rec)?;
        }
    }
    out.push('\n');
    Ok(())
}

fn write_scan_row(out: &mut String, dataset: DatasetName, rec: &RunRecord) -> Result<()> {
    if let Some(tag) = failure_tag(&rec.result.notes) {
        out.push_str(&format!(
            "| {} | {} | - | {tag} | - | - | - | - | - |\n",
            dataset_display(dataset),
            engine_display(
                rec.key.engine,
                rec.key.lance_file_version.as_deref(),
                rec.key.lance_disable_compression,
                rec.key.lance_index_columns.as_deref(),
            ),
        ));
        return Ok(());
    }
    let repeats = scan_repeats(&rec.result.meta.params);
    let rows_total = rec.result.rows.unwrap_or_default();
    let bytes_total = rec.result.bytes.unwrap_or_default();
    let time_us_total = timing_us(&rec.result.timing);

    let rows = if repeats > 1 {
        rows_total / (repeats as u64)
    } else {
        rows_total
    };
    let bytes = if repeats > 1 {
        bytes_total / (repeats as u64)
    } else {
        bytes_total
    };

    let (p50_us, p95_us, p99_us) = rec
        .result
        .latency
        .as_ref()
        .map(|l| (Some(l.p50_us), Some(l.p95_us), Some(l.p99_us)))
        .unwrap_or((None, None, None));
    let (p50, p95, p99) = match (p50_us, p95_us, p99_us) {
        (Some(p50), Some(p95), Some(p99)) => (
            fmt_time_ms_u64(p50),
            fmt_time_ms_u64(p95),
            fmt_time_ms_u64(p99),
        ),
        _ => {
            let time_us_avg = if repeats > 1 {
                (time_us_total as f64) / (repeats as f64)
            } else {
                time_us_total as f64
            };
            (
                fmt_time_ms_f64(time_us_avg),
                "-".to_string(),
                "-".to_string(),
            )
        }
    };

    if time_us_total == 0 {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | - | {} | - |\n",
            dataset_display(dataset),
            engine_display(
                rec.key.engine,
                rec.key.lance_file_version.as_deref(),
                rec.key.lance_disable_compression,
                rec.key.lance_index_columns.as_deref(),
            ),
            rows,
            p50,
            p95,
            p99,
            fmt_bytes(bytes),
        ));
        return Ok(());
    }

    let (rows_s, bytes_s) = throughput(rows_total, bytes_total, time_us_total);
    out.push_str(&format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
        dataset_display(dataset),
        engine_display(
            rec.key.engine,
            rec.key.lance_file_version.as_deref(),
            rec.key.lance_disable_compression,
            rec.key.lance_index_columns.as_deref(),
        ),
        rows,
        p50,
        p95,
        p99,
        fmt_rows_rate(rows_s),
        fmt_bytes(bytes),
        fmt_bytes_rate(bytes_s),
    ));
    Ok(())
}

fn write_take_section(out: &mut String, records: &[RunRecord]) -> Result<()> {
    out.push_str("### Random Take (1000 iters)\n\n");
    out.push_str("- Query shape: Open the dataset/file once; K times pick a random `row_offset`; projection=`bench_small_u64`; Lance uses `Dataset::take(&[row_offset], projection)`; Parquet reads the row group containing `row_offset` with a projected `ParquetRecordBatchReader`.\n\n");
    out.push_str("| Dataset | Engine | Time | p50 | p95 | p99 |\n");
    out.push_str("|---|---|---:|---:|---:|---:|\n");

    for dataset in dataset_iter() {
        if let Some(rec) = find_record(
            records,
            dataset,
            EngineName::Parquet,
            None,
            false,
            None,
            Workload::RandomTake,
        ) {
            write_take_row(out, dataset, rec)?;
        }

        for case in lance_cases_for_workload(records, dataset, Workload::RandomTake) {
            let Some(rec) = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&case.file_version),
                case.disable_compression,
                case.index_columns.as_deref(),
                Workload::RandomTake,
            ) else {
                continue;
            };
            write_take_row(out, dataset, rec)?;
        }
    }
    out.push('\n');
    Ok(())
}

fn write_take_row(out: &mut String, dataset: DatasetName, rec: &RunRecord) -> Result<()> {
    if let Some(tag) = failure_tag(&rec.result.notes) {
        out.push_str(&format!(
            "| {} | {} | {tag} | - | - | - |\n",
            dataset_display(dataset),
            engine_display(
                rec.key.engine,
                rec.key.lance_file_version.as_deref(),
                rec.key.lance_disable_compression,
                rec.key.lance_index_columns.as_deref(),
            ),
        ));
        return Ok(());
    }
    let time_us = timing_us(&rec.result.timing);
    let Some(lat) = &rec.result.latency else {
        return Ok(());
    };
    out.push_str(&format!(
        "| {} | {} | {} | {} us | {} us | {} us |\n",
        dataset_display(dataset),
        engine_display(
            rec.key.engine,
            rec.key.lance_file_version.as_deref(),
            rec.key.lance_disable_compression,
            rec.key.lance_index_columns.as_deref(),
        ),
        fmt_time_ms(time_us),
        lat.p50_us,
        lat.p95_us,
        lat.p99_us
    ));
    Ok(())
}

fn write_blob_section(out: &mut String, records: &[RunRecord]) -> Result<()> {
    out.push_str("### Random Blob (1000 iters)\n\n");
    out.push_str("- Query shape: K times pick a random `row_offset`; Lance calls `Dataset::take_blobs_by_indices(&[row_offset], column)` and reads the returned bytes; Parquet projects `[column]` for the row group containing `row_offset` and measures the returned binary length.\n\n");
    out.push_str("| Dataset | Engine | Total bytes | Time | p50 | p95 | p99 |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|\n");

    for dataset in dataset_iter() {
        if let Some(rec) = find_record(
            records,
            dataset,
            EngineName::Parquet,
            None,
            false,
            None,
            Workload::RandomBlob,
        ) {
            write_blob_row(out, dataset, rec)?;
        }

        for case in lance_cases_for_workload(records, dataset, Workload::RandomBlob) {
            let Some(rec) = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&case.file_version),
                case.disable_compression,
                case.index_columns.as_deref(),
                Workload::RandomBlob,
            ) else {
                continue;
            };
            write_blob_row(out, dataset, rec)?;
        }
    }
    out.push('\n');

    out.push_str(
        "- OpenVid blob column: `video_blob` (synthetic, default 256 bytes/row during ingest; configurable via `OPENVID_FAKE_BLOB_BYTES`).\n",
    );
    out.push_str("- LAION blob column: `image` (decoded bytes from WebDataset shards).\n\n");
    Ok(())
}

fn write_blob_row(out: &mut String, dataset: DatasetName, rec: &RunRecord) -> Result<()> {
    if rec.result.notes.iter().any(|n| n.starts_with("panic:")) {
        let total_bytes = rec.result.bytes.unwrap_or_default();
        out.push_str(&format!(
            "| {} | {} | {} | PANIC | - | - | - |\n",
            dataset_display(dataset),
            engine_display(
                rec.key.engine,
                rec.key.lance_file_version.as_deref(),
                rec.key.lance_disable_compression,
                rec.key.lance_index_columns.as_deref(),
            ),
            fmt_bytes(total_bytes),
        ));
        return Ok(());
    }
    if rec.result.notes.iter().any(|n| n.starts_with("error:")) {
        let total_bytes = rec.result.bytes.unwrap_or_default();
        out.push_str(&format!(
            "| {} | {} | {} | ERROR | - | - | - |\n",
            dataset_display(dataset),
            engine_display(
                rec.key.engine,
                rec.key.lance_file_version.as_deref(),
                rec.key.lance_disable_compression,
                rec.key.lance_index_columns.as_deref(),
            ),
            fmt_bytes(total_bytes),
        ));
        return Ok(());
    }

    let Some(lat) = &rec.result.latency else {
        return Ok(());
    };
    let total_bytes = rec.result.bytes.unwrap_or_default();
    let time_us = timing_us(&rec.result.timing);
    out.push_str(&format!(
        "| {} | {} | {} | {} | {} us | {} us | {} us |\n",
        dataset_display(dataset),
        engine_display(
            rec.key.engine,
            rec.key.lance_file_version.as_deref(),
            rec.key.lance_disable_compression,
            rec.key.lance_index_columns.as_deref(),
        ),
        fmt_bytes(total_bytes),
        fmt_time_ms(time_us),
        lat.p50_us,
        lat.p95_us,
        lat.p99_us
    ));
    Ok(())
}

fn write_evolve_section(out: &mut String, records: &[RunRecord]) -> Result<()> {
    out.push_str("### Evolution / Backfill\n\n");
    out.push_str("- Query shape: Lance uses `Dataset::open(uri).add_columns(SqlExpressions([(new_column, expr)]))`; Parquet rewrites the file by scanning all rows and writing a new file with the derived column.\n\n");
    out.push_str(
        "| Dataset | Engine | Base size | Evolved size | Bytes written | Time | Notes |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|---|\n");

    for dataset in dataset_iter() {
        if let Some(rec) = find_record(
            records,
            dataset,
            EngineName::Parquet,
            None,
            false,
            None,
            Workload::EvolutionBackfill,
        ) {
            write_evolve_row(out, dataset, rec)?;
        }

        for case in lance_cases_for_workload(records, dataset, Workload::EvolutionBackfill) {
            let Some(rec) = find_record(
                records,
                dataset,
                EngineName::Lance,
                Some(&case.file_version),
                case.disable_compression,
                case.index_columns.as_deref(),
                Workload::EvolutionBackfill,
            ) else {
                continue;
            };
            write_evolve_row(out, dataset, rec)?;
        }
    }
    out.push('\n');
    Ok(())
}

fn write_evolve_row(out: &mut String, dataset: DatasetName, rec: &RunRecord) -> Result<()> {
    if let Some(tag) = failure_tag(&rec.result.notes) {
        out.push_str(&format!(
            "| {} | {} | - | - | - | {tag} | {} |\n",
            dataset_display(dataset),
            engine_display(
                rec.key.engine,
                rec.key.lance_file_version.as_deref(),
                rec.key.lance_disable_compression,
                rec.key.lance_index_columns.as_deref(),
            ),
            rec.result.notes.join("; "),
        ));
        return Ok(());
    }
    let time_us = timing_us(&rec.result.timing);
    let base_size_bytes = rec
        .result
        .meta
        .params
        .get("base_size_bytes")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let evolved_size_bytes = rec
        .result
        .meta
        .params
        .get("evolved_size_bytes")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let bytes_written = rec.result.bytes.unwrap_or(0);
    let notes = evolve_notes(&rec.result.meta.params);
    out.push_str(&format!(
        "| {} | {} | {} | {} | {} | {} | {} |\n",
        dataset_display(dataset),
        engine_display(
            rec.key.engine,
            rec.key.lance_file_version.as_deref(),
            rec.key.lance_disable_compression,
            rec.key.lance_index_columns.as_deref(),
        ),
        fmt_bytes(base_size_bytes),
        fmt_bytes(evolved_size_bytes),
        fmt_bytes(bytes_written),
        fmt_time_ms(time_us),
        notes
    ));
    Ok(())
}

fn evolve_notes(params: &BTreeMap<String, String>) -> String {
    if let Some(expr) = params.get("expr") {
        return format!("expr={expr}");
    }
    if let Some(rewrite_out) = params.get("rewrite_out") {
        return format!("rewrite_out={rewrite_out}");
    }
    "-".to_string()
}

fn find_record<'a>(
    records: &'a [RunRecord],
    dataset: DatasetName,
    engine: EngineName,
    lance_file_version: Option<&str>,
    lance_disable_compression: bool,
    lance_index_columns: Option<&str>,
    workload: Workload,
) -> Option<&'a RunRecord> {
    records.iter().rev().find(|r| {
        r.key.dataset == dataset
            && r.key.engine == engine
            && r.key.workload == workload
            && match engine {
                EngineName::Lance => {
                    r.key.lance_file_version.as_deref() == lance_file_version
                        && r.key.lance_disable_compression == lance_disable_compression
                        && r.key.lance_index_columns.as_deref() == lance_index_columns
                }
                _ => {
                    !lance_disable_compression
                        && lance_file_version.is_none()
                        && lance_index_columns.is_none()
                }
            }
    })
}

#[derive(Debug, Clone)]
struct LanceCase {
    file_version: String,
    disable_compression: bool,
    index_columns: Option<String>,
}

fn lance_cases_for_workload(
    records: &[RunRecord],
    dataset: DatasetName,
    workload: Workload,
) -> Vec<LanceCase> {
    let mut out: Vec<LanceCase> = Vec::new();
    for r in records {
        if r.key.dataset != dataset
            || r.key.engine != EngineName::Lance
            || r.key.workload != workload
        {
            continue;
        }
        let Some(v) = r.key.lance_file_version.as_deref() else {
            continue;
        };
        let case = LanceCase {
            file_version: v.to_string(),
            disable_compression: r.key.lance_disable_compression,
            index_columns: r.key.lance_index_columns.clone(),
        };
        if !out.iter().any(|x| {
            x.file_version == case.file_version
                && x.disable_compression == case.disable_compression
                && x.index_columns == case.index_columns
        }) {
            out.push(case);
        }
    }
    out.sort_by(|a, b| {
        lance_file_version_order(Some(a.file_version.as_str()))
            .cmp(&lance_file_version_order(Some(b.file_version.as_str())))
            .then_with(|| {
                lance_disable_compression_order(a.disable_compression)
                    .cmp(&lance_disable_compression_order(b.disable_compression))
            })
            .then_with(|| {
                lance_index_columns_order(a.index_columns.as_deref())
                    .cmp(&lance_index_columns_order(b.index_columns.as_deref()))
            })
            .then_with(|| {
                a.index_columns
                    .clone()
                    .unwrap_or_default()
                    .cmp(&b.index_columns.clone().unwrap_or_default())
            })
    });
    out
}

fn lance_versions_for_layout(
    records: &[RunRecord],
    dataset: DatasetName,
    workload: Workload,
) -> Vec<String> {
    let mut out = Vec::new();
    for r in records {
        if r.key.dataset == dataset
            && r.key.engine == EngineName::Lance
            && r.key.workload == workload
            && !r.key.lance_disable_compression
            && r.key.lance_index_columns.is_none()
        {
            if let Some(v) = r.key.lance_file_version.as_deref() {
                if !out.iter().any(|x| x == v) {
                    out.push(v.to_string());
                }
            }
        }
    }
    out.sort_by_key(|v| lance_file_version_order(Some(v.as_str())));
    out
}

fn engine_display(
    engine: EngineName,
    lance_file_version: Option<&str>,
    lance_disable_compression: bool,
    lance_index_columns: Option<&str>,
) -> String {
    match engine {
        EngineName::Lance => {
            let mut base = format!("lance@{}", lance_file_version.unwrap_or("unknown"));
            if lance_disable_compression {
                base = format!("{base}+no-compression");
            }
            if let Some(cols) = lance_index_columns {
                if cols.is_empty() {
                    base
                } else {
                    format!("{base}+idx({cols})")
                }
            } else {
                base
            }
        }
        EngineName::Parquet => "parquet".to_string(),
        EngineName::LanceFragment => "lance-fragment".to_string(),
        EngineName::Unknown => "unknown".to_string(),
    }
}

fn timing_us(t: &Timing) -> u128 {
    if t.wall_time_us != 0 {
        t.wall_time_us
    } else {
        t.wall_time_ms.saturating_mul(1000)
    }
}

fn scan_repeats(params: &BTreeMap<String, String>) -> u32 {
    params
        .get("repeats")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1)
}

fn throughput(rows: u64, bytes: u64, time_us: u128) -> (f64, f64) {
    if time_us == 0 {
        return (0.0, 0.0);
    }
    let secs = (time_us as f64) / 1_000_000.0;
    ((rows as f64) / secs, (bytes as f64) / secs)
}

fn dataset_iter() -> [DatasetName; 5] {
    [
        DatasetName::FineWeb,
        DatasetName::OpenVid,
        DatasetName::Laion10m,
        DatasetName::LeRobotPushT,
        DatasetName::LeRobotPushTImage,
    ]
}

fn query_shape_for_workload(workload: Workload) -> &'static str {
    match workload {
        Workload::ScanFull => {
            "Parquet: open file once, then per scan build a projected `ParquetRecordBatchReader` and count rows (filter cases apply the predicate in-memory); Lance: open `Dataset` once, build `Dataset::scan().project(all_non_blob_cols?)` (and `.filter(sql_predicate?)` for filter cases), then per scan build a DataFusion plan via `create_plan()` and `execute(..)`; each timed scan run is preceded by one untimed warmup scan with the same query shape; optional `--repeats N` reuses the opened dataset/file and scan configuration, runs N timed scans, and the report shows p50/p95/p99 across repeats."
        }
        Workload::ScanProject => {
            "Same as full scan but projection is `project([\"bench_small_u64\"])`; each timed scan run is preceded by one untimed warmup scan with the same query shape; optional `--repeats N` reuses the opened dataset/file and scan configuration, runs N timed scans, and the report shows p50/p95/p99 across repeats."
        }
        Workload::ScanFilterLow | Workload::ScanFilterHigh => {
            "Same as projection scan plus predicate `bench_small_u64 < THRESHOLD` (Parquet applies the predicate in-memory after decoding; Lance uses a SQL filter string with predicate pushdown); each timed scan run is preceded by one untimed warmup scan with the same query shape; optional `--repeats N` reuses the opened dataset/file and scan configuration, runs N timed scans, and the report shows p50/p95/p99 across repeats."
        }
        _ => "-",
    }
}

fn dataset_display(dataset: DatasetName) -> &'static str {
    match dataset {
        DatasetName::FineWeb => "FineWeb",
        DatasetName::OpenVid => "OpenVid",
        DatasetName::Laion10m => "LAION",
        DatasetName::LeRobotPushT => "LeRobot PushT",
        DatasetName::LeRobotPushTImage => "LeRobot PushT Image",
        DatasetName::Unknown => "Unknown",
    }
}

fn dataset_order(dataset: DatasetName) -> u8 {
    match dataset {
        DatasetName::FineWeb => 0,
        DatasetName::OpenVid => 1,
        DatasetName::Laion10m => 2,
        DatasetName::LeRobotPushT => 3,
        DatasetName::LeRobotPushTImage => 4,
        DatasetName::Unknown => 255,
    }
}

fn engine_order(engine: EngineName) -> u8 {
    match engine {
        EngineName::Parquet => 0,
        EngineName::Lance => 1,
        EngineName::LanceFragment => 255,
        EngineName::Unknown => 255,
    }
}

fn lance_file_version_order(version: Option<&str>) -> u8 {
    let Some(version) = version else {
        return 0;
    };
    match version {
        "2.0" | "0.3" | "stable" => 0,
        "2.1" => 1,
        "2.2" => 2,
        _ => 254,
    }
}

fn lance_index_columns_order(cols: Option<&str>) -> u8 {
    match cols {
        None => 0,
        Some(_) => 1,
    }
}

fn lance_disable_compression_order(disabled: bool) -> u8 {
    if disabled {
        1
    } else {
        0
    }
}

fn workload_order(workload: Workload) -> u8 {
    match workload {
        Workload::Ingest => 0,
        Workload::ScanFull => 1,
        Workload::ScanProject => 2,
        Workload::ScanFilterLow => 3,
        Workload::ScanFilterHigh => 4,
        Workload::RandomTake => 5,
        Workload::RandomBlob => 6,
        Workload::EvolutionBackfill => 7,
        Workload::Size => 8,
    }
}

fn fmt_time_ms(time_us: u128) -> String {
    if time_us < 1000 {
        format!("{:.3} ms", (time_us as f64) / 1000.0)
    } else {
        format!("{} ms", time_us / 1000)
    }
}

fn fmt_time_ms_u64(time_us: u64) -> String {
    fmt_time_ms(time_us as u128)
}

fn fmt_time_ms_f64(time_us: f64) -> String {
    if time_us < 1000.0 {
        format!("{:.3} ms", time_us / 1000.0)
    } else {
        format!("{} ms", (time_us / 1000.0).floor() as u128)
    }
}

fn fmt_rows_rate(rows_per_sec: f64) -> String {
    if rows_per_sec <= 0.0 {
        return "-".to_string();
    }
    let (value, unit) = if rows_per_sec >= 1_000_000_000.0 {
        (rows_per_sec / 1_000_000_000.0, "Brows/s")
    } else if rows_per_sec >= 1_000_000.0 {
        (rows_per_sec / 1_000_000.0, "Mrows/s")
    } else if rows_per_sec >= 1_000.0 {
        (rows_per_sec / 1_000.0, "Krows/s")
    } else {
        (rows_per_sec, "rows/s")
    };
    format!("{value:.2} {unit}")
}

fn fmt_bytes_rate(bytes_per_sec: f64) -> String {
    if bytes_per_sec <= 0.0 {
        return "-".to_string();
    }
    format!("{}/s", fmt_bytes(bytes_per_sec.round() as u64))
}

fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}
