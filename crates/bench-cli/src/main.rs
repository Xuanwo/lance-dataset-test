use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use arrow_array::builder::StringBuilder;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use bench_core::dataset::BENCH_SMALL_U64;
use bench_core::metrics::LatencySummary;
use bench_core::metrics::WallTimer;
use bench_core::query::Filter;
use bench_core::result::{RunMetadata, RunResult};
use bench_core::workload::{DatasetName, EngineName, Workload};
use bench_core::{dataset, fs};
use clap::{Parser, ValueEnum};
use hdrhistogram::Histogram;
use parquet::arrow::ArrowWriter;
use rand::prelude::*;
use tracing_subscriber::EnvFilter;

mod plot;
mod report;

fn maybe_add_lance_index_params(
    engine: EngineName,
    path: &str,
    params: &mut BTreeMap<String, String>,
) {
    if engine != EngineName::Lance {
        return;
    }
    if path.contains(".index") {
        params.insert(
            "lance_index_columns".to_string(),
            BENCH_SMALL_U64.to_string(),
        );
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum EngineArg {
    Lance,
    LanceFragment,
    Parquet,
}

impl From<EngineArg> for EngineName {
    fn from(value: EngineArg) -> Self {
        match value {
            EngineArg::Lance => EngineName::Lance,
            EngineArg::LanceFragment => EngineName::LanceFragment,
            EngineArg::Parquet => EngineName::Parquet,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum DatasetArg {
    Laion10m,
    OpenVid,
    FineWeb,
    LeRobotPushT,
    LeRobotPushTImage,
}

impl From<DatasetArg> for DatasetName {
    fn from(value: DatasetArg) -> Self {
        match value {
            DatasetArg::Laion10m => DatasetName::Laion10m,
            DatasetArg::OpenVid => DatasetName::OpenVid,
            DatasetArg::FineWeb => DatasetName::FineWeb,
            DatasetArg::LeRobotPushT => DatasetName::LeRobotPushT,
            DatasetArg::LeRobotPushTImage => DatasetName::LeRobotPushTImage,
        }
    }
}

impl DatasetArg {
    fn default_out_dir(&self) -> &'static str {
        match self {
            DatasetArg::Laion10m => "data/laion10m",
            DatasetArg::OpenVid => "data/openvid",
            DatasetArg::FineWeb => "data/fineweb",
            DatasetArg::LeRobotPushT => "data/lerobot-pusht",
            DatasetArg::LeRobotPushTImage => "data/lerobot-pusht_image",
        }
    }

    fn id(&self) -> &'static str {
        match self {
            DatasetArg::Laion10m => "laion10m",
            DatasetArg::OpenVid => "openvid",
            DatasetArg::FineWeb => "fineweb",
            DatasetArg::LeRobotPushT => "lerobot-pusht",
            DatasetArg::LeRobotPushTImage => "lerobot-pusht_image",
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum LanceWriteModeArg {
    Create,
    Append,
    Overwrite,
}

#[derive(Debug, Parser)]
#[command(name = "bench")]
struct Cli {
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Download {
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        out: Option<String>,
    },
    #[command(name = "prepare-openvid")]
    PrepareOpenVid {
        #[arg(long)]
        input: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 1_000_000)]
        limit_rows: u64,
        #[arg(long, default_value_t = 8192)]
        batch_size: usize,
    },
    Smoke {
        engine: EngineArg,
    },
    Run {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        workload: WorkloadArg,
        #[arg(long)]
        path: String,
        #[arg(long)]
        out: String,
        #[arg(long)]
        projection: Option<String>,
        #[arg(long, default_value_t = 1)]
        repeats: u32,
        #[arg(long, default_value_t = 1000)]
        iters: u64,
        #[arg(long)]
        column: Option<String>,
        #[arg(long)]
        new_column: Option<String>,
        #[arg(long)]
        evolve_out: Option<String>,
        #[arg(long)]
        expr: Option<String>,
    },
    Ingest {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        input: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 8192)]
        batch_size: usize,
        #[arg(long)]
        limit_rows: Option<u64>,
        #[arg(long)]
        lance_file_version: Option<String>,
        #[arg(long)]
        lance_index_columns: Option<String>,
        #[arg(long)]
        lance_write_mode: Option<LanceWriteModeArg>,
        #[arg(long)]
        lance_max_rows_per_file: Option<usize>,
        #[arg(long)]
        lance_max_bytes_per_file: Option<usize>,
        #[arg(long, default_value_t = 0)]
        row_id_offset: u64,
        #[arg(long)]
        result_out: Option<String>,
    },
    #[command(name = "ingest-lance-versions")]
    IngestLanceVersions {
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        input: String,
        #[arg(long, default_value = "results")]
        out_root: String,
        #[arg(long, default_value = "2.0,2.1,2.2")]
        lance_file_versions: String,
        #[arg(long, default_value_t = false)]
        with_index: bool,
        #[arg(long, default_value = "bench_small_u64")]
        index_columns: String,
        #[arg(long, default_value_t = 8192)]
        batch_size: usize,
        #[arg(long)]
        limit_rows: Option<u64>,
        #[arg(long, default_value_t = default_results_dir())]
        results_dir: String,
    },
    #[command(name = "create-index")]
    CreateIndex {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        path: String,
        #[arg(long)]
        columns: String,
    },
    ScanFull {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        path: String,
        #[arg(long, default_value_t = 1)]
        repeats: u32,
        #[arg(long)]
        result_out: Option<String>,
    },
    Scan {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        path: String,
        #[arg(long)]
        mode: ScanModeArg,
        #[arg(long)]
        projection: Option<String>,
        #[arg(long)]
        limit_rows: Option<u64>,
        #[arg(long)]
        row_offset: Option<u64>,
        #[arg(long, default_value_t = 1)]
        repeats: u32,
        #[arg(long)]
        result_out: Option<String>,
    },
    Take {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        path: String,
        #[arg(long, default_value_t = 1000)]
        iters: u64,
        #[arg(long)]
        projection: Option<String>,
        #[arg(long)]
        result_out: Option<String>,
    },
    Blob {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        path: String,
        #[arg(long)]
        column: String,
        #[arg(long, default_value_t = 1000)]
        iters: u64,
        #[arg(long)]
        result_out: Option<String>,
    },
    Evolve {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        dataset: DatasetArg,
        #[arg(long)]
        path: String,
        #[arg(long)]
        new_column: String,
        #[arg(long)]
        out: Option<String>,
        #[arg(long)]
        expr: Option<String>,
        #[arg(long)]
        result_out: Option<String>,
    },
    Size {
        #[arg(long)]
        engine: Option<EngineArg>,
        #[arg(long)]
        dataset: Option<DatasetArg>,
        #[arg(long)]
        path: String,
        #[arg(long)]
        result_out: Option<String>,
    },
    Report {
        #[arg(long, default_value_t = default_results_dir())]
        results_dir: String,
        #[arg(long, default_value = "data")]
        data_dir: String,
        #[arg(long, default_value = "REPORT.md")]
        out: String,
    },
    Plot {
        #[arg(long, default_value_t = default_results_dir())]
        results_dir: String,
        #[arg(long, default_value_t = default_plots_dir())]
        out_dir: String,
    },
    Suite {
        #[arg(long, default_value = "data")]
        data_dir: String,
        #[arg(long, default_value = "results")]
        out_root: String,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        dataset: Vec<DatasetArg>,
        #[arg(long, default_value = "2.0,2.1,2.2")]
        lance_file_versions: String,
        #[arg(long, default_value_t = 8192)]
        batch_size: usize,
        #[arg(long)]
        limit_rows: Option<u64>,
        #[arg(long, default_value_t = 3)]
        scan_repeats: u32,
        #[arg(long, default_value_t = 1000)]
        take_iters: u64,
        #[arg(long, default_value_t = 1000)]
        blob_iters: u64,
        #[arg(long, default_value = "bench_derived_u64")]
        evolve_new_column: String,
        #[arg(long)]
        evolve_expr: Option<String>,
    },
    Schema {
        #[arg(long)]
        engine: EngineArg,
        #[arg(long)]
        path: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum ScanModeArg {
    Full,
    Project,
    FilterLow,
    FilterHigh,
}

#[derive(Debug, Clone, ValueEnum)]
enum WorkloadArg {
    ScanFull,
    ScanProject,
    ScanFilterLow,
    ScanFilterHigh,
    RandomTake,
    RandomBlob,
    EvolutionBackfill,
    Size,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Download { dataset, out } => {
            let out = out.unwrap_or_else(|| dataset.default_out_dir().to_string());
            run_download(dataset, out).await
        }
        Command::PrepareOpenVid {
            input,
            out,
            limit_rows,
            batch_size,
        } => run_prepare_openvid(input, out, Some(limit_rows), batch_size),
        Command::Smoke { engine } => run_smoke(cli.seed, engine).await,
        Command::Run {
            engine,
            dataset,
            workload,
            path,
            out,
            projection,
            repeats,
            iters,
            column,
            new_column,
            evolve_out,
            expr,
        } => {
            run_dispatch(
                cli.seed, engine, dataset, workload, path, out, projection, repeats, iters, column,
                new_column, evolve_out, expr,
            )
            .await
        }
        Command::Ingest {
            engine,
            dataset,
            input,
            out,
            batch_size,
            limit_rows,
            lance_file_version,
            lance_index_columns,
            lance_write_mode,
            lance_max_rows_per_file,
            lance_max_bytes_per_file,
            row_id_offset,
            result_out,
        } => {
            run_ingest(
                cli.seed,
                engine,
                dataset,
                input,
                out,
                batch_size,
                limit_rows,
                lance_file_version,
                lance_index_columns,
                lance_write_mode,
                lance_max_rows_per_file,
                lance_max_bytes_per_file,
                row_id_offset,
                result_out,
            )
            .await
        }
        Command::IngestLanceVersions {
            dataset,
            input,
            out_root,
            lance_file_versions,
            with_index,
            index_columns,
            batch_size,
            limit_rows,
            results_dir,
        } => {
            run_ingest_lance_versions(
                cli.seed,
                dataset,
                input,
                out_root,
                lance_file_versions,
                with_index,
                index_columns,
                batch_size,
                limit_rows,
                results_dir,
            )
            .await
        }
        Command::CreateIndex {
            engine,
            path,
            columns,
        } => run_create_index(engine, path, columns).await,
        Command::ScanFull {
            engine,
            dataset,
            path,
            repeats,
            result_out,
        } => run_scan_full(cli.seed, engine, dataset, path, repeats, result_out).await,
        Command::Scan {
            engine,
            dataset,
            path,
            mode,
            projection,
            limit_rows,
            row_offset,
            repeats,
            result_out,
        } => {
            run_scan(
                cli.seed,
                engine,
                dataset,
                path,
                mode,
                projection,
                limit_rows,
                row_offset,
                repeats,
                result_out,
            )
            .await
        }
        Command::Take {
            engine,
            dataset,
            path,
            iters,
            projection,
            result_out,
        } => {
            run_take(
                cli.seed, engine, dataset, path, iters, projection, result_out,
            )
            .await
        }
        Command::Blob {
            engine,
            dataset,
            path,
            column,
            iters,
            result_out,
        } => run_blob(cli.seed, engine, dataset, path, column, iters, result_out).await,
        Command::Evolve {
            engine,
            dataset,
            path,
            new_column,
            out,
            expr,
            result_out,
        } => {
            run_evolve(
                cli.seed, engine, dataset, path, new_column, out, expr, result_out,
            )
            .await
        }
        Command::Size {
            engine,
            dataset,
            path,
            result_out,
        } => run_size(cli.seed, engine, dataset, path, result_out).await,
        Command::Report {
            results_dir,
            data_dir,
            out,
        } => report::run_report(
            std::path::Path::new(&results_dir),
            std::path::Path::new(&data_dir),
            std::path::Path::new(&out),
        ),
        Command::Plot {
            results_dir,
            out_dir,
        } => plot::run_plots(
            std::path::Path::new(&results_dir),
            std::path::Path::new(&out_dir),
        ),
        Command::Suite {
            data_dir,
            out_root,
            run_id,
            dataset,
            lance_file_versions,
            batch_size,
            limit_rows,
            scan_repeats,
            take_iters,
            blob_iters,
            evolve_new_column,
            evolve_expr,
        } => {
            run_suite(
                cli.seed,
                data_dir,
                out_root,
                run_id,
                dataset,
                lance_file_versions,
                batch_size,
                limit_rows,
                scan_repeats,
                take_iters,
                blob_iters,
                evolve_new_column,
                evolve_expr,
            )
            .await
        }
        Command::Schema { engine, path } => run_schema(engine, path).await,
    }
}

fn default_results_dir() -> String {
    default_run_subdir("json", "results/json")
}

fn default_plots_dir() -> String {
    default_run_subdir("plots", "results/plots")
}

fn default_run_subdir(subdir: &str, fallback: &str) -> String {
    let root = std::path::Path::new("results");
    let current = root.join("_current_run");
    let run_id = std::fs::read_to_string(&current)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let Some(run_id) = run_id else {
        return fallback.to_string();
    };
    root.join("runs")
        .join(run_id)
        .join(subdir)
        .to_string_lossy()
        .to_string()
}

fn generate_run_id() -> String {
    let output = std::process::Command::new("date")
        .args(["-u", "+%Y%m%dT%H%M%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    output.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| format!("unix{}", d.as_secs()))
            .unwrap_or_else(|_| "unix0".to_string())
    })
}

async fn run_ingest_lance_versions(
    seed: u64,
    dataset: DatasetArg,
    input: String,
    out_root: String,
    lance_file_versions: String,
    with_index: bool,
    index_columns: String,
    batch_size: usize,
    limit_rows: Option<u64>,
    results_dir: String,
) -> Result<()> {
    let versions: Vec<String> = lance_file_versions
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .collect();
    if versions.is_empty() {
        anyhow::bail!("--lance-file-versions must not be empty");
    }

    tokio::fs::create_dir_all(&results_dir).await?;
    tokio::fs::create_dir_all(&out_root).await?;

    let index_columns = index_columns
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>()
        .join(",");
    let index_columns = if with_index {
        Some(index_columns)
    } else {
        None
    };

    for version in versions {
        let tag = version.replace('.', "_");
        let out = format!("{}/{}.lance.v{}", out_root, dataset.id(), tag);
        let result_out = format!(
            "{}/ingest.{}.lance.v{}.json",
            results_dir,
            dataset.id(),
            tag
        );

        run_ingest(
            seed,
            EngineArg::Lance,
            dataset.clone(),
            input.clone(),
            out,
            batch_size,
            limit_rows,
            Some(version.clone()),
            None,
            None,
            None,
            None,
            0,
            Some(result_out),
        )
        .await?;

        if let Some(index_columns) = &index_columns {
            let out = format!("{}/{}.lance.v{}.index", out_root, dataset.id(), tag);
            let result_out = format!(
                "{}/ingest.{}.lance.v{}.index.json",
                results_dir,
                dataset.id(),
                tag
            );
            run_ingest(
                seed,
                EngineArg::Lance,
                dataset.clone(),
                input.clone(),
                out,
                batch_size,
                limit_rows,
                Some(version),
                Some(index_columns.clone()),
                None,
                None,
                None,
                0,
                Some(result_out),
            )
            .await?;
        }
    }

    Ok(())
}

fn parse_versions_csv(input: &str) -> Result<Vec<String>> {
    let versions: Vec<String> = input
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .collect();
    if versions.is_empty() {
        anyhow::bail!("version list must not be empty");
    }
    Ok(versions)
}

fn default_suite_limit_rows(dataset: DatasetArg, override_limit_rows: Option<u64>) -> Option<u64> {
    if override_limit_rows.is_some() {
        return override_limit_rows;
    }
    match dataset {
        DatasetArg::FineWeb => Some(10_000_000),
        DatasetArg::OpenVid => Some(1_000_000),
        DatasetArg::Laion10m => Some(200_000),
        DatasetArg::LeRobotPushT | DatasetArg::LeRobotPushTImage => None,
    }
}

async fn run_suite(
    seed: u64,
    data_dir: String,
    out_root: String,
    run_id: Option<String>,
    datasets: Vec<DatasetArg>,
    lance_file_versions: String,
    batch_size: usize,
    limit_rows: Option<u64>,
    scan_repeats: u32,
    take_iters: u64,
    blob_iters: u64,
    evolve_new_column: String,
    evolve_expr: Option<String>,
) -> Result<()> {
    async fn write_failure_result(
        seed: u64,
        engine: EngineName,
        dataset: DatasetName,
        workload: Workload,
        started_at_unix_ms: u128,
        mut params: BTreeMap<String, String>,
        out: String,
        err: String,
    ) -> Result<()> {
        params.insert("error".to_string(), err.clone());
        let notes = if err.contains("panicked") || err.contains("JoinError") {
            vec![format!("panic: {err}")]
        } else {
            vec![format!("error: {err}")]
        };
        let result = RunResult {
            meta: make_meta(engine, dataset, workload, seed, started_at_unix_ms, params),
            timing: WallTimer::start().stop(),
            rows: None,
            bytes: None,
            latency: None,
            notes,
        };
        write_result(&result, Some(out)).await
    }

    async fn run_case(
        seed: u64,
        engine: EngineName,
        dataset: DatasetName,
        workload: Workload,
        started_at_unix_ms: u128,
        params: BTreeMap<String, String>,
        out: String,
        fut: impl std::future::Future<Output = Result<()>> + Send + 'static,
    ) -> Result<()> {
        match tokio::spawn(fut).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => {
                write_failure_result(
                    seed,
                    engine,
                    dataset,
                    workload,
                    started_at_unix_ms,
                    params,
                    out,
                    err.to_string(),
                )
                .await
            }
            Err(join_err) => {
                write_failure_result(
                    seed,
                    engine,
                    dataset,
                    workload,
                    started_at_unix_ms,
                    params,
                    out,
                    join_err.to_string(),
                )
                .await
            }
        }
    }

    let run_id = run_id.unwrap_or_else(generate_run_id);
    let out_root = std::path::PathBuf::from(out_root);
    let run_root = out_root.join("runs").join(&run_id);
    let datasets_root = run_root.join("datasets");
    let json_dir = run_root.join("json");
    let plots_dir = run_root.join("plots");
    let tmp_dir = run_root.join("tmp");

    tokio::fs::create_dir_all(&datasets_root).await?;
    tokio::fs::create_dir_all(&json_dir).await?;
    tokio::fs::create_dir_all(&plots_dir).await?;
    tokio::fs::create_dir_all(&tmp_dir).await?;

    tokio::fs::write(out_root.join("_current_run"), format!("{run_id}\n")).await?;

    let datasets = if datasets.is_empty() {
        vec![
            DatasetArg::FineWeb,
            DatasetArg::OpenVid,
            DatasetArg::Laion10m,
            DatasetArg::LeRobotPushT,
            DatasetArg::LeRobotPushTImage,
        ]
    } else {
        datasets
    };

    let lance_versions = parse_versions_csv(&lance_file_versions)?;

    for dataset in datasets {
        let dataset_id = dataset.id().to_string();
        let dataset_name: DatasetName = dataset.clone().into();
        let input_path = std::path::Path::new(&data_dir).join(dataset.id());
        let input = input_path.to_string_lossy().to_string();
        let limit_rows = default_suite_limit_rows(dataset.clone(), limit_rows);

        let dataset_out_dir = datasets_root.join(&dataset_id);
        tokio::fs::create_dir_all(&dataset_out_dir).await?;

        let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        let lance_variants: Vec<(String, String)> = lance_versions
            .iter()
            .map(|v| (v.clone(), v.replace('.', "_")))
            .collect();

        let parquet_out = dataset_out_dir.join(format!("{dataset_id}.parquet"));
        let parquet_ingest_out = json_dir
            .join(format!("ingest.{dataset_id}.parquet.json"))
            .to_string_lossy()
            .to_string();
        let mut parquet_ingest_params = BTreeMap::new();
        parquet_ingest_params.insert("input".to_string(), input.clone());
        parquet_ingest_params.insert(
            "dataset_out".to_string(),
            parquet_out.to_string_lossy().to_string(),
        );
        parquet_ingest_params.insert("batch_size".to_string(), batch_size.to_string());
        if let Some(limit) = limit_rows {
            parquet_ingest_params.insert("limit_rows".to_string(), limit.to_string());
        }
        run_case(
            seed,
            EngineName::Parquet,
            dataset_name,
            Workload::Ingest,
            started_at_unix_ms,
            parquet_ingest_params,
            parquet_ingest_out.clone(),
            {
                let input = input.clone();
                let out = parquet_out.to_string_lossy().to_string();
                let dataset_for_task = dataset.clone();
                let result_out = parquet_ingest_out.clone();
                async move {
                    run_ingest(
                        seed,
                        EngineArg::Parquet,
                        dataset_for_task,
                        input,
                        out,
                        batch_size,
                        limit_rows,
                        None,
                        None,
                        None,
                        None,
                        None,
                        0,
                        Some(result_out),
                    )
                    .await
                }
            },
        )
        .await?;

        for (version, tag) in &lance_variants {
            let lance_out = dataset_out_dir.join(format!("{dataset_id}.lance.v{tag}"));
            let lance_ingest_out = json_dir
                .join(format!("ingest.{dataset_id}.lance.v{tag}.json"))
                .to_string_lossy()
                .to_string();
            let mut params = BTreeMap::new();
            params.insert("input".to_string(), input.clone());
            params.insert(
                "dataset_out".to_string(),
                lance_out.to_string_lossy().to_string(),
            );
            params.insert("batch_size".to_string(), batch_size.to_string());
            params.insert("lance_file_version_requested".to_string(), version.clone());
            params.insert("lance_file_version".to_string(), version.clone());
            if let Some(limit) = limit_rows {
                params.insert("limit_rows".to_string(), limit.to_string());
            }
            run_case(
                seed,
                EngineName::Lance,
                dataset_name,
                Workload::Ingest,
                started_at_unix_ms,
                params,
                lance_ingest_out.clone(),
                {
                    let input = input.clone();
                    let out = lance_out.to_string_lossy().to_string();
                    let version = version.clone();
                    let dataset_for_task = dataset.clone();
                    let result_out = lance_ingest_out.clone();
                    async move {
                        run_ingest(
                            seed,
                            EngineArg::Lance,
                            dataset_for_task,
                            input,
                            out,
                            batch_size,
                            limit_rows,
                            Some(version),
                            None,
                            None,
                            None,
                            None,
                            0,
                            Some(result_out),
                        )
                        .await
                    }
                },
            )
            .await?;
        }

        let parquet_size_out = json_dir
            .join(format!("size.{dataset_id}.parquet.json"))
            .to_string_lossy()
            .to_string();
        let mut parquet_size_params = BTreeMap::new();
        parquet_size_params.insert(
            "path".to_string(),
            parquet_out.to_string_lossy().to_string(),
        );
        run_case(
            seed,
            EngineName::Parquet,
            dataset_name,
            Workload::Size,
            started_at_unix_ms,
            parquet_size_params,
            parquet_size_out.clone(),
            {
                let path = parquet_out.to_string_lossy().to_string();
                let dataset_for_task = dataset.clone();
                let result_out = parquet_size_out.clone();
                async move {
                    run_size(
                        seed,
                        Some(EngineArg::Parquet),
                        Some(dataset_for_task),
                        path,
                        Some(result_out),
                    )
                    .await
                }
            },
        )
        .await?;

        for (version, tag) in &lance_variants {
            let lance_out = dataset_out_dir.join(format!("{dataset_id}.lance.v{tag}"));
            let lance_size_out = json_dir
                .join(format!("size.{dataset_id}.lance.v{tag}.json"))
                .to_string_lossy()
                .to_string();
            let mut params = BTreeMap::new();
            params.insert("path".to_string(), lance_out.to_string_lossy().to_string());
            params.insert("lance_file_version".to_string(), version.clone());
            run_case(
                seed,
                EngineName::Lance,
                dataset_name,
                Workload::Size,
                started_at_unix_ms,
                params,
                lance_size_out.clone(),
                {
                    let path = lance_out.to_string_lossy().to_string();
                    let dataset_for_task = dataset.clone();
                    let result_out = lance_size_out.clone();
                    async move {
                        run_size(
                            seed,
                            Some(EngineArg::Lance),
                            Some(dataset_for_task),
                            path,
                            Some(result_out),
                        )
                        .await
                    }
                },
            )
            .await?;
        }

        for (engine, engine_tag, path) in [
            (
                EngineArg::Parquet,
                "parquet".to_string(),
                parquet_out.to_string_lossy().to_string(),
            ),
        ] {
            let engine_name: EngineName = engine.clone().into();
            for (mode, mode_tag) in [
                (ScanModeArg::Full, "scan-full"),
                (ScanModeArg::Project, "scan-project"),
                (ScanModeArg::FilterLow, "scan-filter-low"),
                (ScanModeArg::FilterHigh, "scan-filter-high"),
            ] {
                let scan_out = json_dir
                    .join(format!("{mode_tag}.{dataset_id}.{engine_tag}.json"))
                    .to_string_lossy()
                    .to_string();
                let mut params = BTreeMap::new();
                params.insert("path".to_string(), path.clone());
                params.insert("repeats".to_string(), scan_repeats.to_string());
                match mode {
                    ScanModeArg::FilterLow => {
                        params.insert(
                            "filter".to_string(),
                            "bench_small_u64 < 184467440737095516".to_string(),
                        );
                    }
                    ScanModeArg::FilterHigh => {
                        params.insert(
                            "filter".to_string(),
                            "bench_small_u64 < 9223372036854775807".to_string(),
                        );
                    }
                    _ => {}
                }
                run_case(
                    seed,
                    engine_name,
                    dataset_name,
                    match mode {
                        ScanModeArg::Full => Workload::ScanFull,
                        ScanModeArg::Project => Workload::ScanProject,
                        ScanModeArg::FilterLow => Workload::ScanFilterLow,
                        ScanModeArg::FilterHigh => Workload::ScanFilterHigh,
                    },
                    started_at_unix_ms,
                    params,
                    scan_out.clone(),
                    {
                        let path = path.clone();
                        let dataset_for_task = dataset.clone();
                        let engine_for_task = engine.clone();
                        let mode_for_task = mode.clone();
                        let result_out = scan_out.clone();
                        async move {
                            run_scan(
                                seed,
                                engine_for_task,
                                dataset_for_task,
                                path,
                                mode_for_task,
                                None,
                                None,
                                None,
                                scan_repeats,
                                Some(result_out),
                            )
                            .await
                        }
                    },
                )
                .await?;
            }

            let take_out = json_dir
                .join(format!("take.{dataset_id}.{engine_tag}.json"))
                .to_string_lossy()
                .to_string();
            let mut params = BTreeMap::new();
            params.insert("path".to_string(), path.clone());
            params.insert("iters".to_string(), take_iters.to_string());
            run_case(
                seed,
                engine_name,
                dataset_name,
                Workload::RandomTake,
                started_at_unix_ms,
                params,
                take_out.clone(),
                {
                    let path = path.clone();
                    let dataset_for_task = dataset.clone();
                    let engine_for_task = engine.clone();
                    let result_out = take_out.clone();
                    async move {
                        run_take(
                            seed,
                            engine_for_task,
                            dataset_for_task,
                            path,
                            take_iters,
                            None,
                            Some(result_out),
                        )
                        .await
                    }
                },
            )
            .await?;

            let blob_cols = blob_columns(dataset_name);
            if let Some(&col) = blob_cols.first() {
                let blob_out = json_dir
                    .join(format!("blob.{dataset_id}.{engine_tag}.json"))
                    .to_string_lossy()
                    .to_string();
                let mut params = BTreeMap::new();
                params.insert("path".to_string(), path.clone());
                params.insert("iters".to_string(), blob_iters.to_string());
                params.insert("column".to_string(), col.to_string());
                run_case(
                    seed,
                    engine_name,
                    dataset_name,
                    Workload::RandomBlob,
                    started_at_unix_ms,
                    params,
                    blob_out.clone(),
                    {
                        let path = path.clone();
                        let col = col.to_string();
                        let dataset_for_task = dataset.clone();
                        let engine_for_task = engine.clone();
                        let result_out = blob_out.clone();
                        async move {
                            run_blob(
                                seed,
                                engine_for_task,
                                dataset_for_task,
                                path,
                                col,
                                blob_iters,
                                Some(result_out),
                            )
                            .await
                        }
                    },
                )
                .await?;
            }

            let evolve_out = match engine {
                EngineArg::Parquet => Some(
                    tmp_dir
                        .join(format!("{dataset_id}.{engine_tag}.evolved.parquet"))
                        .to_string_lossy()
                        .to_string(),
                ),
                _ => None,
            };
            let evolve_out_json = json_dir
                .join(format!("evolve.{dataset_id}.{engine_tag}.json"))
                .to_string_lossy()
                .to_string();
            let mut params = BTreeMap::new();
            params.insert("path".to_string(), path.clone());
            params.insert("new_column".to_string(), evolve_new_column.clone());
            if let Some(expr) = &evolve_expr {
                params.insert("expr".to_string(), expr.clone());
            }
            if let Some(rewrite_out) = &evolve_out {
                params.insert("rewrite_out".to_string(), rewrite_out.clone());
            }
            run_case(
                seed,
                engine_name,
                dataset_name,
                Workload::EvolutionBackfill,
                started_at_unix_ms,
                params,
                evolve_out_json.clone(),
                {
                    let path = path.clone();
                    let evolve_new_column = evolve_new_column.clone();
                    let evolve_out = evolve_out.clone();
                    let evolve_expr = evolve_expr.clone();
                    let dataset_for_task = dataset.clone();
                    let engine_for_task = engine.clone();
                    let result_out = evolve_out_json.clone();
                    async move {
                        run_evolve(
                            seed,
                            engine_for_task,
                            dataset_for_task,
                            path,
                            evolve_new_column,
                            evolve_out,
                            evolve_expr,
                            Some(result_out),
                        )
                        .await
                    }
                },
            )
            .await?;
        }

        for (version, tag) in &lance_variants {
            let path = dataset_out_dir
                .join(format!("{dataset_id}.lance.v{tag}"))
                .to_string_lossy()
                .to_string();
            let engine_tag = format!("lance.v{tag}");

            for (mode, mode_tag) in [
                (ScanModeArg::Full, "scan-full"),
                (ScanModeArg::Project, "scan-project"),
                (ScanModeArg::FilterLow, "scan-filter-low"),
                (ScanModeArg::FilterHigh, "scan-filter-high"),
            ] {
                let scan_out = json_dir
                    .join(format!("{mode_tag}.{dataset_id}.{engine_tag}.json"))
                    .to_string_lossy()
                    .to_string();
                let mut params = BTreeMap::new();
                params.insert("path".to_string(), path.clone());
                params.insert("repeats".to_string(), scan_repeats.to_string());
                params.insert("lance_file_version".to_string(), version.clone());
                match mode {
                    ScanModeArg::FilterLow => {
                        params.insert(
                            "filter".to_string(),
                            "bench_small_u64 < 184467440737095516".to_string(),
                        );
                    }
                    ScanModeArg::FilterHigh => {
                        params.insert(
                            "filter".to_string(),
                            "bench_small_u64 < 9223372036854775807".to_string(),
                        );
                    }
                    _ => {}
                }
                run_case(
                    seed,
                    EngineName::Lance,
                    dataset_name,
                    match mode {
                        ScanModeArg::Full => Workload::ScanFull,
                        ScanModeArg::Project => Workload::ScanProject,
                        ScanModeArg::FilterLow => Workload::ScanFilterLow,
                        ScanModeArg::FilterHigh => Workload::ScanFilterHigh,
                    },
                    started_at_unix_ms,
                    params,
                    scan_out.clone(),
                    {
                        let path = path.clone();
                        let dataset_for_task = dataset.clone();
                        let mode_for_task = mode.clone();
                        let result_out = scan_out.clone();
                        async move {
                            run_scan(
                                seed,
                                EngineArg::Lance,
                                dataset_for_task,
                                path,
                                mode_for_task,
                                None,
                                None,
                                None,
                                scan_repeats,
                                Some(result_out),
                            )
                            .await
                        }
                    },
                )
                .await?;
            }

            let take_out = json_dir
                .join(format!("take.{dataset_id}.{engine_tag}.json"))
                .to_string_lossy()
                .to_string();
            let mut params = BTreeMap::new();
            params.insert("path".to_string(), path.clone());
            params.insert("iters".to_string(), take_iters.to_string());
            params.insert("lance_file_version".to_string(), version.clone());
            run_case(
                seed,
                EngineName::Lance,
                dataset_name,
                Workload::RandomTake,
                started_at_unix_ms,
                params,
                take_out.clone(),
                {
                    let path = path.clone();
                    let dataset_for_task = dataset.clone();
                    let result_out = take_out.clone();
                    async move {
                        run_take(
                            seed,
                            EngineArg::Lance,
                            dataset_for_task,
                            path,
                            take_iters,
                            None,
                            Some(result_out),
                        )
                        .await
                    }
                },
            )
            .await?;

            let blob_cols = blob_columns(dataset_name);
            if let Some(&col) = blob_cols.first() {
                let blob_out = json_dir
                    .join(format!("blob.{dataset_id}.{engine_tag}.json"))
                    .to_string_lossy()
                    .to_string();
                let mut params = BTreeMap::new();
                params.insert("path".to_string(), path.clone());
                params.insert("iters".to_string(), blob_iters.to_string());
                params.insert("column".to_string(), col.to_string());
                params.insert("lance_file_version".to_string(), version.clone());
                run_case(
                    seed,
                    EngineName::Lance,
                    dataset_name,
                    Workload::RandomBlob,
                    started_at_unix_ms,
                    params,
                    blob_out.clone(),
                    {
                        let path = path.clone();
                        let col = col.to_string();
                        let dataset_for_task = dataset.clone();
                        let result_out = blob_out.clone();
                        async move {
                            run_blob(
                                seed,
                                EngineArg::Lance,
                                dataset_for_task,
                                path,
                                col,
                                blob_iters,
                                Some(result_out),
                            )
                            .await
                        }
                    },
                )
                .await?;
            }

            let evolve_out_json = json_dir
                .join(format!("evolve.{dataset_id}.{engine_tag}.json"))
                .to_string_lossy()
                .to_string();
            let mut params = BTreeMap::new();
            params.insert("path".to_string(), path.clone());
            params.insert("new_column".to_string(), evolve_new_column.clone());
            params.insert("lance_file_version".to_string(), version.clone());
            if let Some(expr) = &evolve_expr {
                params.insert("expr".to_string(), expr.clone());
            }
            run_case(
                seed,
                EngineName::Lance,
                dataset_name,
                Workload::EvolutionBackfill,
                started_at_unix_ms,
                params,
                evolve_out_json.clone(),
                {
                    let path = path.clone();
                    let evolve_new_column = evolve_new_column.clone();
                    let evolve_expr = evolve_expr.clone();
                    let dataset_for_task = dataset.clone();
                    let result_out = evolve_out_json.clone();
                    async move {
                        run_evolve(
                            seed,
                            EngineArg::Lance,
                            dataset_for_task,
                            path,
                            evolve_new_column,
                            None,
                            evolve_expr,
                            Some(result_out),
                        )
                        .await
                    }
                },
            )
            .await?;
        }
    }

    Ok(())
}

async fn run_schema(engine: EngineArg, path: String) -> Result<()> {
    let path = std::path::Path::new(&path);
    let fields = match engine {
        EngineArg::Parquet => {
            engine_parquet::ParquetEngine::new()
                .schema_field_names(path)
                .await?
        }
        EngineArg::Lance | EngineArg::LanceFragment => {
            engine_lance::LanceEngine::new()
                .schema_field_names(path)
                .await?
        }
    };
    for f in fields {
        println!("{f}");
    }
    Ok(())
}

async fn run_smoke(seed: u64, engine: EngineArg) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let timer = WallTimer::start();
    match engine.clone() {
        EngineArg::Lance => engine_lance::LanceEngine::new().smoke_check().await?,
        EngineArg::LanceFragment => engine_lance::LanceEngine::new().smoke_check().await?,
        EngineArg::Parquet => engine_parquet::ParquetEngine::new().smoke_check().await?,
    }
    let timing = timer.stop();

    let engine_name: EngineName = engine.into();
    let mut params = BTreeMap::new();
    params.insert("smoke".to_string(), "true".to_string());
    let result = RunResult {
        meta: make_meta(
            engine_name,
            DatasetName::LeRobotPushT,
            Workload::Ingest,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: None,
        bytes: None,
        latency: None,
        notes: vec!["Smoke check only".to_string()],
    };
    write_result(&result, None).await?;
    Ok(())
}

fn rustc_version() -> String {
    if let Some(v) = option_env!("BENCH_RUSTC_VERSION") {
        let v = v.trim();
        if !v.is_empty() {
            return v.to_string();
        }
    }
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

async fn run_download(dataset: DatasetArg, out: String) -> Result<()> {
    std::fs::create_dir_all(&out)?;

    let hf_cli = std::env::var("HUGGINGFACE_CLI").unwrap_or_else(|_| "hf".to_string());
    let mut cmd = std::process::Command::new(hf_cli);
    if std::env::var_os("HF_HUB_DISABLE_XET").is_none() {
        cmd.env("HF_HUB_DISABLE_XET", "1");
    }
    if std::env::var_os("HF_HUB_DOWNLOAD_TIMEOUT").is_none() {
        cmd.env("HF_HUB_DOWNLOAD_TIMEOUT", "600");
    }
    if std::env::var_os("HF_HUB_ETAG_TIMEOUT").is_none() {
        cmd.env("HF_HUB_ETAG_TIMEOUT", "60");
    }
    match dataset {
        DatasetArg::Laion10m => {
            let shard_count = std::env::var("LAION_SHARDS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(1);
            if shard_count < 1 {
                anyhow::bail!("LAION_SHARDS must be >= 1 (got {shard_count})");
            }
            cmd.args(["download", "Leonardo6/laion10m"]);
            cmd.arg(".gitattributes");
            for i in 0..shard_count {
                cmd.arg(format!("{:05}.tar", i));
            }
            cmd.args(["--repo-type", "dataset"]);
            cmd.args(["--max-workers", "1"]);
            cmd.args(["--local-dir", &out]);

            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf download failed with status: {status}");
            }
        }
        DatasetArg::FineWeb => {
            cmd.args(["download", "HuggingFaceFW/fineweb"]);
            cmd.arg("sample/10BT/000_00000.parquet");
            cmd.args(["--repo-type", "dataset"]);
            cmd.args(["--max-workers", "1"]);
            cmd.args(["--local-dir", &out]);

            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf download failed with status: {status}");
            }
        }
        DatasetArg::LeRobotPushT => {
            cmd.args(["download", "lerobot/pusht"]);
            cmd.args(["--repo-type", "dataset"]);
            cmd.args(["--max-workers", "1"]);
            cmd.args(["--local-dir", &out]);

            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf download failed with status: {status}");
            }
        }
        DatasetArg::LeRobotPushTImage => {
            cmd.args(["download", "lerobot/pusht_image"]);
            cmd.args(["--repo-type", "dataset"]);
            cmd.args(["--max-workers", "1"]);
            cmd.args(["--local-dir", &out]);

            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf download failed with status: {status}");
            }
        }
        DatasetArg::OpenVid => {
            let target_rows = std::env::var("OPENVID_ROWS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1_000_000);
            let raw_dir = PathBuf::from(&out).join("_raw");
            let csv_rel_path = PathBuf::from("data/train/OpenVid-1M.csv");
            std::fs::create_dir_all(&raw_dir)?;

            cmd.args(["download", "nkp37/OpenVid-1M"]);
            cmd.arg(csv_rel_path.to_string_lossy().to_string());
            cmd.args(["--repo-type", "dataset"]);
            cmd.args(["--max-workers", "1"]);
            cmd.args(["--local-dir", raw_dir.to_string_lossy().as_ref()]);

            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf download failed with status: {status}");
            }

            let input_csv = raw_dir.join(&csv_rel_path);
            run_prepare_openvid(
                input_csv.to_string_lossy().to_string(),
                out.clone(),
                Some(target_rows),
                8192,
            )?;
            let _ = std::fs::remove_dir_all(&raw_dir);
        }
    }
    Ok(())
}

fn run_prepare_openvid(
    input: String,
    out: String,
    limit_rows: Option<u64>,
    batch_size: usize,
) -> Result<()> {
    let out_dir = PathBuf::from(out);
    std::fs::create_dir_all(&out_dir)?;

    let input_file = File::open(&input)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(input_file));
    let headers = reader.headers()?.clone();

    let mut field_names: Vec<String> = Vec::with_capacity(headers.len());
    for (idx, name) in headers.iter().enumerate() {
        let name = name.trim();
        if name.is_empty() {
            field_names.push(format!("col_{idx}"));
        } else {
            field_names.push(name.to_string());
        }
    }

    let fields: Vec<Field> = field_names
        .iter()
        .map(|name| Field::new(name, DataType::Utf8, true))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let tmp_path = out_dir.join("openvid.parquet.tmp");
    let out_path = out_dir.join("openvid.parquet");
    let output = File::create(&tmp_path)?;
    let mut writer = ArrowWriter::try_new(output, schema.clone(), None)?;

    let mut builders: Vec<StringBuilder> = (0..field_names.len())
        .map(|_| StringBuilder::new())
        .collect();
    let mut rows_in_batch: usize = 0;
    let mut total_rows: u64 = 0;

    for record in reader.records() {
        let record = record?;
        for (col_idx, builder) in builders.iter_mut().enumerate() {
            let value = record.get(col_idx);
            match value {
                None => builder.append_null(),
                Some(v) if v.is_empty() => builder.append_null(),
                Some(v) => builder.append_value(v),
            }
        }

        rows_in_batch += 1;
        total_rows += 1;

        if let Some(limit) = limit_rows {
            if total_rows >= limit {
                break;
            }
        }

        if rows_in_batch >= batch_size {
            flush_openvid_batch(&schema, &mut writer, &mut builders)?;
            rows_in_batch = 0;
        }
    }

    if rows_in_batch > 0 {
        flush_openvid_batch(&schema, &mut writer, &mut builders)?;
    }

    writer.close()?;
    let _ = std::fs::remove_file(&out_path);
    std::fs::rename(tmp_path, out_path)?;
    Ok(())
}

fn flush_openvid_batch(
    schema: &Arc<Schema>,
    writer: &mut ArrowWriter<File>,
    builders: &mut [StringBuilder],
) -> Result<()> {
    let arrays: Vec<ArrayRef> = builders
        .iter_mut()
        .map(|b| Arc::new(b.finish()) as ArrayRef)
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    writer.write(&batch)?;
    Ok(())
}

async fn run_dispatch(
    seed: u64,
    engine: EngineArg,
    dataset: DatasetArg,
    workload: WorkloadArg,
    path: String,
    out: String,
    projection: Option<String>,
    repeats: u32,
    iters: u64,
    column: Option<String>,
    new_column: Option<String>,
    evolve_out: Option<String>,
    expr: Option<String>,
) -> Result<()> {
    if matches!(engine, EngineArg::LanceFragment)
        && !matches!(
            workload,
            WorkloadArg::ScanFull
                | WorkloadArg::ScanProject
                | WorkloadArg::ScanFilterLow
                | WorkloadArg::ScanFilterHigh
        )
    {
        anyhow::bail!("engine lance-fragment supports scan workloads only");
    }
    match workload {
        WorkloadArg::ScanFull => {
            run_scan(
                seed,
                engine,
                dataset,
                path,
                ScanModeArg::Full,
                projection,
                None,
                None,
                repeats,
                Some(out),
            )
            .await
        }
        WorkloadArg::ScanProject => {
            run_scan(
                seed,
                engine,
                dataset,
                path,
                ScanModeArg::Project,
                projection,
                None,
                None,
                repeats,
                Some(out),
            )
            .await
        }
        WorkloadArg::ScanFilterLow => {
            run_scan(
                seed,
                engine,
                dataset,
                path,
                ScanModeArg::FilterLow,
                projection,
                None,
                None,
                repeats,
                Some(out),
            )
            .await
        }
        WorkloadArg::ScanFilterHigh => {
            run_scan(
                seed,
                engine,
                dataset,
                path,
                ScanModeArg::FilterHigh,
                projection,
                None,
                None,
                repeats,
                Some(out),
            )
            .await
        }
        WorkloadArg::RandomTake => {
            run_take(seed, engine, dataset, path, iters, projection, Some(out)).await
        }
        WorkloadArg::RandomBlob => {
            let column =
                column.ok_or_else(|| anyhow::anyhow!("--column is required for random-blob"))?;
            run_blob(seed, engine, dataset, path, column, iters, Some(out)).await
        }
        WorkloadArg::EvolutionBackfill => {
            let new_column = new_column.ok_or_else(|| {
                anyhow::anyhow!("--new-column is required for evolution-backfill")
            })?;
            run_evolve(
                seed,
                engine,
                dataset,
                path,
                new_column,
                evolve_out,
                expr,
                Some(out),
            )
            .await
        }
        WorkloadArg::Size => run_size(seed, Some(engine), Some(dataset), path, Some(out)).await,
    }
}

async fn run_create_index(engine: EngineArg, path: String, columns: String) -> Result<()> {
    let columns: Vec<String> = columns
        .split(',')
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .collect();
    if columns.is_empty() {
        anyhow::bail!("--columns must not be empty");
    }

    let path = PathBuf::from(path);
    match engine {
        EngineArg::Lance | EngineArg::LanceFragment => {
            engine_lance::LanceEngine::new()
                .create_scalar_btree_indices(&path, &columns)
                .await?;
            Ok(())
        }
        EngineArg::Parquet => {
            anyhow::bail!("create-index supports lance datasets only")
        }
    }
}

async fn run_ingest(
    seed: u64,
    engine: EngineArg,
    dataset_arg: DatasetArg,
    input: String,
    out: String,
    batch_size: usize,
    limit_rows: Option<u64>,
    lance_file_version: Option<String>,
    lance_index_columns: Option<String>,
    lance_write_mode: Option<LanceWriteModeArg>,
    lance_max_rows_per_file: Option<usize>,
    lance_max_bytes_per_file: Option<usize>,
    row_id_offset: u64,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let timer = WallTimer::start();

    let dataset = dataset_arg.into();
    let engine_name: EngineName = engine.clone().into();
    let opened = dataset::open_dataset(
        dataset,
        std::path::Path::new(&input),
        dataset::DatasetReadOptions {
            batch_size,
            limit_rows,
            row_id_offset,
        },
    )
    .await?;

    let counters = Arc::new(BatchCounters::default());
    let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
        CountingRecordBatchReader::new(opened.reader, counters.clone()),
    );

    match engine {
        EngineArg::Lance => {
            let mut ingest_options = engine_lance::IngestOptions::default();
            if matches!(dataset, DatasetName::FineWeb)
                && lance_max_rows_per_file.is_none()
                && lance_max_bytes_per_file.is_none()
            {
                ingest_options.max_rows_per_file = Some(usize::MAX);
                ingest_options.max_bytes_per_file = Some(usize::MAX);
            }
            ingest_options.max_rows_per_file =
                lance_max_rows_per_file.or(ingest_options.max_rows_per_file);
            ingest_options.max_bytes_per_file =
                lance_max_bytes_per_file.or(ingest_options.max_bytes_per_file);
            ingest_options.data_storage_version = lance_file_version.clone();
            ingest_options.write_mode = lance_write_mode.as_ref().map(|mode| match mode {
                LanceWriteModeArg::Create => "create".to_string(),
                LanceWriteModeArg::Append => "append".to_string(),
                LanceWriteModeArg::Overwrite => "overwrite".to_string(),
            });
            if let Some(cols) = &lance_index_columns {
                ingest_options.index_columns = cols
                    .split(',')
                    .map(|v| v.trim())
                    .filter(|v| !v.is_empty())
                    .map(|v| v.to_string())
                    .collect();
            }
            engine_lance::LanceEngine::new()
                .ingest(reader, std::path::Path::new(&out), ingest_options)
                .await?
        }
        EngineArg::LanceFragment => {
            anyhow::bail!("engine lance-fragment supports scan workloads only")
        }
        EngineArg::Parquet => {
            engine_parquet::ParquetEngine::new()
                .ingest(reader, std::path::Path::new(&out))
                .await?
        }
    }

    let timing = timer.stop();
    let ingested_rows = counters.rows.load(Ordering::Relaxed);
    let ingested_bytes = counters.bytes.load(Ordering::Relaxed);
    let mut params = BTreeMap::new();
    params.insert("input".to_string(), input);
    let dataset_out = out;
    params.insert("dataset_out".to_string(), dataset_out.clone());
    params.insert("batch_size".to_string(), batch_size.to_string());
    if let Some(limit) = limit_rows {
        params.insert("limit_rows".to_string(), limit.to_string());
    }
    params.insert("row_id_offset".to_string(), row_id_offset.to_string());
    if matches!(engine_name, EngineName::Lance | EngineName::LanceFragment) {
        let v = engine_lance::LanceEngine::new()
            .data_storage_version(std::path::Path::new(&dataset_out))
            .await?;
        params.insert("lance_file_version".to_string(), v);
        if let Some(v) = lance_file_version {
            params.insert("lance_file_version_requested".to_string(), v);
        }
        if let Some(cols) = lance_index_columns {
            if !cols.is_empty() {
                params.insert("lance_index_columns".to_string(), cols);
            }
        }
        if let Some(mode) = lance_write_mode {
            params.insert(
                "lance_write_mode".to_string(),
                match mode {
                    LanceWriteModeArg::Create => "create",
                    LanceWriteModeArg::Append => "append",
                    LanceWriteModeArg::Overwrite => "overwrite",
                }
                .to_string(),
            );
        }
        if let Some(v) = lance_max_rows_per_file {
            params.insert("lance_max_rows_per_file".to_string(), v.to_string());
        }
        if let Some(v) = lance_max_bytes_per_file {
            params.insert("lance_max_bytes_per_file".to_string(), v.to_string());
        }
    }
    params.insert(
        "schema_fields".to_string(),
        opened.schema.fields().len().to_string(),
    );
    params.insert("rows".to_string(), ingested_rows.to_string());
    params.insert("bytes".to_string(), ingested_bytes.to_string());
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset,
            Workload::Ingest,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: Some(ingested_rows),
        bytes: Some(ingested_bytes),
        latency: None,
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn scan_count_rows_measured(
    engine: EngineArg,
    path: &std::path::Path,
    projection: Option<&[String]>,
    filter: Option<&Filter>,
    limit_rows: Option<u64>,
    row_offset: Option<u64>,
    repeats: u32,
) -> Result<(bench_core::metrics::Timing, LatencySummary, u64, u64)> {
    match engine {
        EngineArg::Lance => {
            engine_lance::LanceEngine::new()
                .scan_count_rows_measured(path, projection, filter, limit_rows, row_offset, repeats)
                .await
        }
        EngineArg::LanceFragment => {
            engine_lance::LanceEngine::new()
                .scan_count_rows_fragments_measured(path, projection, filter, repeats)
                .await
        }
        EngineArg::Parquet => {
            engine_parquet::ParquetEngine::new()
                .scan_count_rows_measured(path, projection, filter, limit_rows, row_offset, repeats)
                .await
        }
    }
}

async fn run_scan_full(
    seed: u64,
    engine: EngineArg,
    dataset: DatasetArg,
    path: String,
    repeats: u32,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let dataset_name = dataset.into();
    let engine_name: EngineName = engine.clone().into();
    let projection =
        default_full_projection(engine.clone(), dataset_name, std::path::Path::new(&path)).await?;

    let (timing, latency, rows, bytes) = scan_count_rows_measured(
        engine.clone(),
        std::path::Path::new(&path),
        Some(&projection),
        None,
        None,
        None,
        repeats,
    )
    .await?;
    let (lance_fragment_count, lance_data_file_count) = match engine_name {
        EngineName::Lance | EngineName::LanceFragment => {
            let lance = engine_lance::LanceEngine::new();
            let fragment_count = lance.fragment_count(std::path::Path::new(&path)).await?;
            let data_file_count = if is_uri_path(&path) {
                None
            } else {
                Some(lance.data_file_count(std::path::Path::new(&path))?)
            };
            (Some(fragment_count), data_file_count)
        }
        _ => (None, None),
    };
    let mut params = BTreeMap::new();
    let path_for_index = path.clone();
    params.insert("path".to_string(), path);
    maybe_add_lance_index_params(engine_name, &path_for_index, &mut params);
    params.insert("projection".to_string(), projection.join(","));
    params.insert("repeats".to_string(), repeats.to_string());
    if matches!(engine_name, EngineName::Lance | EngineName::LanceFragment) {
        let v = engine_lance::LanceEngine::new()
            .data_storage_version(std::path::Path::new(params.get("path").unwrap()))
            .await?;
        params.insert("lance_file_version".to_string(), v);
    }
    if let Some(v) = lance_fragment_count {
        params.insert("lance_fragment_count".to_string(), v.to_string());
    }
    if let Some(v) = lance_data_file_count {
        params.insert("lance_data_file_count".to_string(), v.to_string());
    }
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset_name,
            Workload::ScanFull,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: Some(rows),
        bytes: Some(bytes),
        latency: Some(latency),
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn run_scan(
    seed: u64,
    engine: EngineArg,
    dataset: DatasetArg,
    path: String,
    mode: ScanModeArg,
    projection: Option<String>,
    limit_rows: Option<u64>,
    row_offset: Option<u64>,
    repeats: u32,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

    let dataset_name = dataset.into();
    let engine_name: EngineName = engine.clone().into();
    let mut projection = parse_projection(projection);
    if projection.is_none() {
        if matches!(mode, ScanModeArg::Full) {
            projection = Some(
                default_full_projection(engine.clone(), dataset_name, std::path::Path::new(&path))
                    .await?,
            );
        } else {
            projection = Some(vec![BENCH_SMALL_U64.to_string()]);
        }
    }

    let (workload, filter, threshold) = match mode {
        ScanModeArg::Full => (Workload::ScanFull, None, None),
        ScanModeArg::Project => (Workload::ScanProject, None, None),
        ScanModeArg::FilterLow => {
            let threshold = selectivity_threshold_u64(100);
            let filter = Filter::lt_u64(BENCH_SMALL_U64, threshold);
            (Workload::ScanFilterLow, Some(filter), Some(threshold))
        }
        ScanModeArg::FilterHigh => {
            let threshold = selectivity_threshold_u64(2);
            let filter = Filter::lt_u64(BENCH_SMALL_U64, threshold);
            (Workload::ScanFilterHigh, Some(filter), Some(threshold))
        }
    };

    let (timing, latency, rows, bytes) = scan_count_rows_measured(
        engine.clone(),
        std::path::Path::new(&path),
        projection.as_deref(),
        filter.as_ref(),
        limit_rows,
        row_offset,
        repeats,
    )
    .await?;
    let (lance_fragment_count, lance_data_file_count) = match engine_name {
        EngineName::Lance | EngineName::LanceFragment => {
            let lance = engine_lance::LanceEngine::new();
            let fragment_count = lance.fragment_count(std::path::Path::new(&path)).await?;
            let data_file_count = if is_uri_path(&path) {
                None
            } else {
                Some(lance.data_file_count(std::path::Path::new(&path))?)
            };
            (Some(fragment_count), data_file_count)
        }
        _ => (None, None),
    };
    let mut params = BTreeMap::new();
    let path_for_index = path.clone();
    params.insert("path".to_string(), path);
    maybe_add_lance_index_params(engine_name, &path_for_index, &mut params);
    if let Some(projection) = &projection {
        params.insert("projection".to_string(), projection.join(","));
    }
    if let Some(threshold) = threshold {
        params.insert(
            "filter".to_string(),
            format!("{} < {}", BENCH_SMALL_U64, threshold),
        );
    }
    params.insert("repeats".to_string(), repeats.to_string());
    if let Some(limit_rows) = limit_rows {
        params.insert("limit_rows".to_string(), limit_rows.to_string());
    }
    if let Some(row_offset) = row_offset {
        params.insert("row_offset".to_string(), row_offset.to_string());
    }
    if matches!(engine_name, EngineName::Lance | EngineName::LanceFragment) {
        let v = engine_lance::LanceEngine::new()
            .data_storage_version(std::path::Path::new(params.get("path").unwrap()))
            .await?;
        params.insert("lance_file_version".to_string(), v);
    }
    if let Some(v) = lance_fragment_count {
        params.insert("lance_fragment_count".to_string(), v.to_string());
    }
    if let Some(v) = lance_data_file_count {
        params.insert("lance_data_file_count".to_string(), v.to_string());
    }
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset_name,
            workload,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: Some(rows),
        bytes: Some(bytes),
        latency: Some(latency),
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn run_take(
    seed: u64,
    engine: EngineArg,
    dataset: DatasetArg,
    path: String,
    iters: u64,
    projection: Option<String>,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut projection = parse_projection(projection);
    if projection.is_none() {
        projection = Some(vec![bench_core::dataset::BENCH_SMALL_U64.to_string()]);
    }

    let mut histogram = Histogram::<u64>::new(3)?;
    let timer = WallTimer::start();
    let (row_count, lance_file_version) = match engine {
        EngineArg::Lance => {
            let lance = engine_lance::LanceEngine::new();
            let dataset = lance
                .open_dataset(std::path::Path::new(&path))
                .await
                .context("open lance dataset")?;

            let row_count = dataset.count_rows(None).await? as u64;
            let projection_request = match projection.as_deref() {
                Some(cols) => {
                    engine_lance::ProjectionRequest::from_columns(cols.iter(), dataset.schema())
                }
                None => engine_lance::ProjectionRequest::Schema(dataset.schema().clone().into()),
            };

            for _ in 0..iters {
                let row_offset = rng.random_range(0..row_count.max(1));
                let start = std::time::Instant::now();
                let _ = dataset
                    .take(&[row_offset], projection_request.clone())
                    .await?;
                let elapsed_us = start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                histogram.record(elapsed_us)?;
            }

            let v = dataset
                .manifest
                .data_storage_format
                .lance_file_version()?
                .to_string();

            (row_count, Some(v))
        }
        EngineArg::LanceFragment => {
            anyhow::bail!("engine lance-fragment supports scan workloads only")
        }
        EngineArg::Parquet => {
            let parquet = engine_parquet::ParquetEngine::new();
            let opened = parquet
                .open_file(std::path::Path::new(&path))
                .await
                .context("open parquet file")?;
            let row_count = opened.row_count();

            for _ in 0..iters {
                let row_offset = rng.random_range(0..row_count.max(1));
                let start = std::time::Instant::now();
                parquet
                    .take_one_opened(&opened, row_offset, projection.as_deref())
                    .await?;
                let elapsed_us = start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
                histogram.record(elapsed_us)?;
            }

            (row_count, None)
        }
    };
    let timing = timer.stop();

    let engine_name: EngineName = engine.clone().into();
    let dataset_name: DatasetName = dataset.into();
    let mut params = BTreeMap::new();
    let path_for_index = path.clone();
    params.insert("path".to_string(), path);
    maybe_add_lance_index_params(engine_name, &path_for_index, &mut params);
    params.insert("iters".to_string(), iters.to_string());
    params.insert("row_count".to_string(), row_count.to_string());
    if let Some(projection) = &projection {
        params.insert("projection".to_string(), projection.join(","));
    }
    if let Some(v) = lance_file_version {
        params.insert("lance_file_version".to_string(), v);
    }
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset_name,
            Workload::RandomTake,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: Some(iters),
        bytes: None,
        latency: Some(LatencySummary::from_histogram(&histogram)),
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn run_blob(
    seed: u64,
    engine: EngineArg,
    dataset: DatasetArg,
    path: String,
    column: String,
    iters: u64,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let mut rng = StdRng::seed_from_u64(seed);
    let lance_engine = engine_lance::LanceEngine::new();
    let parquet_engine = engine_parquet::ParquetEngine::new();

    let lance_dataset = if matches!(engine, EngineArg::Lance) {
        Some(
            lance_engine
                .open_dataset(std::path::Path::new(&path))
                .await
                .context("open lance dataset")?,
        )
    } else {
        None
    };
    let row_count = match engine {
        EngineArg::Lance => {
            lance_dataset
                .as_ref()
                .expect("lance dataset must be opened")
                .count_rows(None)
                .await? as u64
        }
        EngineArg::LanceFragment => {
            anyhow::bail!("engine lance-fragment supports scan workloads only")
        }
        EngineArg::Parquet => parquet_engine
            .open_file(std::path::Path::new(&path))
            .await
            .context("open parquet file")?
            .row_count(),
    };

    let lance_file_version = lance_dataset.as_ref().map(|dataset| {
        dataset
            .manifest
            .data_storage_format
            .lance_file_version()
            .map(|v| v.to_string())
    });
    let lance_file_version = lance_file_version.transpose()?;

    if matches!(engine, EngineArg::Lance) && lance_file_version.as_deref() == Some("2.2") {
        let dataset_for_probe = lance_dataset
            .as_ref()
            .expect("lance dataset must be opened")
            .clone();
        let column_for_probe = column.clone();
        let probe = tokio::spawn(async move {
            engine_lance::LanceEngine::new()
                .take_blob_one_opened(&dataset_for_probe, 0, &column_for_probe)
                .await
        })
        .await;

        match probe {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => return Err(err),
            Err(join_err) => {
                let timing = WallTimer::start().stop();
                let engine_name: EngineName = engine.clone().into();
                let dataset_name: DatasetName = dataset.into();
                let mut params = BTreeMap::new();
                let path_for_index = path.clone();
                params.insert("path".to_string(), path);
                maybe_add_lance_index_params(engine_name, &path_for_index, &mut params);
                params.insert("iters".to_string(), iters.to_string());
                params.insert("row_count".to_string(), row_count.to_string());
                params.insert("column".to_string(), column);
                if let Some(v) = lance_file_version {
                    params.insert("lance_file_version".to_string(), v);
                }
                let result = RunResult {
                    meta: make_meta(
                        engine_name,
                        dataset_name,
                        Workload::RandomBlob,
                        seed,
                        started_at_unix_ms,
                        params,
                    ),
                    timing,
                    rows: Some(iters),
                    bytes: Some(0),
                    latency: None,
                    notes: vec![format!("panic: {join_err}")],
                };
                write_result(&result, result_out).await?;
                return Ok(());
            }
        }
    }

    let mut histogram = Histogram::<u64>::new(3)?;
    let timer = WallTimer::start();
    let mut bytes = 0u64;
    for _ in 0..iters {
        let row_offset = rng.random_range(0..row_count.max(1));
        let start = std::time::Instant::now();
        let nbytes = match engine {
            EngineArg::Lance => {
                let dataset = lance_dataset
                    .as_ref()
                    .expect("lance dataset must be opened before iteration");
                lance_engine
                    .take_blob_one_opened(dataset, row_offset, &column)
                    .await? as u64
            }
            EngineArg::LanceFragment => {
                anyhow::bail!("engine lance-fragment supports scan workloads only")
            }
            EngineArg::Parquet => {
                parquet_engine
                    .take_binary_one(std::path::Path::new(&path), row_offset, &column)
                    .await? as u64
            }
        };
        bytes = bytes.saturating_add(nbytes);
        let elapsed_us = start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        histogram.record(elapsed_us)?;
    }
    let timing = timer.stop();

    let engine_name: EngineName = engine.clone().into();
    let dataset_name: DatasetName = dataset.into();
    let mut params = BTreeMap::new();
    let path_for_index = path.clone();
    params.insert("path".to_string(), path);
    maybe_add_lance_index_params(engine_name, &path_for_index, &mut params);
    params.insert("iters".to_string(), iters.to_string());
    params.insert("row_count".to_string(), row_count.to_string());
    params.insert("column".to_string(), column.clone());
    if let Some(v) = lance_file_version {
        params.insert("lance_file_version".to_string(), v);
    }
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset_name,
            Workload::RandomBlob,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: Some(iters),
        bytes: Some(bytes),
        latency: if histogram.len() > 0 {
            Some(LatencySummary::from_histogram(&histogram))
        } else {
            None
        },
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn run_evolve(
    seed: u64,
    engine: EngineArg,
    dataset: DatasetArg,
    path: String,
    new_column: String,
    out: Option<String>,
    expr: Option<String>,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let timer = WallTimer::start();

    let engine_name: EngineName = engine.clone().into();
    let dataset_name: DatasetName = dataset.into();
    let mut params = BTreeMap::new();
    params.insert("path".to_string(), path.clone());
    maybe_add_lance_index_params(engine_name, &path, &mut params);
    params.insert("new_column".to_string(), new_column.clone());
    if matches!(engine_name, EngineName::Lance | EngineName::LanceFragment) {
        let v = engine_lance::LanceEngine::new()
            .data_storage_version(std::path::Path::new(&path))
            .await?;
        params.insert("lance_file_version".to_string(), v);
    }
    let base_size_bytes = fs::dir_size_bytes(std::path::Path::new(&path)).unwrap_or(0);
    params.insert("base_size_bytes".to_string(), base_size_bytes.to_string());

    let bytes_written = match engine {
        EngineArg::Lance => {
            let expr_str =
                expr.unwrap_or_else(|| format!("{} + 1", bench_core::dataset::BENCH_ROW_ID));
            params.insert("expr".to_string(), expr_str.clone());
            engine_lance::LanceEngine::new()
                .evolution_add_column_sql(std::path::Path::new(&path), &new_column, &expr_str)
                .await?;
            let evolved_size_bytes = fs::dir_size_bytes(std::path::Path::new(&path)).unwrap_or(0);
            params.insert(
                "evolved_size_bytes".to_string(),
                evolved_size_bytes.to_string(),
            );
            evolved_size_bytes.saturating_sub(base_size_bytes)
        }
        EngineArg::LanceFragment => {
            anyhow::bail!("engine lance-fragment supports scan workloads only")
        }
        EngineArg::Parquet => {
            let out_path = out.unwrap_or_else(|| format!("{path}.evolved.parquet"));
            params.insert("rewrite_out".to_string(), out_path.clone());
            engine_parquet::ParquetEngine::new()
                .evolution_add_column_rewrite(
                    std::path::Path::new(&path),
                    std::path::Path::new(&out_path),
                    &new_column,
                )
                .await?;
            let evolved_size_bytes =
                fs::dir_size_bytes(std::path::Path::new(&out_path)).unwrap_or(0);
            params.insert(
                "evolved_size_bytes".to_string(),
                evolved_size_bytes.to_string(),
            );
            evolved_size_bytes
        }
    };

    let timing = timer.stop();
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset_name,
            Workload::EvolutionBackfill,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: None,
        bytes: Some(bytes_written),
        latency: None,
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn run_size(
    seed: u64,
    engine: Option<EngineArg>,
    dataset: Option<DatasetArg>,
    path: String,
    result_out: Option<String>,
) -> Result<()> {
    let started_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let timer = WallTimer::start();
    let bytes = fs::dir_size_bytes(std::path::Path::new(&path))?;
    let timing = timer.stop();
    let engine_name = engine
        .clone()
        .map(EngineName::from)
        .unwrap_or(EngineName::Unknown);
    let dataset_name = dataset
        .map(DatasetName::from)
        .unwrap_or(DatasetName::Unknown);
    let mut params = BTreeMap::new();
    let path_for_index = path.clone();
    params.insert("path".to_string(), path);
    maybe_add_lance_index_params(engine_name, &path_for_index, &mut params);
    if matches!(engine_name, EngineName::Lance | EngineName::LanceFragment) {
        let v = engine_lance::LanceEngine::new()
            .data_storage_version(std::path::Path::new(params.get("path").unwrap()))
            .await?;
        params.insert("lance_file_version".to_string(), v);
    }
    let result = RunResult {
        meta: make_meta(
            engine_name,
            dataset_name,
            Workload::Size,
            seed,
            started_at_unix_ms,
            params,
        ),
        timing,
        rows: None,
        bytes: Some(bytes),
        latency: None,
        notes: vec![],
    };
    write_result(&result, result_out).await?;
    Ok(())
}

async fn write_result(result: &RunResult, out: Option<String>) -> Result<()> {
    let json = serde_json::to_string_pretty(result)?;
    if let Some(path) = out {
        tokio::fs::write(path, json).await?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_projection(projection: Option<String>) -> Option<Vec<String>> {
    projection.map(|p| {
        p.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
    })
}

fn is_uri_path(path: &str) -> bool {
    path.contains("://")
}

fn selectivity_threshold_u64(denom: u64) -> u64 {
    if denom <= 1 {
        return u64::MAX;
    }
    u64::MAX / denom
}

async fn default_full_projection(
    engine: EngineArg,
    dataset: DatasetName,
    path: &std::path::Path,
) -> Result<Vec<String>> {
    let mut cols = match engine {
        EngineArg::Parquet => {
            engine_parquet::ParquetEngine::new()
                .schema_field_names(path)
                .await?
        }
        EngineArg::Lance => {
            engine_lance::LanceEngine::new()
                .schema_field_names(path)
                .await?
        }
        EngineArg::LanceFragment => {
            engine_lance::LanceEngine::new()
                .schema_field_names(path)
                .await?
        }
    };
    let blobs = blob_columns(dataset);
    cols.retain(|c| !blobs.iter().any(|b| *b == c.as_str()));
    Ok(cols)
}

fn blob_columns(dataset: DatasetName) -> &'static [&'static str] {
    match dataset {
        DatasetName::Laion10m => &["image"],
        DatasetName::OpenVid => &["video_blob"],
        _ => &[],
    }
}

fn make_meta(
    engine: EngineName,
    dataset: DatasetName,
    workload: Workload,
    seed: u64,
    started_at_unix_ms: u128,
    params: BTreeMap<String, String>,
) -> RunMetadata {
    RunMetadata {
        bench_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: git_commit(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count: std::thread::available_parallelism()
            .map(|n| n.get() as u64)
            .unwrap_or(0),
        engine_version: engine_version(engine),
        engine,
        dataset,
        workload,
        seed,
        started_at_unix_ms,
        rustc: rustc_version(),
        params,
    }
}

fn engine_version(engine: EngineName) -> String {
    match engine {
        EngineName::Parquet => format!(
            "adapter={} dep={}",
            engine_parquet::ENGINE_ADAPTER_VERSION,
            engine_parquet::PARQUET_DEP_VERSION
        ),
        EngineName::Lance => format!(
            "adapter={} dep={}",
            engine_lance::ENGINE_ADAPTER_VERSION,
            engine_lance::LANCE_DEP_VERSION
        ),
        EngineName::LanceFragment => format!(
            "adapter={} dep={}",
            engine_lance::ENGINE_ADAPTER_VERSION,
            engine_lance::LANCE_DEP_VERSION
        ),
        EngineName::Unknown => "unknown".to_string(),
    }
}

fn git_commit() -> String {
    if let Some(v) = option_env!("BENCH_GIT_COMMIT") {
        let v = v.trim();
        if !v.is_empty() {
            return v.to_string();
        }
    }
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "no-commits".to_string())
}

#[derive(Default)]
struct BatchCounters {
    rows: AtomicU64,
    bytes: AtomicU64,
}

struct CountingRecordBatchReader {
    inner: Box<dyn arrow_array::RecordBatchReader + Send>,
    counters: Arc<BatchCounters>,
}

impl CountingRecordBatchReader {
    fn new(
        inner: Box<dyn arrow_array::RecordBatchReader + Send>,
        counters: Arc<BatchCounters>,
    ) -> Self {
        Self { inner, counters }
    }
}

impl arrow_array::RecordBatchReader for CountingRecordBatchReader {
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.inner.schema()
    }
}

impl Iterator for CountingRecordBatchReader {
    type Item = std::result::Result<arrow_array::RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        if let Ok(batch) = &batch {
            self.counters
                .rows
                .fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
            self.counters
                .bytes
                .fetch_add(batch.get_array_memory_size() as u64, Ordering::Relaxed);
        }
        Some(batch)
    }
}
