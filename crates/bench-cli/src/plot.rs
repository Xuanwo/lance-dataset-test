use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::{fs as stdfs, io};

use anyhow::{Context, Result};
use bench_core::result::RunResult;
use bench_core::workload::{DatasetName, EngineName, Workload};
use plotters::prelude::*;

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

pub fn run_plots(results_dir: &Path, out_dir: &Path) -> Result<()> {
    let records = load_results(results_dir)?;
    stdfs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let datasets = dataset_iter().to_vec();
    let engines = engines_for_report(&records);

    plot_ingest_time_ms(out_dir, &records, &datasets, &engines)?;
    plot_ingest_output_size_bytes(out_dir, &records, &datasets, &engines)?;

    plot_scan_p50_ms(
        out_dir,
        &records,
        &datasets,
        &engines,
        Workload::ScanFull,
        "scan-full",
    )?;
    plot_scan_p50_ms(
        out_dir,
        &records,
        &datasets,
        &engines,
        Workload::ScanProject,
        "scan-project",
    )?;
    plot_scan_p50_ms(
        out_dir,
        &records,
        &datasets,
        &engines,
        Workload::ScanFilterLow,
        "scan-filter-low",
    )?;
    plot_scan_p50_ms(
        out_dir,
        &records,
        &datasets,
        &engines,
        Workload::ScanFilterHigh,
        "scan-filter-high",
    )?;

    plot_latency_p50_us(
        out_dir,
        &records,
        &datasets,
        &engines,
        Workload::RandomTake,
        "take",
    )?;
    plot_latency_p50_us(
        out_dir,
        &records,
        &datasets,
        &engines,
        Workload::RandomBlob,
        "blob",
    )?;

    plot_evolve_time_ms(out_dir, &records, &datasets, &engines)?;
    plot_evolve_bytes_written(out_dir, &records, &datasets, &engines)?;

    Ok(())
}

fn plot_ingest_time_ms(
    out_dir: &Path,
    records: &[RunRecord],
    datasets: &[DatasetName],
    engines: &[EngineSpec],
) -> Result<()> {
    let out_path = out_dir.join("ingest.time_ms.svg");
    plot_per_dataset_bars(
        &out_path,
        "Ingest time",
        "Time (ms)",
        datasets,
        engines,
        |dataset, engine| {
            let rec = find_record(records, *dataset, engine, Workload::Ingest)?;
            Some(rec.result.timing.wall_time_ms as f64)
        },
        |v| fmt_number(v, 2),
    )
}

fn plot_ingest_output_size_bytes(
    out_dir: &Path,
    records: &[RunRecord],
    datasets: &[DatasetName],
    engines: &[EngineSpec],
) -> Result<()> {
    let out_path = out_dir.join("ingest.output_size_bytes.svg");
    plot_per_dataset_bars(
        &out_path,
        "Ingest output size",
        "Size (bytes)",
        datasets,
        engines,
        |dataset, engine| {
            let rec = find_record(records, *dataset, engine, Workload::Size)?;
            rec.result.bytes.map(|b| b as f64)
        },
        |v| fmt_bytes(v as u64),
    )
}

fn plot_scan_p50_ms(
    out_dir: &Path,
    records: &[RunRecord],
    datasets: &[DatasetName],
    engines: &[EngineSpec],
    workload: Workload,
    name: &str,
) -> Result<()> {
    let out_path = out_dir.join(format!("{name}.p50_ms.svg"));
    plot_per_dataset_bars(
        &out_path,
        format!("{} p50", workload_title(workload)),
        "p50 (ms)",
        datasets,
        engines,
        |dataset, engine| {
            let rec = find_record(records, *dataset, engine, workload)?;
            let p50_us = rec.result.latency.as_ref()?.p50_us;
            Some((p50_us as f64) / 1000.0)
        },
        |v| fmt_number(v, 3),
    )
}

fn plot_latency_p50_us(
    out_dir: &Path,
    records: &[RunRecord],
    datasets: &[DatasetName],
    engines: &[EngineSpec],
    workload: Workload,
    name: &str,
) -> Result<()> {
    let out_path = out_dir.join(format!("{name}.p50_us.svg"));
    plot_per_dataset_bars(
        &out_path,
        format!("{} p50", workload_title(workload)),
        "p50 (us)",
        datasets,
        engines,
        |dataset, engine| {
            let rec = find_record(records, *dataset, engine, workload)?;
            let p50_us = rec.result.latency.as_ref()?.p50_us;
            Some(p50_us as f64)
        },
        |v| fmt_number(v, 0),
    )
}

fn plot_evolve_time_ms(
    out_dir: &Path,
    records: &[RunRecord],
    datasets: &[DatasetName],
    engines: &[EngineSpec],
) -> Result<()> {
    let out_path = out_dir.join("evolve.time_ms.svg");
    plot_per_dataset_bars(
        &out_path,
        "Evolution / Backfill time",
        "Time (ms)",
        datasets,
        engines,
        |dataset, engine| {
            let rec = find_record(records, *dataset, engine, Workload::EvolutionBackfill)?;
            Some(rec.result.timing.wall_time_ms as f64)
        },
        |v| fmt_number(v, 2),
    )
}

fn plot_evolve_bytes_written(
    out_dir: &Path,
    records: &[RunRecord],
    datasets: &[DatasetName],
    engines: &[EngineSpec],
) -> Result<()> {
    let out_path = out_dir.join("evolve.bytes_written.svg");
    plot_per_dataset_bars(
        &out_path,
        "Evolution / Backfill bytes written",
        "Bytes written (bytes)",
        datasets,
        engines,
        |dataset, engine| {
            let rec = find_record(records, *dataset, engine, Workload::EvolutionBackfill)?;
            rec.result.bytes.map(|b| b as f64)
        },
        |v| fmt_bytes(v as u64),
    )
}

fn plot_per_dataset_bars<TTitle: Into<String>, TLabel: Into<String>>(
    out_path: &Path,
    title: TTitle,
    y_label: TLabel,
    datasets: &[DatasetName],
    engines: &[EngineSpec],
    mut lookup: impl FnMut(&DatasetName, &EngineSpec) -> Option<f64>,
    mut fmt_value: impl FnMut(f64) -> String,
) -> Result<()> {
    let title = title.into();
    let y_label = y_label.into();

    let root = SVGBackend::new(out_path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.titled(&title, ("sans-serif", 30))?;
    let mut areas = root.split_evenly((2, 3));

    for (idx, dataset) in datasets.iter().enumerate() {
        if idx >= areas.len() {
            break;
        }
        let area = areas.remove(0);

        let mut values = Vec::new();
        for engine in engines {
            if let Some(v) = lookup(dataset, engine) {
                values.push((engine, v));
            }
        }

        if values.is_empty() {
            area.fill(&WHITE)?;
            continue;
        }

        let max_v = values
            .iter()
            .map(|(_, v)| *v)
            .fold(0.0_f64, |a, b| a.max(b));
        let y_max = if max_v <= 0.0 { 1.0 } else { max_v * 1.15 };

        let n = engines.len() as i32;
        let mut chart = ChartBuilder::on(&area)
            .margin(12)
            .caption(dataset_display(*dataset), ("sans-serif", 20))
            .x_label_area_size(50)
            .y_label_area_size(80)
            .build_cartesian_2d(0i32..n, 0f64..y_max)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_desc(y_label.as_str())
            .x_labels(engines.len())
            .x_label_formatter(&|x| {
                let idx = (*x as usize).min(engines.len().saturating_sub(1));
                engines[idx].label.clone()
            })
            .axis_desc_style(("sans-serif", 16))
            .label_style(("sans-serif", 14))
            .draw()?;

        for (i, engine) in engines.iter().enumerate() {
            let v = lookup(dataset, engine);
            let Some(v) = v else { continue };

            let color = engine_color(engine);
            let x0 = i as i32;
            let x1 = x0 + 1;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, v)],
                color.filled(),
            )))?;

            let label = fmt_value(v);
            chart.draw_series(std::iter::once(Text::new(
                label,
                (x0, v),
                ("sans-serif", 12).into_font().color(&BLACK),
            )))?;
        }
    }

    root.present()
        .with_context(|| format!("write {}", out_path.display()))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct EngineSpec {
    engine: EngineName,
    lance_file_version: Option<String>,
    lance_index_columns: Option<String>,
    label: String,
}

fn engines_for_report(records: &[RunRecord]) -> Vec<EngineSpec> {
    let mut out = Vec::new();
    out.push(EngineSpec {
        engine: EngineName::Parquet,
        lance_file_version: None,
        lance_index_columns: None,
        label: "parquet".to_string(),
    });

    let mut cases: Vec<(String, Option<String>)> = Vec::new();
    for r in records {
        if r.key.engine != EngineName::Lance {
            continue;
        }
        let Some(v) = &r.key.lance_file_version else {
            continue;
        };
        let case = (v.clone(), r.key.lance_index_columns.clone());
        if !cases.iter().any(|x| x == &case) {
            cases.push(case);
        }
    }
    cases.sort_by(|(a_v, a_idx), (b_v, b_idx)| {
        a_v.cmp(b_v)
            .then_with(|| {
                lance_index_columns_order(a_idx.as_deref())
                    .cmp(&lance_index_columns_order(b_idx.as_deref()))
            })
            .then_with(|| {
                a_idx
                    .clone()
                    .unwrap_or_default()
                    .cmp(&b_idx.clone().unwrap_or_default())
            })
    });
    for (v, idx) in cases {
        let label = if let Some(cols) = &idx {
            if cols.is_empty() {
                format!("lance@{v}")
            } else {
                format!("lance@{v}+idx({cols})")
            }
        } else {
            format!("lance@{v}")
        };
        out.push(EngineSpec {
            engine: EngineName::Lance,
            lance_file_version: Some(v.clone()),
            lance_index_columns: idx,
            label,
        });
    }
    out
}

fn engine_color(engine: &EngineSpec) -> RGBColor {
    match (
        engine.engine,
        engine.lance_file_version.as_deref(),
        engine.lance_index_columns.as_deref(),
    ) {
        (EngineName::Parquet, _, _) => RGBColor(175, 82, 222),
        (EngineName::Lance, Some("2.0"), None) => RGBColor(220, 78, 65),
        (EngineName::Lance, Some("2.0"), Some(_)) => RGBColor(180, 50, 40),
        (EngineName::Lance, Some("2.1"), None) => RGBColor(33, 150, 83),
        (EngineName::Lance, Some("2.1"), Some(_)) => RGBColor(20, 110, 60),
        (EngineName::Lance, _, _) => RGBColor(125, 125, 125),
        _ => RGBColor(125, 125, 125),
    }
}

fn dataset_iter() -> &'static [DatasetName] {
    &[
        DatasetName::FineWeb,
        DatasetName::OpenVid,
        DatasetName::Laion10m,
        DatasetName::LeRobotPushT,
        DatasetName::LeRobotPushTImage,
    ]
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

fn workload_title(workload: Workload) -> &'static str {
    match workload {
        Workload::ScanFull => "Scan (full)",
        Workload::ScanProject => "Scan (projection)",
        Workload::ScanFilterLow => "Scan (filter low selectivity)",
        Workload::ScanFilterHigh => "Scan (filter high selectivity)",
        Workload::RandomTake => "Random Take",
        Workload::RandomBlob => "Random Blob",
        Workload::Ingest => "Ingest",
        Workload::EvolutionBackfill => "Evolution / Backfill",
        Workload::Size => "Size",
    }
}

fn find_record<'a>(
    records: &'a [RunRecord],
    dataset: DatasetName,
    engine: &EngineSpec,
    workload: Workload,
) -> Option<&'a RunRecord> {
    records.iter().find(|r| {
        r.key.dataset == dataset
            && r.key.workload == workload
            && r.key.engine == engine.engine
            && r.key.lance_file_version == engine.lance_file_version
            && r.key.lance_index_columns == engine.lance_index_columns
    })
}

fn load_results(results_dir: &Path) -> Result<Vec<RunRecord>> {
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

        let engine = result.meta.engine;
        let dataset = result.meta.dataset;
        let workload = result.meta.workload;
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

        let rec = RunRecord {
            key: RunKey {
                dataset,
                engine,
                lance_file_version,
                lance_disable_compression,
                lance_index_columns,
                workload,
            },
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
            dataset_display(r.key.dataset).to_string(),
            engine_display_key(
                r.key.engine,
                r.key.lance_file_version.as_deref(),
                r.key.lance_disable_compression,
                r.key.lance_index_columns.as_deref(),
            ),
            workload_key(r.key.workload),
            lance_index_columns_order(r.key.lance_index_columns.as_deref()),
            r.key.lance_index_columns.clone().unwrap_or_default(),
            r.source.clone(),
        )
    });
    Ok(records)
}

fn engine_display_key(
    engine: EngineName,
    lance_file_version: Option<&str>,
    lance_disable_compression: bool,
    lance_index_columns: Option<&str>,
) -> String {
    match engine {
        EngineName::Parquet => "parquet".to_string(),
        EngineName::Lance => {
            let mut base = format!("lance@{}", lance_file_version.unwrap_or("?"));
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
        EngineName::LanceFragment => "lance-fragment".to_string(),
        EngineName::Unknown => "unknown".to_string(),
    }
}

fn lance_index_columns_order(cols: Option<&str>) -> u8 {
    match cols {
        None => 0,
        Some(_) => 1,
    }
}

fn workload_key(workload: Workload) -> u8 {
    match workload {
        Workload::Ingest => 10,
        Workload::ScanFull => 20,
        Workload::ScanProject => 21,
        Workload::ScanFilterLow => 22,
        Workload::ScanFilterHigh => 23,
        Workload::RandomTake => 30,
        Workload::RandomBlob => 31,
        Workload::EvolutionBackfill => 40,
        Workload::Size => 50,
    }
}

fn fmt_number(v: f64, decimals: usize) -> String {
    if decimals == 0 {
        return format!("{:.0}", v);
    }
    format!("{:.*}", decimals, v)
}

fn fmt_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let units = ["KiB", "MiB", "GiB", "TiB"];
    let mut unit = "B";
    for u in units {
        value /= 1024.0;
        unit = u;
        if value < 1024.0 {
            break;
        }
    }
    format!("{:.2} {unit}", value)
}
