//! log.txt から experiment.json を復元するCLIツール
//!
//! 学習が中断されてexperiment.jsonが生成されなかった過去の実験を救済する。
//!
//! Usage:
//!   cargo run --release --example recover_experiment -- \
//!     --checkpoint-dir checkpoints/v63 \
//!     --net-id v63 \
//!     --name v63
//!
//! 必須パラメータを指定しない場合はlog.txtの情報のみで最小限のJSONを生成する。

use clap::Parser;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "recover_experiment", about = "Recover experiment.json from log.txt")]
struct Args {
    /// チェックポイントディレクトリ (e.g. checkpoints/v63)
    #[arg(long)]
    checkpoint_dir: PathBuf,

    /// ネットID (e.g. v63)
    #[arg(long)]
    net_id: String,

    /// 実験名 (指定しなければnet_idと同じ)
    #[arg(long)]
    name: Option<String>,

    /// アーキテクチャ名 (e.g. "LayerStack-1536-16-32")
    #[arg(long)]
    architecture: Option<String>,

    /// バケットモード (e.g. "progress8kpabs")
    #[arg(long)]
    bucket_mode: Option<String>,

    /// オプティマイザ (e.g. "Ranger")
    #[arg(long)]
    optimizer: Option<String>,

    /// 学習率
    #[arg(long)]
    lr: Option<f32>,

    /// バッチサイズ
    #[arg(long)]
    batch_size: Option<usize>,

    /// batches_per_superbatch
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// 学習データパス (カンマ区切り、局面数計算に使用)
    #[arg(long)]
    data: Option<String>,

    /// スケール値
    #[arg(long)]
    scale: Option<i32>,

    /// QA
    #[arg(long, default_value_t = 127)]
    qa: i16,

    /// QB
    #[arg(long, default_value_t = 64)]
    qb: i16,

    /// ステータス (running / completed)
    #[arg(long, default_value = "completed")]
    status: String,

    /// コマンド文字列
    #[arg(long)]
    command: Option<String>,

    /// 出力ファイルパス。指定しない場合は --checkpoint-dir の渡し方で決まる:
    ///   --checkpoint-dir が学習の --output と同じ root の場合 (例: checkpoints):
    ///     → checkpoint_dir/<net_id>/experiment.json
    ///   --checkpoint-dir が実験固有ディレクトリの場合 (例: checkpoints/v63 で末尾が net_id と一致):
    ///     → checkpoint_dir/experiment.json
    /// いずれの場合も学習側の write_experiment_json と同じパスに書き戻すよう調整される。
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// 上書き確認なしで実行
    #[arg(long)]
    force: bool,
}

// =============================================================================
// Structures (ExperimentLog と互換)
// =============================================================================

#[derive(Serialize)]
struct ExperimentLog {
    id: String,
    name: String,
    date: String,
    status: String,
    last_updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<String>,
    params: ExperimentParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<ExperimentData>,
    results: ExperimentResults,
    history: Vec<LossEntry>,
    checkpoints: Vec<String>,
}

#[derive(Serialize)]
struct ExperimentParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bucket_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batches_per_superbatch: Option<usize>,
    superbatches: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale: Option<i32>,
    qa: i16,
    qb: i16,
}

#[derive(Serialize)]
struct ExperimentData {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    positions: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_positions: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset_passes: Option<f64>,
}

#[derive(Serialize)]
struct ExperimentResults {
    #[serde(skip_serializing_if = "Option::is_none")]
    training_time_seconds: Option<u64>,
    fv_scale: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_loss_superbatch: Option<usize>,
}

#[derive(Serialize)]
struct LossEntry {
    superbatch: usize,
    loss: f64,
}

// =============================================================================
// Helpers (shogi_layerstack.rs / shogi_simple.rs と同じロジック)
// =============================================================================

fn parse_loss_history(log_path: &Path) -> Vec<LossEntry> {
    let content = match std::fs::read_to_string(log_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Warning: Cannot read {}: {}", log_path.display(), e);
            return Vec::new();
        }
    };
    let mut superbatch_losses: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            if let (Ok(sb), Ok(loss)) = (parts[0].trim().parse::<usize>(), parts[2].trim().parse::<f64>()) {
                let entry = superbatch_losses.entry(sb).or_insert((0.0, 0));
                entry.0 += loss;
                entry.1 += 1;
            }
        }
    }
    superbatch_losses
        .into_iter()
        .map(|(sb, (sum, count))| LossEntry { superbatch: sb, loss: sum / count as f64 })
        .collect()
}

fn collect_checkpoints(output_dir: &Path, net_id: &str) -> Vec<String> {
    let prefix = format!("{}-", net_id);
    let mut checkpoints: Vec<String> = std::fs::read_dir(output_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && entry.path().is_dir() {
                let suffix = &name[prefix.len()..];
                if suffix.parse::<usize>().is_ok() {
                    return Some(name);
                }
            }
            None
        })
        .collect();
    checkpoints.sort_by(|a, b| {
        let prefix_len = prefix.len();
        let a_num: usize = a[prefix_len..].parse().unwrap_or(0);
        let b_num: usize = b[prefix_len..].parse().unwrap_or(0);
        a_num.cmp(&b_num)
    });
    checkpoints
}

fn get_timestamp() -> (String, String) {
    use std::time::SystemTime;
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md as i64 {
            m = i;
            break;
        }
        remaining_days -= md as i64;
    }
    let d = remaining_days + 1;
    let id_ts = format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m + 1, d, hours, minutes, seconds);
    let date = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m + 1, d, hours, minutes, seconds);
    (id_ts, date)
}

fn main() {
    let args = Args::parse();
    let name = args.name.unwrap_or_else(|| args.net_id.clone());

    // Find latest checkpoint with log.txt
    let checkpoints = collect_checkpoints(&args.checkpoint_dir, &args.net_id);
    if checkpoints.is_empty() {
        eprintln!("Error: No checkpoints found in {} for net_id '{}'", args.checkpoint_dir.display(), args.net_id);
        std::process::exit(1);
    }

    let latest_checkpoint = checkpoints.last().unwrap();
    let log_path = args.checkpoint_dir.join(latest_checkpoint).join("log.txt");
    if !log_path.exists() {
        eprintln!("Error: log.txt not found at {}", log_path.display());
        std::process::exit(1);
    }

    println!("Parsing loss history from {} ...", log_path.display());
    let history = parse_loss_history(&log_path);
    println!("  Found {} superbatch entries from {} checkpoints", history.len(), checkpoints.len());

    let num_superbatches = history.last().map(|e| e.superbatch).unwrap_or(0);

    // Best loss
    let (best_loss, best_loss_superbatch) = history
        .iter()
        .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal))
        .map(|entry| (Some(entry.loss), Some(entry.superbatch)))
        .unwrap_or((None, None));

    // FV scale
    let fv_scale = if let Some(scale) = args.scale {
        (i32::from(args.qa) * i32::from(args.qb) + scale / 2) / scale
    } else {
        16 // default
    };

    // Data info
    let data = args.data.as_ref().map(|data_name| {
        const PACKED_SFEN_VALUE_SIZE: u64 = 40;
        let positions: u64 = data_name
            .split(',')
            .filter_map(|path| std::fs::metadata(path.trim()).ok())
            .map(|meta| meta.len() / PACKED_SFEN_VALUE_SIZE)
            .sum();

        let total_positions = match (args.batch_size, args.batches_per_superbatch) {
            (Some(bs), Some(bps)) => Some(bs as u64 * bps as u64 * num_superbatches as u64),
            _ => None,
        };
        let dataset_passes =
            total_positions.and_then(|tp| if positions > 0 { Some(tp as f64 / positions as f64) } else { None });

        ExperimentData {
            name: data_name.clone(),
            positions: if positions > 0 { Some(positions) } else { None },
            total_positions,
            dataset_passes,
        }
    });

    let (id_ts, date) = get_timestamp();
    let id = format!("{}-{}", id_ts, name);

    let experiment = ExperimentLog {
        id,
        name: name.clone(),
        date: date.clone(),
        status: args.status,
        last_updated_at: date,
        commit: None,
        command: args.command,
        params: ExperimentParams {
            architecture: args.architecture,
            bucket_mode: args.bucket_mode,
            optimizer: args.optimizer,
            lr: args.lr,
            batch_size: args.batch_size,
            batches_per_superbatch: args.batches_per_superbatch,
            superbatches: num_superbatches,
            scale: args.scale,
            qa: args.qa,
            qb: args.qb,
        },
        data,
        results: ExperimentResults { training_time_seconds: None, fv_scale, best_loss, best_loss_superbatch },
        history,
        checkpoints,
    };

    let json = serde_json::to_string_pretty(&experiment).expect("Failed to serialize JSON");

    // 学習側の write_experiment_json は output_dir.join(net_id).join("experiment.json")
    // に書く (例: --output checkpoints --net-id v63 → checkpoints/v63/experiment.json)。
    // recover の --checkpoint-dir は呼び出し方によって 2 つの粒度がありうる:
    //   1) 学習の --output と同じ root (例: checkpoints) を渡すケース
    //      → 出力は checkpoint_dir/<net_id>/experiment.json
    //   2) 実験固有ディレクトリ (例: checkpoints/v63) を渡すケース
    //      → 出力は checkpoint_dir/experiment.json
    // checkpoint_dir 末尾の component が net_id と一致するかで判定する。
    let output_path = args.output.unwrap_or_else(|| {
        let dir_already_experiment_root = args.checkpoint_dir.file_name().map(|s| s == name.as_str()).unwrap_or(false);
        if dir_already_experiment_root {
            args.checkpoint_dir.join("experiment.json")
        } else {
            args.checkpoint_dir.join(&name).join("experiment.json")
        }
    });

    // Check if file already exists
    if output_path.exists() && !args.force {
        eprintln!("Error: {} already exists. Use --force to overwrite.", output_path.display());
        std::process::exit(1);
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }
    std::fs::write(&output_path, &json).expect("Failed to write experiment.json");

    println!("Recovered experiment.json saved to {}", output_path.display());
    if let (Some(bl), Some(bls)) = (best_loss, best_loss_superbatch) {
        println!("  Best loss: {:.6} (superbatch {})", bl, bls);
    }
    println!("  Superbatches: {}", num_superbatches);
}
