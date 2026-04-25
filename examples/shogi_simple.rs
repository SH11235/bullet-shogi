/*
Shogi NNUE Training Script

Usage:
    cargo run --release --example shogi_simple -- [OPTIONS]

Options:
    --arch <ARCH>       Architecture preset (default: 256x2-32-32)
                        Presets: 256x2-32-32, 512x2-8-96, 512x2-32-32, 1024x2-8-32
    --l1 <SIZE>         L1 (accumulator) size (overrides preset)
    --l2 <SIZE>         L2 (hidden layer 1) size
    --l3 <SIZE>         L3 (hidden layer 2) size
    --data <PATH>       Training data path (comma-separated for multiple files)
    --batch-size <N>    Batch size (default: 16384)
    --superbatches <N>  Number of superbatches (default: 100)
    --lr <RATE>         Initial learning rate (default: 0.001)
    --wdl <LAMBDA>      WDL lambda for constant scheduler (default: 0.5)
                        Cannot be used with --start-wdl/--end-wdl
    --start-wdl <F>     Start WDL lambda for linear interpolation
    --end-wdl <F>       End WDL lambda for linear interpolation
                        Must use both --start-wdl and --end-wdl together
    --win-rate-model    Use win rate model for score conversion
    --scale <N>         Eval scale (default: 600)
                        FV_SCALE = QA*QB/scale (rounded)
                        QA=127 (CReLU):  8128/scale  -> 600->13, 508->16, 254->32, 1016->8
                        QA=255 (SCReLU): 16320/scale -> 600->27, 510->32, 1020->16
    --batches-per-superbatch <N>  Batches per superbatch (default: auto ~100M positions)
    --lr-gamma <F>      LR decay rate per step (default: 0.992)
    --lr-step <N>       LR decay interval in superbatches (default: 1)
    --start-superbatch <N>  Start superbatch number (default: 1)
    --batch-queue-size <N>  Batch prefetch queue size (default: 64)
    --save-rate <N>     Save interval in superbatches (default: 10)
    --threads <N>       Number of threads (default: 4)
    --output <DIR>      Output directory (default: checkpoints)
    --net-id <NAME>     Network ID (default: shogi-halfka-hm)
    --weight-decay <F>  Weight decay (default: 0.01)

Examples:
    # Train with default settings
    cargo run --release --example shogi_simple -- --data data/train.bin

    # Train with 512x2-8-96 architecture
    cargo run --release --example shogi_simple -- --arch 512x2-8-96 --data data/train.bin

    # Train with custom sizes
    cargo run --release --example shogi_simple -- --l1 1024 --l2 16 --l3 64 --data data/train.bin

    # Train with win rate model
    cargo run --release --example shogi_simple -- --win-rate-model --data data/train.bin

    # Train with linear WDL (start at 0.2, end at 0.8)
    cargo run --release --example shogi_simple -- --data data/train.bin --start-wdl 0.2 --end-wdl 0.8
*/

use std::{path::PathBuf, sync::OnceLock};

use bullet_lib::{
    game::inputs::{ShogiHalfKA, ShogiHalfKA_hm, ShogiHalfKP, SparseInputType},
    nn::{
        BackendMarker, NetworkBuilderNode,
        optimiser::{self, AdamWParams, RAdamParams, RangerParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy)]
struct WrmLossParams {
    nnue2score: f32,
    in_scaling: f32,
}

static WRM_LOSS_PARAMS: OnceLock<WrmLossParams> = OnceLock::new();
use serde::Serialize;

/// Feature set selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum FeatureSet {
    /// HalfKA_hm - Half-Mirrored King-All (73,305 dimensions)
    #[default]
    HalfkaHm,
    /// HalfKA - King-All non-mirrored (138,510 dimensions)
    Halfka,
    /// HalfKP - King-Piece (125,388 dimensions, no mirror)
    HalfKP,
}

/// Output format selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OutputFormat {
    /// bullet format: all i16 (l0w, l0b, l1w, l1b, l2w, l2b, outw, outb)
    Bullet,
    /// standard format: NNUE header + L0 i16 + L1-Out biases i32 + weights i8
    /// Compatible with nnue-pytorch / YaneuraOu
    #[default]
    Standard,
}

/// Activation function selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum ActivationType {
    /// SCReLU - Squared Clipped ReLU: y = clamp(x, 0, qa)²
    /// Higher expressiveness, used in modern Stockfish
    Screlu,
    /// CReLU - Clipped ReLU: y = clamp(x, 0, qa)
    /// Traditional activation, used in YaneuraOu/Suisho
    #[default]
    Crelu,
}

/// Pairwise multiplication mode
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum PairwiseMode {
    /// No pairwise multiplication (standard architecture)
    #[default]
    Off,
    /// Pairwise multiplication after L0 activation
    /// Output: a[0]*a[1], a[2]*a[3], ... (halves dimension)
    /// Best combined with CReLU activation
    On,
}

// =============================================================================
// CLI Arguments
// =============================================================================

/// Optimizer selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizerType {
    /// AdamW - fast convergence but may be unstable with sparse inputs
    AdamW,
    /// RAdam - Rectified Adam, more stable
    RAdam,
    /// Ranger - RAdam + Lookahead (recommended by nnue-pytorch)
    #[default]
    Ranger,
}

#[derive(Parser, Debug)]
#[command(name = "shogi_simple")]
#[command(about = "Shogi NNUE training script")]
struct Args {
    /// Feature set (halfka-hm, halfka, halfkp)
    /// halfka-hm: HalfKA_hm (73,305 dims, Half-Mirror) - nnue-pytorch compatible
    /// halfka: HalfKA (138,510 dims, no mirror) - rshogi compatible
    /// halfkp: HalfKP (125,388 dims, no mirror) - classic NNUE
    #[arg(long, value_enum, default_value = "halfka-hm")]
    features: FeatureSet,

    /// Output format (standard or bullet)
    /// standard: NNUE header + L0 i16 + L1-Out biases i32 + weights i8 (default)
    /// bullet: all i16, no header
    #[arg(long, value_enum, default_value = "standard")]
    output_format: OutputFormat,

    /// Activation function (crelu or screlu)
    /// crelu: Clipped ReLU - traditional, used in YaneuraOu/Suisho (default)
    /// screlu: Squared Clipped ReLU - higher expressiveness
    #[arg(long, value_enum, default_value = "crelu")]
    activation: ActivationType,

    /// Pairwise multiplication mode (off or on)
    /// off: Standard architecture (L1 input = 2*L1_SIZE)
    /// on: Apply pairwise_mul after L0 (L1 input = L1_SIZE, halved)
    /// Best combined with --activation crelu
    #[arg(long, value_enum, default_value = "off")]
    pairwise: PairwiseMode,

    /// Architecture preset
    /// Presets: 256x2-32-32, 512x2-8-96, 512x2-32-32, 1024x2-8-32
    #[arg(long, default_value = "256x2-32-32")]
    arch: String,

    /// Optimizer (adamw, radam, ranger)
    /// ranger = RAdam + Lookahead (same as nnue-pytorch recommendation)
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerType,

    /// L1 (accumulator) size (overrides preset)
    #[arg(long)]
    l1: Option<usize>,

    /// L2 (hidden layer 1) size
    #[arg(long)]
    l2: Option<usize>,

    /// L3 (hidden layer 2) size
    #[arg(long)]
    l3: Option<usize>,

    /// Training data path (comma-separated for multiple files)
    #[arg(long, default_value = "data/train.bin")]
    data: String,

    /// Batch size
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of superbatches
    #[arg(long, default_value = "100")]
    superbatches: usize,

    /// Initial learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// WDL lambda (0.0=eval only, 1.0=game result only, default: 0.5)
    /// Cannot be used with --start-wdl/--end-wdl
    #[arg(long, conflicts_with_all = ["start_wdl", "end_wdl"])]
    wdl: Option<f32>,

    /// Start WDL lambda for linear interpolation
    /// Must be used together with --end-wdl
    #[arg(long, requires = "end_wdl")]
    start_wdl: Option<f32>,

    /// End WDL lambda for linear interpolation
    /// Must be used together with --start-wdl
    #[arg(long, requires = "start_wdl")]
    end_wdl: Option<f32>,

    /// Eval scale for training target sigmoid(score / scale).
    /// Eval_Coef=600 のDL教師データと整合させるため、デフォルト600。
    /// FV_SCALE = QA*QB/scale (rounded).
    ///   QA=127 (CReLU):  600->13, 508->16, 254->32, 1016->8
    ///   QA=255 (SCReLU): 600->27, 510->32, 1020->16
    #[arg(long, default_value = "600")]
    scale: i32,

    /// Save interval (superbatches)
    #[arg(long, default_value = "10")]
    save_rate: usize,

    /// Number of threads
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Output directory
    #[arg(long, default_value = "checkpoints")]
    output: PathBuf,

    /// Network ID
    #[arg(long, default_value = "shogi-halfka-hm")]
    net_id: String,

    /// Quantization factor QA (for L0)
    #[arg(long, default_value = "127")]
    qa: i16,

    /// Quantization factor QB (for later layers)
    #[arg(long, default_value = "64")]
    qb: i16,

    /// Weight decay (L2 regularization)
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Batches per superbatch (default: auto-calculated for ~100M positions)
    /// If not specified, calculated as ceil(100_000_000 / batch_size)
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// LR scheduler gamma (decay rate per step)
    #[arg(long, default_value = "0.992")]
    lr_gamma: f32,

    /// LR scheduler step interval (apply gamma every N superbatches)
    #[arg(long, default_value = "1")]
    lr_step: usize,

    /// Start superbatch number (useful for resuming)
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Batch queue size (number of batches to prefetch)
    #[arg(long, default_value = "64")]
    batch_queue_size: usize,

    /// Resume from checkpoint path (e.g., checkpoints/v47/v47b-69)
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Only re-quantise checkpoint (no training, requires --resume)
    #[arg(long)]
    quantise_only: bool,

    /// Use win rate model (score -> win probability conversion)
    /// When enabled, converts evaluation score using:
    ///   p = (score - 270.0) / 380.0
    ///   pm = (-score - 270.0) / 380.0
    ///   win_rate = 0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
    #[arg(long)]
    win_rate_model: bool,

    /// Apply WRM to network output in loss (nnue-pytorch-nodchip style).
    /// Value is the in_scaling parameter (nodchip default: 340).
    /// Requires --win-rate-model. When set, loss becomes |WRM_in(net) - WRM_out(target)|^2
    /// instead of |sigmoid(net) - WRM_out(target)|^2.
    #[arg(long, requires = "win_rate_model")]
    wrm_in_scaling: Option<f32>,
}

impl Args {
    /// WDL lambda の値（デフォルト 0.5）
    fn wdl_value(&self) -> f32 {
        self.wdl.unwrap_or(0.5)
    }

    /// WDL値が [0.0, 1.0] の範囲内であることを検証
    fn validate_wdl_range(name: &str, value: f32) -> Result<(), String> {
        if (0.0..=1.0).contains(&value) {
            Ok(())
        } else {
            Err(format!("--{} must be between 0.0 and 1.0 (got {})", name, value))
        }
    }

    /// Validates WDL-related arguments and creates the appropriate scheduler.
    fn create_wdl_scheduler(&self) -> Result<wdl::WdlSchedulerEnum, String> {
        match (self.start_wdl, self.end_wdl) {
            (Some(start), Some(end)) => {
                Self::validate_wdl_range("start-wdl", start)?;
                Self::validate_wdl_range("end-wdl", end)?;
                Ok(wdl::WdlSchedulerEnum::linear(start, end))
            }
            // clap の requires で排他制御済みだが念のため
            (Some(_), None) => Err("--start-wdl requires --end-wdl".to_string()),
            (None, Some(_)) => Err("--end-wdl requires --start-wdl".to_string()),
            (None, None) => {
                let wdl = self.wdl_value();
                Self::validate_wdl_range("wdl", wdl)?;
                Ok(wdl::WdlSchedulerEnum::constant(wdl))
            }
        }
    }

    /// Returns a display string for the WDL configuration.
    fn wdl_display(&self) -> String {
        match (self.start_wdl, self.end_wdl) {
            (Some(start), Some(end)) => format!("Linear ({} -> {})", start, end),
            _ => format!("Constant ({})", self.wdl_value()),
        }
    }

    fn validate_wrm_settings(&self) -> Result<(), String> {
        if let Some(in_scaling) = self.wrm_in_scaling {
            if !in_scaling.is_finite() || in_scaling <= 0.0 {
                return Err(format!("--wrm-in-scaling must be a positive finite value (got {})", in_scaling));
            }
        }
        Ok(())
    }
}

// =============================================================================
// Experiment Log Structures
// =============================================================================

#[derive(Serialize, Clone)]
struct ExperimentLog {
    id: String,
    name: String,
    date: String,
    status: String,
    last_updated_at: String,
    commit: String,
    command: String,
    params: ExperimentParams,
    data: ExperimentData,
    results: ExperimentResults,
    history: Vec<LossEntry>,
    checkpoints: Vec<String>,
}

#[derive(Serialize, Clone)]
struct ExperimentResults {
    training_time_seconds: u64,
    fv_scale: i32,
    best_loss: Option<f64>,
    best_loss_superbatch: Option<usize>,
}

#[derive(Serialize, Clone)]
struct ExperimentParams {
    l1: usize,
    l2: usize,
    l3: usize,
    lr: f32,
    lr_gamma: f32,
    lr_step: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    superbatches: usize,
    start_superbatch: usize,
    wdl: f32,
    start_wdl: Option<f32>,
    end_wdl: Option<f32>,
    scale: i32,
    weight_decay: f32,
    win_rate_model: bool,
    optimizer: String,
    activation: String,
    features: String,
    pairwise: bool,
    output_format: String,
    qa: i16,
    qb: i16,
}

#[derive(Serialize, Clone)]
struct ExperimentData {
    name: String,
    positions: Option<u64>,
    total_positions: u64,
    dataset_passes: Option<f64>,
}

#[derive(Serialize, Clone)]
struct LossEntry {
    superbatch: usize,
    loss: f64,
}

// =============================================================================
// Experiment Log Helper Functions
// =============================================================================

fn get_git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
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

/// `prior` (resume 元 experiment.json から引き継いだ history) と
/// `current` (現在 process の log.txt を parse した history) を superbatch でマージ。
/// 同一 superbatch が両方にある場合は current を採用する (再学習で値が更新された場合に対応)。
fn merge_loss_histories(prior: &[LossEntry], current: &[LossEntry]) -> Vec<LossEntry> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<usize, f64> = BTreeMap::new();
    for entry in prior {
        map.insert(entry.superbatch, entry.loss);
    }
    for entry in current {
        map.insert(entry.superbatch, entry.loss);
    }
    map.into_iter().map(|(superbatch, loss)| LossEntry { superbatch, loss }).collect()
}

fn parse_loss_history(log_path: &std::path::Path) -> Vec<LossEntry> {
    use std::collections::BTreeMap;
    let content = match std::fs::read_to_string(log_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
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

fn collect_checkpoints(output_dir: &std::path::Path, net_id: &str) -> Vec<String> {
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
        let a_num: usize = a[prefix.len()..].parse().unwrap_or(0);
        let b_num: usize = b[prefix.len()..].parse().unwrap_or(0);
        a_num.cmp(&b_num)
    });
    checkpoints
}

struct ExperimentContext {
    output_dir: std::path::PathBuf,
    net_id: String,
    command: String,
    params: ExperimentParams,
    data_name: String,
    superbatches: usize,
    fv_scale: i32,
    /// 学習開始時に確定するID・日時・コミット（以後不変）
    experiment_id: String,
    experiment_date: String,
    commit: String,
    training_start: std::time::Instant,
    /// データファイルの総局面数（初期化時に計算、以後不変）
    positions: u64,
    /// resume 時に既存 experiment.json から引き継いだ history。
    /// build_experiment_log() で現在 process の history とマージされる。
    prior_history: Vec<LossEntry>,
    /// resume 時に既存 experiment.json から引き継いだ累積学習時間 (秒)。
    /// build_experiment_log() で現在 process の経過時間に加算される。
    prior_training_seconds: u64,
}

impl ExperimentContext {
    fn new(
        output_dir: std::path::PathBuf,
        net_id: String,
        command: String,
        params: ExperimentParams,
        data_name: String,
        superbatches: usize,
        fv_scale: i32,
    ) -> Self {
        let commit = get_git_commit();
        let (id_ts, date) = get_timestamp();
        let id = format!("{}-{}", id_ts, &net_id);

        const PACKED_SFEN_VALUE_SIZE: u64 = 40;
        let positions: u64 = data_name
            .split(',')
            .filter_map(|path| std::fs::metadata(path.trim()).ok())
            .map(|meta| meta.len() / PACKED_SFEN_VALUE_SIZE)
            .sum();

        Self {
            output_dir,
            net_id,
            command,
            params,
            data_name,
            superbatches,
            fv_scale,
            experiment_id: id,
            experiment_date: date,
            commit,
            training_start: std::time::Instant::now(),
            positions,
            prior_history: Vec::new(),
            prior_training_seconds: 0,
        }
    }

    fn build_experiment_log(&self, status: &str) -> ExperimentLog {
        let latest_checkpoint = collect_checkpoints(&self.output_dir, &self.net_id)
            .last()
            .cloned()
            .unwrap_or_else(|| format!("{}-{}", self.net_id, self.superbatches));
        let log_path = self.output_dir.join(&latest_checkpoint).join("log.txt");
        let current_history = parse_loss_history(&log_path);
        // resume 時は過去 run の history を引き継いだ上で、現在 process の history を上書き合成する。
        // log.txt は checkpoint 単位で current process の error_record から書き直されるため、
        // prior_history を持っていないと sb 1..=resume_point の loss が experiment.json から消える。
        let history = merge_loss_histories(&self.prior_history, &current_history);

        let checkpoints = collect_checkpoints(&self.output_dir, &self.net_id);

        // 実際に完了したsuperbatch数から計算（中間保存時に最終予定値を使わない）
        let actual_superbatches = history.last().map(|e| e.superbatch).unwrap_or(0) as u64;
        let total_positions =
            self.params.batch_size as u64 * self.params.batches_per_superbatch as u64 * actual_superbatches;
        let dataset_passes = if self.positions > 0 { total_positions as f64 / self.positions as f64 } else { 0.0 };

        let (best_loss, best_loss_superbatch) = history
            .iter()
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal))
            .map(|entry| (Some(entry.loss), Some(entry.superbatch)))
            .unwrap_or((None, None));

        let training_time_seconds = self.prior_training_seconds.saturating_add(self.training_start.elapsed().as_secs());
        let (_, last_updated_at) = get_timestamp();

        ExperimentLog {
            id: self.experiment_id.clone(),
            name: self.net_id.clone(),
            date: self.experiment_date.clone(),
            status: status.to_string(),
            last_updated_at,
            commit: self.commit.clone(),
            command: self.command.clone(),
            params: self.params.clone(),
            data: ExperimentData {
                name: self.data_name.clone(),
                positions: Some(self.positions),
                total_positions,
                dataset_passes: Some(dataset_passes),
            },
            results: ExperimentResults {
                training_time_seconds,
                fv_scale: self.fv_scale,
                best_loss,
                best_loss_superbatch,
            },
            history,
            checkpoints,
        }
    }

    fn write_experiment_json(&self, status: &str) -> std::io::Result<()> {
        let experiment = self.build_experiment_log(status);
        let json = serde_json::to_string_pretty(&experiment).map_err(std::io::Error::other)?;
        let json_dir = self.output_dir.join(&self.net_id);
        std::fs::create_dir_all(&json_dir)?;
        let json_path = json_dir.join("experiment.json");
        std::fs::write(&json_path, &json)?;
        println!("Experiment log saved to {} (status: {})", json_path.display(), status);
        Ok(())
    }

    /// resume 時に既存 experiment.json から experiment_id / date / history を引き継ぐ。
    ///
    /// `ExperimentContext::new()` は呼ばれるたびに新しい timestamp ベースの
    /// experiment_id を生成するため、resume 時にそのまま `write_experiment_json`
    /// すると過去 run の experiment.json を別 ID で上書きしてしまい、
    /// 履歴が分断される。本メソッドは resume 元の experiment.json を読んで
    /// id / date を引き継ぐことで、resume が同一実験の続きとして記録されるようにする。
    ///
    /// 加えて `history` 配列も読み込み、`build_experiment_log` で現在 process の
    /// loss history とマージできるようにする。これがないと
    /// log.txt は checkpoint 単位で error_record から書き直されるため、
    /// resume 後の experiment.json から sb 1..=resume_point の loss が消える。
    fn inherit_resume_experiment_id(&mut self) {
        let json_path = self.output_dir.join(&self.net_id).join("experiment.json");
        if !json_path.exists() {
            return;
        }
        let content = match std::fs::read_to_string(&json_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let existing: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return,
        };

        if let Some(id) = existing.get("id").and_then(|v| v.as_str()) {
            if !id.is_empty() {
                println!("Inheriting experiment id from {}: {}", json_path.display(), id);
                self.experiment_id = id.to_string();
            }
        }
        if let Some(date) = existing.get("date").and_then(|v| v.as_str()) {
            if !date.is_empty() {
                self.experiment_date = date.to_string();
            }
        }
        if let Some(secs) =
            existing.get("results").and_then(|v| v.get("training_time_seconds")).and_then(|v| v.as_u64())
        {
            if secs > 0 {
                println!("Inheriting prior training time: {} seconds", secs);
                self.prior_training_seconds = secs;
            }
        }
        if let Some(arr) = existing.get("history").and_then(|v| v.as_array()) {
            let mut history: Vec<LossEntry> = arr
                .iter()
                .filter_map(|entry| {
                    let sb = entry.get("superbatch").and_then(|v| v.as_u64())? as usize;
                    let loss = entry.get("loss").and_then(|v| v.as_f64())?;
                    Some(LossEntry { superbatch: sb, loss })
                })
                .collect();
            history.sort_by_key(|e| e.superbatch);
            if !history.is_empty() {
                println!(
                    "Inheriting {} history entries from previous run (sb {} .. {})",
                    history.len(),
                    history.first().unwrap().superbatch,
                    history.last().unwrap().superbatch,
                );
                self.prior_history = history;
            }
        }
    }
}

// =============================================================================
// Architecture Definition
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct Architecture {
    l1: usize, // Accumulator size
    l2: usize, // Hidden layer 1 size
    l3: usize, // Hidden layer 2 size
}

impl Architecture {
    /// Get architecture from preset name
    fn from_preset(name: &str) -> Option<Self> {
        match name {
            "256x2-32-32" => Some(Self { l1: 256, l2: 32, l3: 32 }),
            "512x2-8-96" => Some(Self { l1: 512, l2: 8, l3: 96 }),
            "512x2-32-32" => Some(Self { l1: 512, l2: 32, l3: 32 }),
            "1024x2-8-32" => Some(Self { l1: 1024, l2: 8, l3: 32 }),
            "1024x2-16-64" => Some(Self { l1: 1024, l2: 16, l3: 64 }),
            _ => None,
        }
    }

    /// List of available presets
    fn available_presets() -> &'static [&'static str] {
        &["256x2-32-32", "512x2-8-96", "512x2-32-32", "1024x2-8-32", "1024x2-16-64"]
    }

    /// Display string
    fn display(&self) -> String {
        format!("{}x2-{}-{}", self.l1, self.l2, self.l3)
    }
}

// =============================================================================
// SIMD Padding Utilities
// =============================================================================

/// 32バイトアライメントにパディング
fn pad32(size: usize) -> usize {
    size.div_ceil(32) * 32
}

// =============================================================================
// NNUE-pytorch 互換ヘッダー計算
// =============================================================================

/// fc_hash計算
///
/// InputSlice hash: 0xEC42E90D
/// Layer hash base: 0xCC03DAE4
/// ClippedReLU hash: 0x538D24C7
fn compute_fc_hash(l1_size: usize, l2_size: usize, l3_size: usize) -> u32 {
    // InputSlice hash
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= (l1_size * 2) as u32;

    // Fully connected layers: [l1, l2, output]
    let layer_sizes = [l2_size, l3_size, 1usize];
    for (i, &out_features) in layer_sizes.iter().enumerate() {
        let mut layer_hash: u32 = 0xCC03DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;

        // Clipped ReLU hash (not for output layer)
        if i < 2 {
            layer_hash = layer_hash.wrapping_add(0x538D24C7);
        }
        prev_hash = layer_hash;
    }

    prev_hash
}

/// 特徴量hash値を取得
fn get_feature_hash(features: FeatureSet) -> u32 {
    use bullet_lib::game::inputs::{FEATURE_HASH, FEATURE_HASH_HM_V2, FEATURE_HASH_NONMIRROR};
    match features {
        FeatureSet::HalfKP => FEATURE_HASH,
        FeatureSet::HalfkaHm => FEATURE_HASH_HM_V2,
        FeatureSet::Halfka => FEATURE_HASH_NONMIRROR,
    }
}

/// nnue-pytorch形式のdescription文字列を生成
fn build_nnue_description(feature_set: FeatureSet, l1_size: usize, l2_size: usize, l3_size: usize) -> String {
    let (feature_name, input_size) = match feature_set {
        FeatureSet::HalfKP => ("HalfKP(Friend)", 125388usize),
        FeatureSet::HalfkaHm => ("HalfKA_hm(Friend)", 73305usize),
        FeatureSet::Halfka => ("HalfKA(Friend)", 138510usize),
    };

    // YaneuraOu互換のdescription文字列
    // 第1層は AffineTransformSparseInput を使用
    let description = format!(
        "Features={}[{}->{}x2],Network=AffineTransform[1<-{}](ClippedReLU[{}](AffineTransform[{}<-{}](ClippedReLU[{}](AffineTransformSparseInput[{}<-{}](InputSlice[{}(0:{})])))))",
        feature_name,
        input_size,
        l1_size,
        l3_size,     // Output layer input
        l3_size,     // L2 output / L3 input
        l3_size,     // L2 output features
        l2_size,     // L2 input features
        l2_size,     // L1 output / L2 input
        l2_size,     // L1 output features
        l1_size * 2, // L1 input (accumulator x2)
        l1_size * 2, // InputSlice size
        l1_size * 2  // InputSlice range
    );

    description
}

/// standard 用に重みをパディング
///
/// standard は SIMD 最適化のため、各層の入力次元を32の倍数にパディングする。
/// 例: 入力次元8 → パディング後32 (24個の0を追加)
///
/// # Arguments
/// * `weights` - row-major の重み [out_dim * in_dim]
/// * `out_dim` - 出力次元
/// * `in_dim` - 入力次元 (パディング前)
fn pad_weights_for_simd(weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let padded_in_dim = pad32(in_dim);

    // パディング不要な場合はそのまま返す
    if padded_in_dim == in_dim {
        return weights.to_vec();
    }

    let mut result = vec![0.0f32; out_dim * padded_in_dim];

    for o in 0..out_dim {
        for i in 0..in_dim {
            result[o * padded_in_dim + i] = weights[o * in_dim + i];
        }
        // 残りは0で埋める (既にvec![0.0; ...]で初期化済み)
    }

    result
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args = Args::parse();
    args.validate_wrm_settings().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });

    // Determine architecture
    let mut arch = Architecture::from_preset(&args.arch).unwrap_or_else(|| {
        eprintln!("Unknown architecture preset: {}", args.arch);
        eprintln!("Available presets: {:?}", Architecture::available_presets());
        std::process::exit(1);
    });

    // Override with individual settings
    if let Some(l1) = args.l1 {
        arch.l1 = l1;
    }
    if let Some(l2) = args.l2 {
        arch.l2 = l2;
    }
    if let Some(l3) = args.l3 {
        arch.l3 = l3;
    }

    let l1_size = arch.l1;
    let l2_size = arch.l2;
    let l3_size = arch.l3;

    // Quantization factors
    let qa = args.qa;
    let qb = args.qb;

    // Feature set info
    let (feature_name, input_size) = match args.features {
        FeatureSet::HalfkaHm => ("HalfKA_hm", ShogiHalfKA_hm.num_inputs()),
        FeatureSet::Halfka => ("HalfKA", ShogiHalfKA.num_inputs()),
        FeatureSet::HalfKP => ("HalfKP", ShogiHalfKP.num_inputs()),
    };

    // Optimizer name
    let optimizer_name = match args.optimizer {
        OptimizerType::AdamW => "AdamW",
        OptimizerType::RAdam => "RAdam",
        OptimizerType::Ranger => "Ranger (RAdam + Lookahead)",
    };

    // Activation function name
    let activation_name = match args.activation {
        ActivationType::Screlu => "SCReLU",
        ActivationType::Crelu => "CReLU",
    };

    // Pairwise mode
    let pairwise_enabled = matches!(args.pairwise, PairwiseMode::On);
    let pairwise_name = if pairwise_enabled { "On" } else { "Off" };

    // L1 input dimension (halved when pairwise is enabled)
    let l1_input_dim = if pairwise_enabled { l1_size } else { 2 * l1_size };

    // Validate QA and activation combination (skip confirmation for --quantise-only)
    // Reckless/Stockfish: Pairwise uses QA=255 with CReLU
    // Traditional: CReLU uses QA=127, SCReLU uses QA=255
    let recommended_qa = match (args.activation, pairwise_enabled) {
        (ActivationType::Screlu, _) => 255,    // SCReLU always uses QA=255
        (ActivationType::Crelu, true) => 255,  // Pairwise + CReLU uses QA=255 (Reckless compatible)
        (ActivationType::Crelu, false) => 127, // Traditional CReLU uses QA=127
    };
    if qa != recommended_qa && !args.quantise_only {
        eprintln!(
            "WARNING: QA={} is not recommended for {} activation{}.",
            qa,
            activation_name,
            if pairwise_enabled { " with pairwise" } else { "" }
        );
        eprintln!("         Recommended: --qa {}", recommended_qa);
        eprintln!("         Using non-standard QA may cause evaluation scale mismatch.");
        eprintln!();
        eprint!("Continue anyway? [y/N]: ");
        use std::io::{self, Write};
        io::stderr().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_lowercase();
        if input != "y" && input != "yes" {
            eprintln!("Aborted.");
            std::process::exit(1);
        }
        eprintln!();
    }

    // Warn about pairwise + SCReLU combination
    if pairwise_enabled && matches!(args.activation, ActivationType::Screlu) {
        eprintln!("WARNING: --pairwise on with SCReLU is unusual.");
        eprintln!("         Pairwise multiplication is typically combined with CReLU.");
        eprintln!("         Consider: --pairwise on --activation crelu --qa 255");
        eprintln!();
    }

    // Print configuration
    println!("=== Shogi NNUE Training ===");
    println!("Features: {} ({} dimensions)", feature_name, input_size);
    println!("Architecture: {} (L1={}, L2={}, L3={})", arch.display(), l1_size, l2_size, l3_size);
    if pairwise_enabled {
        println!(
            "Network: {} -> {}x2 -> pairwise_mul -> {} -> {} -> {} -> 1",
            input_size, l1_size, l1_input_dim, l2_size, l3_size
        );
    } else {
        println!("Network: {} -> {}x2 -> {} -> {} -> 1", input_size, l1_size, l2_size, l3_size);
    }
    println!("Activation: {}", activation_name);
    println!("Pairwise: {} (L1 input = {})", pairwise_name, l1_input_dim);
    println!("Win rate model: {}", if args.win_rate_model { "enabled" } else { "disabled" });
    if let Some(in_scaling) = args.wrm_in_scaling {
        println!("WRM in_scaling: {} (network output WRM enabled)", in_scaling);
    }
    println!("Optimizer: {}", optimizer_name);
    println!("Weight decay: {}", args.weight_decay);
    println!("Scale: {}", args.scale);
    println!("Quantization: QA={}, QB={}", qa, qb);
    let batches_per_superbatch_display =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
    let positions_per_superbatch = batches_per_superbatch_display as u64 * args.batch_size as u64;
    println!("Batch size: {}", args.batch_size);
    println!(
        "Batches/superbatch: {} (~{}M positions)",
        batches_per_superbatch_display,
        positions_per_superbatch / 1_000_000
    );
    println!("Superbatches: {} (start={})", args.superbatches, args.start_superbatch);
    println!("Learning rate: {} (gamma={}, step={})", args.lr, args.lr_gamma, args.lr_step);
    println!("WDL lambda: {}", args.wdl_display());
    println!("Save rate: {}", args.save_rate);
    println!("Threads: {} (queue={})", args.threads, args.batch_queue_size);
    println!("Output: {}", args.output.display());
    println!("Net ID: {}", args.net_id);
    println!("Data: {}", args.data);
    println!("===========================");

    // Capture data for experiment JSON before args.net_id is moved
    let output_format_name = match args.output_format {
        OutputFormat::Bullet => "bullet",
        OutputFormat::Standard => "standard",
    };
    let experiment_params = ExperimentParams {
        l1: l1_size,
        l2: l2_size,
        l3: l3_size,
        lr: args.lr,
        lr_gamma: args.lr_gamma,
        lr_step: args.lr_step,
        batch_size: args.batch_size,
        batches_per_superbatch: batches_per_superbatch_display,
        superbatches: args.superbatches,
        start_superbatch: args.start_superbatch,
        wdl: args.wdl_value(),
        start_wdl: args.start_wdl,
        end_wdl: args.end_wdl,
        scale: args.scale,
        weight_decay: args.weight_decay,
        win_rate_model: args.win_rate_model,
        optimizer: optimizer_name.to_string(),
        activation: activation_name.to_string(),
        features: feature_name.to_string(),
        pairwise: pairwise_enabled,
        output_format: output_format_name.to_string(),
        qa: args.qa,
        qb: args.qb,
    };
    let experiment_quantise_only = args.quantise_only;
    let experiment_fv_scale = (i32::from(args.qa) * i32::from(args.qb) + args.scale / 2) / args.scale;
    let mut experiment_ctx = ExperimentContext::new(
        args.output.clone(),
        args.net_id.clone(),
        std::env::args().collect::<Vec<_>>().join(" "),
        experiment_params,
        args.data.clone(),
        args.superbatches,
        experiment_fv_scale,
    );

    // Create WDL scheduler
    let wdl_scheduler = args.create_wdl_scheduler().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });

    // Training schedule
    let batches_per_superbatch =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
    let schedule = TrainingSchedule {
        net_id: args.net_id,
        eval_scale: args.scale as f32,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch,
            start_superbatch: args.start_superbatch,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler,
        lr_scheduler: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
        save_rate: args.save_rate,
    };

    // resume の場合は experiment_id を引き継ぐ。on_checkpoint_saved closure が
    // experiment_ctx を不変借用する前に行う必要がある。
    if !experiment_quantise_only && args.resume.is_some() {
        experiment_ctx.inherit_resume_experiment_id();
    }

    // Local settings
    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let on_checkpoint_saved = |_superbatch: usize| {
        if let Err(e) = experiment_ctx.write_experiment_json("running") {
            eprintln!("Warning: Failed to update experiment JSON: {}", e);
        }
    };
    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: args.batch_queue_size,
        on_checkpoint_saved: if experiment_quantise_only { None } else { Some(&on_checkpoint_saved) },
    };

    // Data loader (use existing file for --quantise-only to avoid file check)
    let data_files_owned: Vec<String> = if args.quantise_only {
        // Use any existing file - we won't actually load data
        let resume_path = args.resume.as_ref().expect("--quantise-only requires --resume");
        let quantised = resume_path.join("quantised.bin");
        if quantised.exists() {
            vec![quantised.to_str().unwrap().to_string()]
        } else {
            // Fallback: use raw.bin
            vec![resume_path.join("raw.bin").to_str().unwrap().to_string()]
        }
    } else {
        args.data.split(',').map(|s| s.to_string()).collect()
    };
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();
    let data_loader = DirectSequentialDataLoader::new(&data_files_ref);

    // SavedFormat configuration
    // This directly outputs the final format for your engine.
    // Customize as needed:
    //   - .transpose() to change matrix layout
    //   - SavedFormat::custom(bytes) to add headers
    //   - .transform(|store, vals| ...) for custom transformations
    let save_format: Vec<SavedFormat> = match args.output_format {
        OutputFormat::Bullet => {
            // bullet format: all i16 (default)
            vec![
                SavedFormat::id("l0w").round().quantise::<i16>(qa),
                SavedFormat::id("l0b").round().quantise::<i16>(qa),
                SavedFormat::id("l1w").round().quantise::<i16>(qb),
                SavedFormat::id("l1b").round().quantise::<i16>(qa * qb),
                SavedFormat::id("l2w").round().quantise::<i16>(qb),
                SavedFormat::id("l2b").round().quantise::<i16>(qa * qb),
                SavedFormat::id("outw").round().quantise::<i16>(qb),
                SavedFormat::id("outb").round().quantise::<i16>(qa * qb),
            ]
        }
        OutputFormat::Standard => {
            // standard format: NNUE header + L0 i16 + L1-Out biases i32 + weights i8
            //
            // File layout:
            // - Header: version (u32), network_hash (u32), desc_len (u32), description
            // - FeatureTransformer layer hash (u32)
            // - L0: biases i16[L1], weights i16[INPUT×L1]
            // - Network layer hash (u32)
            // - L1: biases i32[L2], weights i8[L2×(L1*2)]
            // - L2: biases i32[L3], weights i8[L3×L2]
            // - Output: biases i32[1], weights i8[1×L3]

            // NNUE version (YaneuraOu/Stockfish compatible)
            const NNUE_VERSION: u32 = 0x7AF32F16;

            // Compute hashes (nnue-pytorch compatible)
            let feature_hash = get_feature_hash(args.features);
            let fc_hash = compute_fc_hash(l1_size, l2_size, l3_size);
            // network_hash = fc_hash ^ feature_hash ^ (l1_size * 2)
            let network_hash = fc_hash ^ feature_hash ^ ((l1_size * 2) as u32);

            // Build nnue-pytorch compatible description string
            let description = build_nnue_description(args.features, l1_size, l2_size, l3_size);
            let desc_bytes = description.as_bytes();

            // Build header (nnue-pytorch format)
            let mut header = Vec::new();
            header.extend_from_slice(&NNUE_VERSION.to_le_bytes());
            header.extend_from_slice(&network_hash.to_le_bytes());
            header.extend_from_slice(&(desc_bytes.len() as u32).to_le_bytes());
            header.extend_from_slice(desc_bytes);

            // FeatureTransformer layer hash (nnue-pytorch format: feature_hash ^ (l1_size * 2))
            let ft_hash = (feature_hash ^ ((l1_size * 2) as u32)).to_le_bytes().to_vec();
            // Network layer hash (fc_hash)
            let network_hash_bytes = fc_hash.to_le_bytes().to_vec();

            // L1バイアスのスケール:
            // L1層入力スケールは活性化関数の出力スケールに依存:
            //
            // | 活性化関数 | QA  | 出力スケール | L1 bias scale |
            // |------------|-----|--------------|---------------|
            // | CReLU      | 127 | 127          | 127 × qb      |
            // | CReLU      | 255 | 255          | 255 × qb      |
            // | SCReLU     | 255 | 127 (x²>>9)  | 127 × qb      |
            // | Pairwise   | 255 | 127 (ab>>9)  | 127 × qb      |
            //
            // 注: SCReLU/Pairwise は QA=255 でも出力が 127 にスケールダウンされる
            let l1_bias_scale = match (args.activation, pairwise_enabled, qa) {
                // Pairwise: (qa * qa) >> shift で 127 スケール
                (_, true, _) => {
                    let qa_i32 = i32::from(qa);
                    let shift = if qa >= 255 { 9 } else { 7 };
                    ((qa_i32 * qa_i32) >> shift) * i32::from(qb)
                }
                // SCReLU QA=255: x² >> 9 で 127 スケール
                (ActivationType::Screlu, false, qa) if qa >= 255 => 127 * i32::from(qb),
                // CReLU / その他: qa スケール
                _ => i32::from(qa) * i32::from(qb),
            };

            vec![
                // Header
                SavedFormat::custom(header),
                // FeatureTransformer layer hash
                SavedFormat::custom(ft_hash),
                // L0: biases first, then weights (standard order)
                SavedFormat::id("l0b").round().quantise::<i16>(qa),
                SavedFormat::id("l0w").round().quantise::<i16>(qa),
                // Network layer hash
                SavedFormat::custom(network_hash_bytes),
                // L1-Output層の重みは .transpose() で row-major に変換
                // 理由: Stockfish/nnue-pytorch は row-major で推論する
                // bullet 内部は column-major だが、これは GPU (cuBLAS) 最適化のため
                // 変換コストは出力時の1回のみで、学習効率には影響しない
                //
                // 重要: standard は SIMD 最適化のため 32バイトアライメントを要求
                // 各層の入力次元を pad32() でパディングする必要がある
                //
                // L1: biases i32, weights i8 (row-major, padded)
                // 入力次元: l1_input_dim → pad32(l1_input_dim)
                // Pairwise時はl1_size、通常時は2*l1_size
                SavedFormat::id("l1b").round().quantise::<i32>(l1_bias_scale),
                SavedFormat::id("l1w")
                    .transpose()
                    .transform({
                        let out_dim = l2_size;
                        let in_dim = l1_input_dim;
                        move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
                    })
                    .round()
                    .quantise::<i8>(qb),
                // L2: biases i32, weights i8 (row-major, padded)
                // 入力次元: l2 → pad32(l2)
                // L2入力スケール: crelu_i32_to_u8 後は常に 127 スケール
                SavedFormat::id("l2b").round().quantise::<i32>(127 * i32::from(qb)),
                SavedFormat::id("l2w")
                    .transpose()
                    .transform({
                        let out_dim = l3_size;
                        let in_dim = l2_size;
                        move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
                    })
                    .round()
                    .quantise::<i8>(qb),
                // Output: biases i32, weights i8 (row-major, padded)
                // 入力次元: l3 → pad32(l3)
                // Output入力スケール: crelu_i32_to_u8 後は常に 127 スケール
                SavedFormat::id("outb").round().quantise::<i32>(127 * i32::from(qb)),
                SavedFormat::id("outw")
                    .transpose()
                    .transform({
                        let out_dim = 1;
                        let in_dim = l3_size;
                        move |_, vals| pad_weights_for_simd(&vals, out_dim, in_dim)
                    })
                    .round()
                    .quantise::<i8>(qb),
            ]
        }
    };

    type Nbn<'a> = NetworkBuilderNode<'a, BackendMarker>;

    /// Loss function: WRM applied to network output (nodchip style).
    fn loss_fn_wrm<'a>(output: Nbn<'a>, target: Nbn<'a>) -> Nbn<'a> {
        let params =
            *WRM_LOSS_PARAMS.get().expect("WRM loss parameters must be initialized before building the trainer");
        let offset = 270.0f32;
        let scorenet = output * params.nnue2score;
        let q = ((scorenet.copy() - offset) / params.in_scaling).sigmoid();
        let qm = ((-scorenet - offset) / params.in_scaling).sigmoid();
        let qf = (1.0 + q - qm) * 0.5;
        qf.squared_error(target)
    }

    /// Loss function: standard sigmoid
    fn loss_fn_sigmoid<'a>(output: Nbn<'a>, target: Nbn<'a>) -> Nbn<'a> {
        output.sigmoid().squared_error(target)
    }

    let loss_fn: for<'a> fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a> = if let Some(in_scaling) = args.wrm_in_scaling {
        WRM_LOSS_PARAMS
            .set(WrmLossParams { nnue2score: args.scale as f32, in_scaling })
            .expect("WRM loss parameters should only be initialized once");
        loss_fn_wrm
    } else {
        loss_fn_sigmoid
    };

    // Network builder macro with SCReLU activation (no pairwise)
    macro_rules! build_trainer_screlu {
        ($opt:expr, $input:expr, $use_win_rate:expr) => {{
            let mut builder = ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .save_format(&save_format)
                .loss_fn(loss_fn);
            if $use_win_rate {
                builder = builder.use_win_rate_model();
            }
            builder.build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", input_size, l1_size);
                let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
                let l2 = builder.new_affine("l2", l2_size, l3_size);
                let out = builder.new_affine("out", l3_size, 1);

                let stm_hidden = l0.forward(stm_inputs).screlu();
                let ntm_hidden = l0.forward(ntm_inputs).screlu();
                let combined = stm_hidden.concat(ntm_hidden);

                let hidden1 = l1.forward(combined).screlu();
                let hidden2 = l2.forward(hidden1).screlu();

                out.forward(hidden2)
            })
        }};
    }

    // Network builder macro with SCReLU activation + pairwise multiplication
    macro_rules! build_trainer_screlu_pairwise {
        ($opt:expr, $input:expr, $use_win_rate:expr) => {{
            let mut builder = ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .save_format(&save_format)
                .loss_fn(loss_fn);
            if $use_win_rate {
                builder = builder.use_win_rate_model();
            }
            builder.build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", input_size, l1_size);
                let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
                let l2 = builder.new_affine("l2", l2_size, l3_size);
                let out = builder.new_affine("out", l3_size, 1);

                // SCReLU + pairwise_mul (unusual but supported)
                let stm_hidden = l0.forward(stm_inputs).screlu().pairwise_mul();
                let ntm_hidden = l0.forward(ntm_inputs).screlu().pairwise_mul();
                let combined = stm_hidden.concat(ntm_hidden);

                let hidden1 = l1.forward(combined).screlu();
                let hidden2 = l2.forward(hidden1).screlu();

                out.forward(hidden2)
            })
        }};
    }

    // Network builder macro with CReLU (Clipped ReLU) activation (no pairwise)
    macro_rules! build_trainer_crelu {
        ($opt:expr, $input:expr, $use_win_rate:expr) => {{
            let mut builder = ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .save_format(&save_format)
                .loss_fn(loss_fn);
            if $use_win_rate {
                builder = builder.use_win_rate_model();
            }
            builder.build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", input_size, l1_size);
                let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
                let l2 = builder.new_affine("l2", l2_size, l3_size);
                let out = builder.new_affine("out", l3_size, 1);

                let stm_hidden = l0.forward(stm_inputs).crelu();
                let ntm_hidden = l0.forward(ntm_inputs).crelu();
                let combined = stm_hidden.concat(ntm_hidden);

                let hidden1 = l1.forward(combined).crelu();
                let hidden2 = l2.forward(hidden1).crelu();

                out.forward(hidden2)
            })
        }};
    }

    // Network builder macro with CReLU activation + pairwise multiplication
    // This is the recommended combination for pairwise multiplication
    macro_rules! build_trainer_crelu_pairwise {
        ($opt:expr, $input:expr, $use_win_rate:expr) => {{
            let mut builder = ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .save_format(&save_format)
                .loss_fn(loss_fn);
            if $use_win_rate {
                builder = builder.use_win_rate_model();
            }
            builder.build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", input_size, l1_size);
                let l1 = builder.new_affine("l1", l1_input_dim, l2_size);
                let l2 = builder.new_affine("l2", l2_size, l3_size);
                let out = builder.new_affine("out", l3_size, 1);

                // CReLU + pairwise_mul (recommended combination)
                let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
                let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
                let combined = stm_hidden.concat(ntm_hidden);

                let hidden1 = l1.forward(combined).crelu();
                let hidden2 = l2.forward(hidden1).crelu();

                out.forward(hidden2)
            })
        }};
    }

    // Helper macro to either run training or just re-quantise
    macro_rules! maybe_run_or_quantise {
        ($trainer:expr) => {{
            if args.quantise_only {
                let resume_path = args.resume.as_ref().expect("--quantise-only requires --resume");
                let resume_str = resume_path.to_str().unwrap();
                println!("Loading checkpoint from {}...", resume_str);
                $trainer.load_from_checkpoint(resume_str);

                // Create output directory if needed
                let output_dir = args.output.to_str().unwrap_or("checkpoints");
                let output_path = format!("{}/requantised.bin", output_dir);
                std::fs::create_dir_all(output_dir).unwrap_or(());

                println!("Saving re-quantised weights to {}...", output_path);
                $trainer.save_quantised(&output_path).expect("Failed to save quantised weights");
                println!("Done!");
            } else {
                if let Some(ref resume_path) = args.resume {
                    let resume_str = resume_path.to_str().unwrap();
                    println!("Resuming from checkpoint: {}", resume_str);
                    $trainer.load_from_checkpoint(resume_str);
                }
                $trainer.run(&schedule, &settings, &data_loader);
            }
        }};
    }

    // Run training macro (to reduce duplication across feature sets, activations, and pairwise)
    macro_rules! run_training {
        ($input:expr, screlu, false, $win_rate:expr) => {{
            let weight_decay = args.weight_decay;
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer = build_trainer_screlu!(optimiser::AdamW, $input, $win_rate);
                    trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::RAdam => {
                    let mut trainer = build_trainer_screlu!(optimiser::RAdam, $input, $win_rate);
                    let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger => {
                    let mut trainer = build_trainer_screlu!(optimiser::Ranger, $input, $win_rate);
                    trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
            }
        }};
        ($input:expr, screlu, true, $win_rate:expr) => {{
            let weight_decay = args.weight_decay;
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer = build_trainer_screlu_pairwise!(optimiser::AdamW, $input, $win_rate);
                    trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::RAdam => {
                    let mut trainer = build_trainer_screlu_pairwise!(optimiser::RAdam, $input, $win_rate);
                    let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger => {
                    let mut trainer = build_trainer_screlu_pairwise!(optimiser::Ranger, $input, $win_rate);
                    trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
            }
        }};
        ($input:expr, crelu, false, $win_rate:expr) => {{
            let weight_decay = args.weight_decay;
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer = build_trainer_crelu!(optimiser::AdamW, $input, $win_rate);
                    trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::RAdam => {
                    let mut trainer = build_trainer_crelu!(optimiser::RAdam, $input, $win_rate);
                    let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger => {
                    let mut trainer = build_trainer_crelu!(optimiser::Ranger, $input, $win_rate);
                    trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
            }
        }};
        ($input:expr, crelu, true, $win_rate:expr) => {{
            let weight_decay = args.weight_decay;
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer = build_trainer_crelu_pairwise!(optimiser::AdamW, $input, $win_rate);
                    trainer.optimiser.set_params(AdamWParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::RAdam => {
                    let mut trainer = build_trainer_crelu_pairwise!(optimiser::RAdam, $input, $win_rate);
                    let params: RAdamParams = RAdamParams { decay: weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger => {
                    let mut trainer = build_trainer_crelu_pairwise!(optimiser::Ranger, $input, $win_rate);
                    trainer.optimiser.set_params(RangerParams { decay: weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
            }
        }};
    }

    // Run training based on feature set, activation, and pairwise mode
    let use_win_rate_model = args.win_rate_model;
    match (args.features, args.activation, pairwise_enabled) {
        (FeatureSet::HalfkaHm, ActivationType::Screlu, false) => {
            run_training!(ShogiHalfKA_hm, screlu, false, use_win_rate_model)
        }
        (FeatureSet::HalfkaHm, ActivationType::Screlu, true) => {
            run_training!(ShogiHalfKA_hm, screlu, true, use_win_rate_model)
        }
        (FeatureSet::HalfkaHm, ActivationType::Crelu, false) => {
            run_training!(ShogiHalfKA_hm, crelu, false, use_win_rate_model)
        }
        (FeatureSet::HalfkaHm, ActivationType::Crelu, true) => {
            run_training!(ShogiHalfKA_hm, crelu, true, use_win_rate_model)
        }
        (FeatureSet::Halfka, ActivationType::Screlu, false) => {
            run_training!(ShogiHalfKA, screlu, false, use_win_rate_model)
        }
        (FeatureSet::Halfka, ActivationType::Screlu, true) => {
            run_training!(ShogiHalfKA, screlu, true, use_win_rate_model)
        }
        (FeatureSet::Halfka, ActivationType::Crelu, false) => {
            run_training!(ShogiHalfKA, crelu, false, use_win_rate_model)
        }
        (FeatureSet::Halfka, ActivationType::Crelu, true) => {
            run_training!(ShogiHalfKA, crelu, true, use_win_rate_model)
        }
        (FeatureSet::HalfKP, ActivationType::Screlu, false) => {
            run_training!(ShogiHalfKP, screlu, false, use_win_rate_model)
        }
        (FeatureSet::HalfKP, ActivationType::Screlu, true) => {
            run_training!(ShogiHalfKP, screlu, true, use_win_rate_model)
        }
        (FeatureSet::HalfKP, ActivationType::Crelu, false) => {
            run_training!(ShogiHalfKP, crelu, false, use_win_rate_model)
        }
        (FeatureSet::HalfKP, ActivationType::Crelu, true) => {
            run_training!(ShogiHalfKP, crelu, true, use_win_rate_model)
        }
    }

    // Generate final experiment JSON (status: completed)
    if !experiment_quantise_only {
        if let Err(e) = experiment_ctx.write_experiment_json("completed") {
            eprintln!("Warning: Failed to generate experiment JSON: {}", e);
        }
    }
}

// =============================================================================
// Inference Network Structure (reference for engine integration)
// =============================================================================

/// Square Clipped ReLU - activation function
#[inline]
fn _screlu(x: i16, qa: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(qa));
    y * y
}
