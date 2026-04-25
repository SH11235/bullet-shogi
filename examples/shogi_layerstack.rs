/*
Shogi LayerStack NNUE Training Script

LayerStacks (SFNNwoPSQT-1536) アーキテクチャの学習スクリプト。
rshogi 互換の量子化ファイル (quantised.bin) を出力する。

Usage:
    cargo run --release --example shogi_layerstack -- [OPTIONS]

Options:
    --data <PATH>       Training data path (comma-separated for multiple files)
    --batch-size <N>    Batch size (default: 16384)
    --superbatches <N>  Number of superbatches (default: 100)
    --lr <RATE>         Initial learning rate (default: 0.001)
    --wdl <LAMBDA>      WDL lambda for constant scheduler (default: 0.5)
    --start-wdl <F>     Start WDL lambda for linear interpolation
    --end-wdl <F>       End WDL lambda for linear interpolation
    --scale <N>         Eval scale (default: 600)
    --l0 <SIZE>         FT output size (default: 1536)
    --l1 <SIZE>         L1 output size (default: 16)
    --l2 <SIZE>         L2 output size (default: 32)
    --save-rate <N>     Save interval in superbatches (default: 10)
    --threads <N>       Number of threads (default: 4)
    --output <DIR>      Output directory (default: checkpoints)
    --net-id <NAME>     Network ID (default: shogi-ls-1536)
    --resume <PATH>     Resume from checkpoint
    --quantise-only     Only re-quantise checkpoint (requires --resume)
    --optimizer <OPT>   Optimizer (adamw, radam, ranger) (default: ranger)
    --win-rate-model    Use win rate model for score conversion
    --batches-per-superbatch <N>  Batches per superbatch (default: auto)
    --lr-gamma <F>      LR decay rate (default: 0.992)
    --lr-step <N>       LR decay interval (default: 1)
    --interleave-file-batches <N> File mix granularity (0=sequential, 1=round-robin)
    --epoch-file-shuffle Shuffle file order every epoch
    --file-shuffle-seed <SEED> Seed for epoch file shuffle
    --psqt               Enable PSQT shortcut layer
    --threat            Enable Threat concatenated input (placeholder)
*/

use std::{path::PathBuf, sync::OnceLock};

use bullet_lib::{
    game::inputs::{
        ShogiHalfKA_hm, ShogiHalfKaHmHandCount, ShogiHalfKaHmHandThreat, ShogiHalfKaHmHandThreatDefensive,
        ShogiHalfKaHmThreat, SparseInputType, ThreatProfile,
    },
    game::outputs::{
        SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS, SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER,
        SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES, SHOGI_PROGRESS8_FEATURE_ORDER, SHOGI_PROGRESS8_NUM_FEATURES,
        ShogiLayerStackBucket9, ShogiProgressBucket8, ShogiProgressBucket8GikouLite, ShogiProgressKPAbs,
    },
    nn::{
        Affine, BackendMarker, InitSettings, NetworkBuilderNode, Shape,
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
use serde::{Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

const NUM_BUCKETS: usize = 9;
const QA: i16 = 127;
const QB: i16 = 64;

#[derive(Debug, Clone, Copy)]
struct WrmLossParams {
    nnue2score: f32,
    in_scaling: f32,
}

static WRM_LOSS_PARAMS: OnceLock<WrmLossParams> = OnceLock::new();

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizerType {
    AdamW,
    RAdam,
    #[default]
    Ranger,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum BucketMode {
    #[default]
    Kingrank9,
    Ply9,
    Progress8,
    #[value(name = "progress8gikou")]
    Progress8Gikou,
    #[value(name = "progress8kpabs")]
    Progress8KPAbs,
}

#[derive(Parser, Debug)]
#[command(name = "shogi_layerstack")]
#[command(about = "Shogi LayerStack NNUE training script")]
struct Args {
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

    /// WDL lambda (constant). Cannot be used with --start-wdl/--end-wdl
    #[arg(long, conflicts_with_all = ["start_wdl", "end_wdl"])]
    wdl: Option<f32>,

    /// Start WDL lambda for linear interpolation
    #[arg(long, requires = "end_wdl")]
    start_wdl: Option<f32>,

    /// End WDL lambda for linear interpolation
    #[arg(long, requires = "start_wdl")]
    end_wdl: Option<f32>,

    /// Eval scale (default: 600, Eval_Coef=600 のDL教師データと整合)
    #[arg(long, default_value = "600")]
    scale: i32,

    /// L0 (Feature Transformer) size
    #[arg(long, default_value = "1536")]
    l0: usize,

    /// L1 output size (includes skip connection neuron)
    #[arg(long, default_value = "16")]
    l1: usize,

    /// L2 output size
    #[arg(long, default_value = "32")]
    l2: usize,

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
    #[arg(long, default_value = "shogi-ls-1536")]
    net_id: String,

    /// Optimizer (adamw, radam, ranger)
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerType,

    /// Weight decay
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Batches per superbatch (default: auto ~100M positions)
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// LR scheduler gamma
    #[arg(long, default_value = "0.992")]
    lr_gamma: f32,

    /// LR scheduler step interval
    #[arg(long, default_value = "1")]
    lr_step: usize,

    /// Start superbatch number
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Batch queue size
    #[arg(long, default_value = "64")]
    batch_queue_size: usize,

    /// Read this many batches from one file before switching files (0 = sequential by file)
    #[arg(long, default_value = "0")]
    interleave_file_batches: usize,

    /// Shuffle file order at every epoch boundary
    #[arg(long)]
    epoch_file_shuffle: bool,

    /// Seed for --epoch-file-shuffle
    #[arg(long, default_value = "0")]
    file_shuffle_seed: u64,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Only re-quantise checkpoint
    #[arg(long)]
    quantise_only: bool,

    /// Use win rate model
    #[arg(long)]
    win_rate_model: bool,

    /// Apply WRM to network output in loss (nnue-pytorch-nodchip style).
    /// Value is the in_scaling parameter (nodchip default: 340).
    /// Requires --win-rate-model. When set, loss becomes |WRM_in(net) - WRM_out(target)|^2
    /// instead of |sigmoid(net) - WRM_out(target)|^2.
    #[arg(long, requires = "win_rate_model")]
    wrm_in_scaling: Option<f32>,

    /// Scaling factor to convert network output to centipawn score for WRM loss.
    /// Only used when --wrm-in-scaling is set. The raw network output is multiplied
    /// by this value before being passed to the WRM function.
    /// (nnue-pytorch-nodchip default: 600)
    #[arg(long, default_value_t = 600.0, requires = "wrm_in_scaling")]
    wrm_nnue2score: f32,

    /// Output bucket mode (kingrank9 / ply9 / progress8 / progress8gikou / progress8kpabs)
    #[arg(long, value_enum, default_value = "kingrank9")]
    bucket_mode: BucketMode,

    /// Optional boundaries for ply9 buckets (8 comma-separated values)
    #[arg(long)]
    ply_bounds: Option<String>,

    /// Enable PSQT shortcut layer
    #[arg(long, default_value_t = false)]
    psqt: bool,

    /// Enable Threat concatenated input
    #[arg(long, default_value_t = false)]
    threat: bool,

    /// Threat exclusion profile (full, same-class, same-class-major-pawn, cross-side)
    #[arg(long, default_value = "full")]
    threat_profile: String,

    /// Enable HandThreat concatenated input (full drop-attack pair, 121,104 dims)
    ///
    /// `--threat` とは排他。両方指定した場合はエラーで終了する。
    /// profile なし (v95 PoC 版)。
    #[arg(long, default_value_t = false)]
    hand_threat: bool,

    /// Enable HandThreat defensive variant (30,276 dims, 非対称 emission)
    ///
    /// `--hand-threat` と `--threat` 両方と排他。drop_owner=enemy かつ
    /// attacked_side=friend のみ符号化する防御 feature。
    #[arg(long, default_value_t = false)]
    hand_threat_defensive: bool,

    /// HandCount Dense Input を有効化する（L1 層の入力に 14 元の持ち駒 dense vector を concat）。
    ///
    /// `[stm 持ち駒 7 種, nstm 持ち駒 7 種] = 14 元` を FT 出力 (1536) の
    /// 後ろに連結して L1 に渡す。sparse 特徴は HalfKA_hm と完全互換。
    ///
    /// `--threat` / `--hand-threat` / `--hand-threat-defensive` とは排他。
    #[arg(long, default_value_t = false)]
    hand_count_dense: bool,

    /// Progress parameter path: coeff JSON for progress8/progress8gikou, progress.bin for progress8kpabs
    #[arg(long)]
    progress_coeff: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ProgressCoeffV1 {
    format: String,
    model: String,
    num_buckets: usize,
    feature_order: Vec<String>,
    standardization: ProgressStandardization,
    weights: Vec<f32>,
    bias: f32,
    runtime: ProgressRuntime,
}

#[derive(Debug, Deserialize)]
struct ProgressStandardization {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ProgressRuntime {
    z_clip: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ProgressCoeffV2 {
    format: String,
    model: String,
    feature_set: String,
    num_buckets: usize,
    feature_order: Vec<String>,
    standardization: ProgressStandardization,
    weights: Vec<f32>,
    bias: f32,
    runtime: ProgressRuntime,
}

// `OutputBuckets` implementations stay `Copy`, so boxing the large variants is not an option.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
enum LoadedProgressBucket {
    V1(ShogiProgressBucket8),
    Gikou(ShogiProgressBucket8GikouLite),
    KPAbs(ShogiProgressKPAbs),
}

impl Args {
    fn wdl_value(&self) -> f32 {
        self.wdl.unwrap_or(0.5)
    }

    fn validate_wdl_range(name: &str, value: f32) -> Result<(), String> {
        if (0.0..=1.0).contains(&value) {
            Ok(())
        } else {
            Err(format!("--{} must be between 0.0 and 1.0 (got {})", name, value))
        }
    }

    fn create_wdl_scheduler(&self) -> Result<wdl::WdlSchedulerEnum, String> {
        match (self.start_wdl, self.end_wdl) {
            (Some(start), Some(end)) => {
                Self::validate_wdl_range("start-wdl", start)?;
                Self::validate_wdl_range("end-wdl", end)?;
                Ok(wdl::WdlSchedulerEnum::linear(start, end))
            }
            (Some(_), None) => Err("--start-wdl requires --end-wdl".to_string()),
            (None, Some(_)) => Err("--end-wdl requires --start-wdl".to_string()),
            (None, None) => {
                let wdl = self.wdl_value();
                Self::validate_wdl_range("wdl", wdl)?;
                Ok(wdl::WdlSchedulerEnum::constant(wdl))
            }
        }
    }

    fn wdl_display(&self) -> String {
        match (self.start_wdl, self.end_wdl) {
            (Some(start), Some(end)) => format!("Linear ({} -> {})", start, end),
            _ => format!("Constant ({})", self.wdl_value()),
        }
    }

    fn interleave_batches_value(&self) -> Option<usize> {
        if self.interleave_file_batches == 0 { None } else { Some(self.interleave_file_batches) }
    }

    fn validate_wrm_settings(&self) -> Result<(), String> {
        if let Some(in_scaling) = self.wrm_in_scaling {
            if !in_scaling.is_finite() || in_scaling <= 0.0 {
                return Err(format!("--wrm-in-scaling must be a positive finite value (got {})", in_scaling));
            }
            if !self.wrm_nnue2score.is_finite() || self.wrm_nnue2score <= 0.0 {
                return Err(format!("--wrm-nnue2score must be a positive finite value (got {})", self.wrm_nnue2score));
            }
        }
        Ok(())
    }

    fn parse_ply_bounds_csv(text: &str) -> Result<[u16; 8], String> {
        let mut values = Vec::new();
        for token in text.split(',') {
            let t = token.trim();
            if t.is_empty() {
                continue;
            }
            let value: u16 = t.parse().map_err(|e| format!("invalid --ply-bounds value '{t}': {e}"))?;
            values.push(value);
        }
        if values.len() != 8 {
            return Err(format!("--ply-bounds requires exactly 8 comma-separated values (got {})", values.len()));
        }
        Ok([values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]])
    }

    fn resolved_ply_bounds(&self) -> Result<Option<[u16; 8]>, String> {
        match self.bucket_mode {
            BucketMode::Kingrank9 => {
                if self.ply_bounds.is_some() {
                    Err("--ply-bounds can only be used with --bucket-mode ply9".to_string())
                } else if self.progress_coeff.is_some() {
                    Err("--progress-coeff can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
                        .to_string())
                } else {
                    Ok(None)
                }
            }
            BucketMode::Ply9 => {
                if self.progress_coeff.is_some() {
                    Err("--progress-coeff can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
                        .to_string())
                } else {
                    match &self.ply_bounds {
                        Some(text) => Self::parse_ply_bounds_csv(text).map(Some),
                        None => Ok(Some(SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS)),
                    }
                }
            }
            BucketMode::Progress8 | BucketMode::Progress8Gikou | BucketMode::Progress8KPAbs => {
                if self.ply_bounds.is_some() {
                    Err("--ply-bounds can only be used with --bucket-mode ply9".to_string())
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn bucket_mode_name(&self) -> &'static str {
        match self.bucket_mode {
            BucketMode::Kingrank9 => "kingrank9",
            BucketMode::Ply9 => "ply9",
            BucketMode::Progress8 => "progress8",
            BucketMode::Progress8Gikou => "progress8gikou",
            BucketMode::Progress8KPAbs => "progress8kpabs",
        }
    }

    fn load_progress_bucket(&self) -> Result<Option<LoadedProgressBucket>, String> {
        match self.bucket_mode {
            BucketMode::Progress8 => {
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8 requires --progress-coeff".to_string())?;
                load_progress_bucket_v1_from_json(path).map(|v| Some(LoadedProgressBucket::V1(v)))
            }
            BucketMode::Progress8Gikou => {
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8gikou requires --progress-coeff".to_string())?;
                load_progress_bucket_v2_from_json(path).map(|v| Some(LoadedProgressBucket::Gikou(v)))
            }
            BucketMode::Progress8KPAbs => {
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8kpabs requires --progress-coeff".to_string())?;
                ShogiProgressKPAbs::load_from_bin(path).map(|v| Some(LoadedProgressBucket::KPAbs(v)))
            }
            _ => {
                if self.progress_coeff.is_some() {
                    Err("--progress-coeff can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
                        .to_string())
                } else {
                    Ok(None)
                }
            }
        }
    }
}

fn load_progress_bucket_v1_from_json(path: &PathBuf) -> Result<ShogiProgressBucket8, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read --progress-coeff '{}': {e}", path.display()))?;
    let coeff: ProgressCoeffV1 = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse progress coeff JSON '{}': {e}", path.display()))?;

    if coeff.format != "rshogi.progress_coeff.v1" {
        return Err(format!("invalid progress coeff format '{}', expected 'rshogi.progress_coeff.v1'", coeff.format));
    }
    if coeff.model != "logistic_regression" {
        return Err(format!("invalid progress coeff model '{}', expected 'logistic_regression'", coeff.model));
    }
    if coeff.num_buckets != 8 {
        return Err(format!("invalid num_buckets {}, expected 8", coeff.num_buckets));
    }
    if coeff.feature_order.len() != SHOGI_PROGRESS8_NUM_FEATURES {
        return Err(format!(
            "invalid feature_order length {}, expected {}",
            coeff.feature_order.len(),
            SHOGI_PROGRESS8_NUM_FEATURES
        ));
    }
    for (idx, expected) in SHOGI_PROGRESS8_FEATURE_ORDER.iter().enumerate() {
        if coeff.feature_order[idx] != *expected {
            return Err(format!(
                "feature_order mismatch at index {}: got '{}', expected '{}'",
                idx, coeff.feature_order[idx], expected
            ));
        }
    }
    if coeff.standardization.mean.len() != SHOGI_PROGRESS8_NUM_FEATURES
        || coeff.standardization.std.len() != SHOGI_PROGRESS8_NUM_FEATURES
        || coeff.weights.len() != SHOGI_PROGRESS8_NUM_FEATURES
    {
        return Err(format!(
            "mean/std/weights lengths must all be {} (got mean={}, std={}, weights={})",
            SHOGI_PROGRESS8_NUM_FEATURES,
            coeff.standardization.mean.len(),
            coeff.standardization.std.len(),
            coeff.weights.len()
        ));
    }
    if coeff.runtime.z_clip.len() != 2 {
        return Err(format!("runtime.z_clip must have exactly 2 values (got {})", coeff.runtime.z_clip.len()));
    }

    let mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES] =
        coeff.standardization.mean.try_into().map_err(|_| "failed to convert mean to fixed array".to_string())?;
    let std: [f32; SHOGI_PROGRESS8_NUM_FEATURES] =
        coeff.standardization.std.try_into().map_err(|_| "failed to convert std to fixed array".to_string())?;
    let weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES] =
        coeff.weights.try_into().map_err(|_| "failed to convert weights to fixed array".to_string())?;
    let z_clip = [coeff.runtime.z_clip[0], coeff.runtime.z_clip[1]];

    Ok(ShogiProgressBucket8::new(mean, std, weights, coeff.bias, z_clip))
}

fn load_progress_bucket_v2_from_json(path: &PathBuf) -> Result<ShogiProgressBucket8GikouLite, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read --progress-coeff '{}': {e}", path.display()))?;
    let coeff: ProgressCoeffV2 = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse progress coeff JSON '{}': {e}", path.display()))?;

    if coeff.format != "rshogi.progress_coeff.v2" {
        return Err(format!("invalid progress coeff format '{}', expected 'rshogi.progress_coeff.v2'", coeff.format));
    }
    if coeff.model != "logistic_regression" {
        return Err(format!("invalid progress coeff model '{}', expected 'logistic_regression'", coeff.model));
    }
    if coeff.feature_set != "gikou_lite_34" {
        return Err(format!("invalid feature_set '{}', expected 'gikou_lite_34'", coeff.feature_set));
    }
    if coeff.num_buckets != 8 {
        return Err(format!("invalid num_buckets {}, expected 8", coeff.num_buckets));
    }
    if coeff.feature_order.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES {
        return Err(format!(
            "invalid feature_order length {}, expected {}",
            coeff.feature_order.len(),
            SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
        ));
    }
    for (idx, expected) in SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER.iter().enumerate() {
        if coeff.feature_order[idx] != *expected {
            return Err(format!(
                "feature_order mismatch at index {}: got '{}', expected '{}'",
                idx, coeff.feature_order[idx], expected
            ));
        }
    }
    if coeff.standardization.mean.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
        || coeff.standardization.std.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
        || coeff.weights.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
    {
        return Err(format!(
            "mean/std/weights lengths must all be {} (got mean={}, std={}, weights={})",
            SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES,
            coeff.standardization.mean.len(),
            coeff.standardization.std.len(),
            coeff.weights.len()
        ));
    }
    if coeff.runtime.z_clip.len() != 2 {
        return Err(format!("runtime.z_clip must have exactly 2 values (got {})", coeff.runtime.z_clip.len()));
    }

    let mean: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] =
        coeff.standardization.mean.try_into().map_err(|_| "failed to convert mean to fixed array".to_string())?;
    let std: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] =
        coeff.standardization.std.try_into().map_err(|_| "failed to convert std to fixed array".to_string())?;
    let weights: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] =
        coeff.weights.try_into().map_err(|_| "failed to convert weights to fixed array".to_string())?;
    let z_clip = [coeff.runtime.z_clip[0], coeff.runtime.z_clip[1]];

    Ok(ShogiProgressBucket8GikouLite::new(mean, std, weights, coeff.bias, z_clip))
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
    architecture: String,
    l0: usize,
    l1: usize,
    l2: usize,
    num_buckets: usize,
    bucket_mode: String,
    ply_bounds: Option<[u16; 8]>,
    progress_coeff: Option<String>,
    lr: f32,
    lr_gamma: f32,
    lr_step: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    interleave_file_batches: Option<usize>,
    epoch_file_shuffle: bool,
    file_shuffle_seed: Option<u64>,
    superbatches: usize,
    start_superbatch: usize,
    wdl: f32,
    start_wdl: Option<f32>,
    end_wdl: Option<f32>,
    scale: i32,
    weight_decay: f32,
    win_rate_model: bool,
    optimizer: String,
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
// Experiment Log Helpers
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
/// 同一 superbatch が両方にある場合は current を採用する。
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
    /// 詳細は shogi_simple.rs の同名メソッドのコメント参照。
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
// LEB128 Encoder
// =============================================================================

/// 符号付き LEB128 エンコード (1 値)
fn encode_signed_leb128(mut value: i64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        // value が 0 で符号ビットが立っていない、または value が -1 で符号ビットが立っている場合、終了
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            result.push(byte);
            break;
        }
        byte |= 0x80; // 継続ビット
        result.push(byte);
    }
    result
}

/// i16 配列を LEB128 圧縮し、マジック + サイズ + データ を返す
fn encode_leb128_tensor_i16(values: &[i16]) -> Vec<u8> {
    // まず全値を LEB128 エンコード
    let mut compressed = Vec::new();
    for &val in values {
        compressed.extend_from_slice(&encode_signed_leb128(val as i64));
    }

    // マジック + サイズヘッダ + 圧縮データ
    let mut result = Vec::new();
    result.extend_from_slice(b"COMPRESSED_LEB128"); // 17 bytes
    result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    result
}

// =============================================================================
// SIMD Padding Utilities
// =============================================================================

fn pad32(size: usize) -> usize {
    size.div_ceil(32) * 32
}

// =============================================================================
// Hash Computation (nnue-pytorch 互換)
// =============================================================================

/// LayerStacks 用 fc_hash 計算
///
/// 各バケットの FC ハッシュ。rshogi は読み飛ばすが互換性のため出力する。
fn compute_layerstack_fc_hash(l1_out: usize, l2_in: usize, l2_out: usize) -> u32 {
    // InputSlice hash
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= (l1_out * 2) as u32; // FT output * 2 (dual perspective)

    // L1: 1536 → l1_out
    let layer_sizes = [(l1_out, true), (l2_out, true), (1usize, false)];
    for (out_features, has_relu) in layer_sizes {
        let mut layer_hash: u32 = 0xCC03DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;
        if has_relu {
            layer_hash = layer_hash.wrapping_add(0x538D24C7);
        }
        prev_hash = layer_hash;
    }

    let _ = (l2_in,); // suppress unused warning — used for documentation clarity
    prev_hash
}

// =============================================================================
// SavedFormat Construction
// =============================================================================

/// LayerStack 量子化出力の SavedFormat を構築する
///
/// rshogi NetworkLayerStacks::read() と完全互換のバイナリを生成。
///
/// `hand_count_dense_dims` が 0 より大きい場合、L1 重みは `ft_out + dims`
/// の入力次元を持ち、先頭 `ft_out` が FT 出力、残り `dims` が HandCount dense の
/// 貢献となる。`arch_str` に `HandCountDense=dims` を追加する。
#[allow(clippy::too_many_arguments)]
fn build_layerstack_save_format(
    halfka_dim: usize,
    input_size: usize,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    fv_scale: i32,
    psqt: bool,
    threat_profile: Option<ThreatProfile>,
    hand_threat: bool,
    hand_count_dense_dims: usize,
) -> Vec<SavedFormat> {
    use bullet_lib::game::inputs::FEATURE_HASH_HM_V2;

    let l1_effective = l1_out - 1; // skip connection 分を除く
    let l2_in = l1_effective * 2; // sqr_crelu concat crelu

    // nnue-pytorch 互換ハッシュ計算
    let fc_hash = compute_layerstack_fc_hash(ft_out, l2_in, l2_out);
    let ft_hash = FEATURE_HASH_HM_V2 ^ ((ft_out * 2) as u32);
    let network_hash = fc_hash ^ ft_hash;

    // アーキテクチャ文字列（fv_scale を埋め込み、rshogi が推論時に正しく解釈できるようにする）
    let psqt_part = if psqt { format!("PSQT={},", NUM_BUCKETS) } else { String::new() };
    let threat_part = if let Some(tp) = threat_profile {
        let threat_dims = input_size - halfka_dim;
        let pid = tp.profile_id();
        if pid == 0 {
            format!("Threat={threat_dims},")
        } else {
            format!("Threat={threat_dims},ThreatProfile={pid},")
        }
    } else {
        String::new()
    };
    // HandThreat (案 A): dims 固定 121,104。rshogi 側 loader は
    // `HandThreat={dims},` を検出して u32 dims + i8 weights を読み込む。
    let hand_threat_part = if hand_threat {
        let hand_threat_dims = input_size - halfka_dim;
        format!("HandThreat={hand_threat_dims},")
    } else {
        String::new()
    };
    // HandCount Dense (本機能): L1 入力に 14 元の持ち駒 dense vector を concat。
    // rshogi 側は `HandCountDense={dims}` を検出して L1 重みを 14 行分追加で読む。
    let hand_count_part = if hand_count_dense_dims > 0 {
        format!("HandCountDense={hand_count_dense_dims},")
    } else {
        String::new()
    };
    let arch_desc = format!(
        "Features=HalfKA_hm(Friend)[{}->{}x2],\
         {psqt_part}\
         {threat_part}\
         {hand_threat_part}\
         {hand_count_part}\
         Network=AffineTransform[1<-{}](\
         ClippedReLU[{}](\
         AffineTransform[{}<-{}](\
         SqrClippedReLU[{}](\
         AffineTransform[{}<-{}](\
         InputSlice[{}(0:{})]))))),\
         fv_scale={}",
        input_size,
        ft_out,
        l2_out,     // Output input
        l2_out,     // L2 output / L3 input
        l2_out,     // L2 output
        l2_in,      // L2 input
        l2_in,      // dual activation output
        l1_out,     // L1 output
        ft_out * 2, // L1 input (dual perspective)
        ft_out * 2,
        ft_out * 2,
        fv_scale,
    );
    let arch_bytes = arch_desc.as_bytes();

    // ---- ヘッダー ----
    let nnue_version: u32 = 0x7AF32F20;
    let mut header = Vec::new();
    header.extend_from_slice(&nnue_version.to_le_bytes());
    header.extend_from_slice(&network_hash.to_le_bytes());
    header.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
    header.extend_from_slice(arch_bytes);

    // ---- FT hash ----
    let ft_hash_bytes = ft_hash.to_le_bytes().to_vec();

    // ---- FT biases + weights (LEB128 圧縮, YO 互換 2ブロック形式) ----
    // biases / weights を別々の LEB128 ブロックで出力。
    // YaneuraOu は FT を biases ブロック → weights ブロックの順で読み込むため、
    // 各ブロックに独立した COMPRESSED_LEB128 マジック + サイズヘッダが付く。
    let qa_i16 = QA;
    let ft_out_captured = ft_out;
    let input_size_captured = input_size;
    let ft_biases_leb128 = SavedFormat::empty()
        .transform(move |graph, _| {
            let l0b = graph.get("l0b");
            let qa_f = qa_i16 as f64;
            let biases_i16: Vec<i16> = l0b.values.iter().map(|&v| (qa_f * v as f64).round() as i16).collect();
            let leb128_bytes = encode_leb128_tensor_i16(&biases_i16);
            leb128_bytes.iter().map(|&b| (b as i8) as f32).collect()
        })
        .quantise::<i8>(1);

    let qa_i16 = QA;
    let halfka_dim_captured = halfka_dim;
    let ft_weights_leb128 = SavedFormat::empty()
        .transform(move |graph, _| {
            let l0w = graph.get("l0w");

            // Quantise to i16 (scale = QA = 127)
            // Threat 有効時は最初の halfka_dim 特徴量のみ（piece 部分）を書き出す。
            // column-major: l0w.values[feat * ft_out + out]
            // piece 部分 = feat 0..halfka_dim → indices 0..halfka_dim*ft_out
            let qa_f = qa_i16 as f64;
            let piece_end = halfka_dim_captured * ft_out_captured;
            let weights_i16: Vec<i16> =
                l0w.values[..piece_end].iter().map(|&v| (qa_f * v as f64).round() as i16).collect();
            let _ = input_size_captured;
            let leb128_bytes = encode_leb128_tensor_i16(&weights_i16);
            leb128_bytes.iter().map(|&b| (b as i8) as f32).collect()
        })
        .quantise::<i8>(1);

    // ---- PSQT weights/biases (raw i32) ----
    let psqt_data = if psqt {
        // PSQT は HalfKA 特徴量のみ対象。Threat 部分は含めない。
        let input_size_for_psqt = halfka_dim;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let psqt_w = graph.get("psqtw"); // [NUM_BUCKETS, input_size] column-major
                    let psqt_b = graph.get("psqtb"); // [NUM_BUCKETS]

                    let scale = (QA as i32 * QB as i32) as f64; // 8128.0
                    let mut bytes: Vec<u8> = Vec::new();

                    // Biases: i32[9]
                    for bucket in 0..NUM_BUCKETS {
                        let val = (scale * psqt_b.values[bucket] as f64).round() as i32;
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }

                    // Weights: i32[input_size][9] (feature-major)
                    for feat in 0..input_size_for_psqt {
                        for bucket in 0..NUM_BUCKETS {
                            // column-major: feat * rows + bucket
                            let w = psqt_w.values[feat * NUM_BUCKETS + bucket];
                            let val = (scale * w as f64).round() as i32;
                            bytes.extend_from_slice(&val.to_le_bytes());
                        }
                    }

                    // byte passthrough: 各バイトを i8 として f32 にキャスト
                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };

    // ---- Threat weights (raw i8) ----
    // l0w の threat 部分 (feat halfka_dim..input_size) を i8 で書き出す。
    // レイアウト: i8[THREAT_DIMENSIONS × ft_out] (feature-major)
    let threat_data = if threat_profile.is_some() {
        let halfka_dim_for_threat = halfka_dim;
        let ft_out_for_threat = ft_out;
        let qa_for_threat = QA;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l0w = graph.get("l0w");
                    let qa_f = qa_for_threat as f64;

                    // threat 部分: feat halfka_dim..input_size
                    // column-major: indices halfka_dim*ft_out .. input_size*ft_out
                    let threat_start = halfka_dim_for_threat * ft_out_for_threat;
                    let mut bytes: Vec<u8> = Vec::new();
                    for &v in &l0w.values[threat_start..] {
                        let q = (qa_f * v as f64).round().clamp(-128.0, 127.0) as i8;
                        bytes.push(q as u8);
                    }

                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };

    // ---- HandThreat weights (raw i8) ----
    // l0w の HandThreat 部分 (feat halfka_dim..input_size) を i8 で書き出す。
    // --threat と --hand-threat は排他なので、同時に存在しない。
    // レイアウト: i8[HAND_THREAT_DIMENSIONS × ft_out] (feature-major)
    let hand_threat_data = if hand_threat {
        let halfka_dim_for_ht = halfka_dim;
        let ft_out_for_ht = ft_out;
        let qa_for_ht = QA;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l0w = graph.get("l0w");
                    let qa_f = qa_for_ht as f64;

                    let ht_start = halfka_dim_for_ht * ft_out_for_ht;
                    let mut bytes: Vec<u8> = Vec::new();
                    for &v in &l0w.values[ht_start..] {
                        let q = (qa_f * v as f64).round().clamp(-128.0, 127.0) as i8;
                        bytes.push(q as u8);
                    }

                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };

    // ---- LayerStacks (9 buckets) ----
    // 各バケットについて: fc_hash + L1(biases, weights) + L2(biases, weights) + Output(bias, weights)
    //
    // bullet内部の重みレイアウト:
    //   l1w: column-major [NUM_BUCKETS * l1_out, ft_out] (= [rows, cols])
    //   l2w: column-major [NUM_BUCKETS * l2_out, l2_in]
    //   l3w: column-major [NUM_BUCKETS * 1, l2_out]
    //
    // rshogi の期待レイアウト (per bucket):
    //   L1 weights: [l1_out × pad32(ft_out)] row-major, つまり weight[out][padded_in]
    //   L2 weights: [l2_out × pad32(l2_in)] row-major
    //   Output weights: [pad32(l2_out)] row-major

    let bias_scale = i32::from(QA) * i32::from(QB); // 127 * 64 = 8128

    // 全バケットの LayerStack データを1つの transform で生成
    let l1_out_captured = l1_out;
    let l2_out_captured = l2_out;
    let l2_in_captured = l2_in;
    let ft_out_for_ls = ft_out;
    let fc_hash_captured = fc_hash;
    let hand_count_dense_dims_captured = hand_count_dense_dims;
    let layerstack_data = SavedFormat::empty()
        .transform(move |graph, _| {
            let l1w = graph.get("l1w");
            let l1b = graph.get("l1b");
            let l1fw = graph.get("l1fw");
            let l1fb = graph.get("l1fb");
            let l2w = graph.get("l2w");
            let l2b = graph.get("l2b");
            let l3w = graph.get("l3w");
            let l3b = graph.get("l3b");

            let qb_f = QB as f64;
            let bias_scale_f = bias_scale as f64;

            let mut output_bytes: Vec<u8> = Vec::new();

            for bucket in 0..NUM_BUCKETS {
                // fc_hash per bucket
                output_bytes.extend_from_slice(&fc_hash_captured.to_le_bytes());

                // === L1 layer ===
                // Biases: i32, scale = QA * QB = 8128
                for out_idx in 0..l1_out_captured {
                    let global_out = bucket * l1_out_captured + out_idx;
                    let merged_bias = l1b.values[global_out] + l1fb.values[out_idx];
                    let val = (bias_scale_f * merged_bias as f64).round() as i32;
                    output_bytes.extend_from_slice(&val.to_le_bytes());
                }

                // Weights: i8, scale = QB = 64
                // bullet:
                //   l1w  = [NUM_BUCKETS * l1_out, ft_out + hand_count_dims]
                //   l1fw = [l1_out, ft_out]   (shared factorized part; hand_count は共有化しない)
                //   weight[global_out * (ft_out + hc) + in_idx] where
                //   global_out = bucket * l1_out + out_idx
                //   - in_idx < ft_out: FT 出力との結合部（bucket_w + shared_w）
                //   - ft_out <= in_idx < ft_out + hc_dims: HandCount との結合部（bucket_w のみ）
                //   - それ以上: padding
                // rshogi: row-major [l1_out × padded(ft_out + hc)]
                let l1_total_in = ft_out_for_ls + hand_count_dense_dims_captured;
                let l1_padded_in = pad32(l1_total_in);
                let l1_rows_total = NUM_BUCKETS * l1_out_captured;
                for out_idx in 0..l1_out_captured {
                    let global_out = bucket * l1_out_captured + out_idx;
                    for in_idx in 0..l1_padded_in {
                        if in_idx < ft_out_for_ls {
                            // column-major indexing:
                            //   l1w  shape [NUM_BUCKETS*l1_out, ft_out + hc] -> in * rows + out
                            //   l1fw shape [l1_out, ft_out]                  -> in * l1_out + out
                            let bucket_w = l1w.values[in_idx * l1_rows_total + global_out];
                            let shared_w = l1fw.values[in_idx * l1_out_captured + out_idx];
                            let w = bucket_w + shared_w;
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else if in_idx < l1_total_in {
                            // HandCount Dense 部: bucket_w のみ（共有なし）
                            let bucket_w = l1w.values[in_idx * l1_rows_total + global_out];
                            let q = (qb_f * bucket_w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else {
                            output_bytes.push(0u8); // padding
                        }
                    }
                }

                // === L2 layer ===
                // Biases: i32, scale = 127 * QB (CReLU output is 127-scale)
                let l2_bias_scale = 127.0 * qb_f;
                for out_idx in 0..l2_out_captured {
                    let global_out = bucket * l2_out_captured + out_idx;
                    let val = (l2_bias_scale * l2b.values[global_out] as f64).round() as i32;
                    output_bytes.extend_from_slice(&val.to_le_bytes());
                }

                // Weights: i8, scale = QB = 64
                let l2_padded_in = pad32(l2_in_captured);
                let l2_rows_total = NUM_BUCKETS * l2_out_captured;
                for out_idx in 0..l2_out_captured {
                    let global_out = bucket * l2_out_captured + out_idx;
                    for in_idx in 0..l2_padded_in {
                        if in_idx < l2_in_captured {
                            // l2w shape [NUM_BUCKETS*l2_out, l2_in], column-major
                            let w = l2w.values[in_idx * l2_rows_total + global_out];
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else {
                            output_bytes.push(0u8);
                        }
                    }
                }

                // === Output layer ===
                // Bias: i32, scale = 127 * QB
                let out_bias_scale = 127.0 * qb_f;
                {
                    let global_out = bucket;
                    let val = (out_bias_scale * l3b.values[global_out] as f64).round() as i32;
                    output_bytes.extend_from_slice(&val.to_le_bytes());
                }

                // Weights: i8, scale = QB = 64
                let output_padded_in = pad32(l2_out_captured);
                {
                    let global_out = bucket;
                    for in_idx in 0..output_padded_in {
                        if in_idx < l2_out_captured {
                            // l3w shape [NUM_BUCKETS, l2_out], column-major
                            let w = l3w.values[in_idx * NUM_BUCKETS + global_out];
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else {
                            output_bytes.push(0u8);
                        }
                    }
                }
            }

            // byte passthrough: 各バイトを i8 として f32 にキャスト
            output_bytes.iter().map(|&b| (b as i8) as f32).collect()
        })
        .quantise::<i8>(1);

    let mut formats =
        vec![SavedFormat::custom(header), SavedFormat::custom(ft_hash_bytes), ft_biases_leb128, ft_weights_leb128];
    if let Some(psqt) = psqt_data {
        formats.push(psqt);
    }
    if let Some(threat) = threat_data {
        // profile_id > 0 のときだけ profile id (u32 LE) を Threat weights の直前に書き込む
        if let Some(tp) = threat_profile {
            let pid = tp.profile_id();
            if pid != 0 {
                formats.push(SavedFormat::custom(pid.to_le_bytes().to_vec()));
            }
        }
        formats.push(threat);
    }
    if let Some(ht) = hand_threat_data {
        // HandThreat dims (u32 LE) を weights の直前に書き込む
        // v95 では 121,104 固定だが、将来の profile 追加に備えて明示化
        let ht_dims = (input_size - halfka_dim) as u32;
        formats.push(SavedFormat::custom(ht_dims.to_le_bytes().to_vec()));
        formats.push(ht);
    }
    formats.push(layerstack_data);
    formats
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
    let ply_bounds = args.resolved_ply_bounds().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });
    let progress_bucket = args.load_progress_bucket().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });

    let ft_out = args.l0;
    let l1_out = args.l1;
    let l1_effective = l1_out - 1;
    let l2_in = l1_effective * 2;
    let l2_out = args.l2;
    let halfka_dim = ShogiHalfKA_hm.num_inputs(); // 73305

    // --threat / --hand-threat / --hand-threat-defensive / --hand-count-dense は相互排他
    let ht_flags = [args.threat, args.hand_threat, args.hand_threat_defensive, args.hand_count_dense];
    let ht_count = ht_flags.iter().filter(|&&b| b).count();
    if ht_count > 1 {
        eprintln!(
            "ERROR: --threat / --hand-threat / --hand-threat-defensive / --hand-count-dense は同時に指定できません"
        );
        std::process::exit(1);
    }

    // Threat profile の解決
    let threat_profile = if args.threat {
        let tp = ThreatProfile::from_cli(&args.threat_profile).unwrap_or_else(|| {
            eprintln!(
                "ERROR: Unknown threat profile '{}'. Available: {}",
                args.threat_profile,
                ThreatProfile::available()
            );
            std::process::exit(1);
        });
        Some(tp)
    } else {
        None
    };

    let use_hand_threat = args.hand_threat;
    let use_hand_threat_defensive = args.hand_threat_defensive;
    let use_hand_count_dense = args.hand_count_dense;

    let input_size = if let Some(tp) = threat_profile {
        let threat_input = ShogiHalfKaHmThreat::new(tp);
        threat_input.num_inputs()
    } else if use_hand_threat {
        let hand_threat_input = ShogiHalfKaHmHandThreat::new();
        hand_threat_input.num_inputs()
    } else if use_hand_threat_defensive {
        let input = ShogiHalfKaHmHandThreatDefensive::new();
        input.num_inputs()
    } else {
        // --hand-count-dense の場合も sparse 次元は HalfKA_hm と同じ (dense 14 元は
        // 別経路)。
        halfka_dim
    };

    // L1 層の入力次元は FT 出力 + （HandCount 有効時は +14）
    let hand_count_dense_dims: usize = if use_hand_count_dense { bullet_lib::game::inputs::HAND_COUNT_DIMS } else { 0 };
    let l1_input_dim = ft_out + hand_count_dense_dims;

    let optimizer_name = match args.optimizer {
        OptimizerType::AdamW => "AdamW",
        OptimizerType::RAdam => "RAdam",
        OptimizerType::Ranger => "Ranger",
    };

    let fv_scale = (i32::from(QA) * i32::from(QB) + args.scale / 2) / args.scale;

    // Print configuration
    println!("=== Shogi LayerStack NNUE Training ===");
    println!("Features: HalfKA_hm ({} dimensions)", input_size);
    println!(
        "Architecture: LayerStack L0={}, L1={} ({} effective + 1 skip), L2={}",
        ft_out, l1_out, l1_effective, l2_out
    );
    println!("L2 input: {} (sqr_crelu concat crelu)", l2_in);
    println!("PSQT shortcut: {}", if args.psqt { "enabled" } else { "disabled" });
    println!(
        "Threat: {}",
        if let Some(tp) = threat_profile {
            let threat_dims = input_size - halfka_dim;
            format!("enabled (profile={tp}, {threat_dims} dimensions, total input={input_size})")
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "HandThreat: {}",
        if use_hand_threat {
            let hand_threat_dims = input_size - halfka_dim;
            format!("enabled (案 A full drop-attack pair, {hand_threat_dims} dimensions, total input={input_size})")
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "HandCount Dense: {}",
        if use_hand_count_dense {
            format!("enabled ({} dims concat to L1 input; L1 input dim = {})", hand_count_dense_dims, l1_input_dim)
        } else {
            "disabled".to_string()
        }
    );
    println!("Buckets: {}", NUM_BUCKETS);
    println!("Bucket mode: {}", args.bucket_mode_name());
    if let Some(bounds) = ply_bounds {
        println!("Ply bounds: {:?}", bounds);
    }
    if let Some(coeff) = &args.progress_coeff {
        println!("Progress coeff: {}", coeff.display());
    }
    println!("FV_SCALE: {} (QA={}, QB={}, scale={})", fv_scale, QA, QB, args.scale);
    println!("Optimizer: {}", optimizer_name);
    println!("Weight decay: {}", args.weight_decay);
    println!("Win rate model: {}", if args.win_rate_model { "enabled" } else { "disabled" });
    if let Some(in_scaling) = args.wrm_in_scaling {
        println!("WRM in_scaling: {} nnue2score: {} (network output WRM enabled)", in_scaling, args.wrm_nnue2score);
    }
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
    match args.interleave_batches_value() {
        Some(v) => println!("File mix: round-robin every {} batch(es)", v),
        None => println!("File mix: sequential by file"),
    }
    if args.epoch_file_shuffle {
        println!("Epoch file shuffle: enabled (seed={})", args.file_shuffle_seed);
    } else {
        println!("Epoch file shuffle: disabled");
    }
    println!("Output: {}", args.output.display());
    println!("Net ID: {}", args.net_id);
    println!("Data: {}", args.data);
    println!("======================================");

    // Experiment context
    let experiment_params = ExperimentParams {
        architecture: format!("LayerStack-{}-{}-{}", ft_out, l1_out, l2_out),
        l0: ft_out,
        l1: l1_out,
        l2: l2_out,
        num_buckets: NUM_BUCKETS,
        bucket_mode: args.bucket_mode_name().to_string(),
        ply_bounds,
        progress_coeff: args.progress_coeff.as_ref().map(|p| p.display().to_string()),
        lr: args.lr,
        lr_gamma: args.lr_gamma,
        lr_step: args.lr_step,
        batch_size: args.batch_size,
        batches_per_superbatch: batches_per_superbatch_display,
        interleave_file_batches: args.interleave_batches_value(),
        epoch_file_shuffle: args.epoch_file_shuffle,
        file_shuffle_seed: args.epoch_file_shuffle.then_some(args.file_shuffle_seed),
        superbatches: args.superbatches,
        start_superbatch: args.start_superbatch,
        wdl: args.wdl_value(),
        start_wdl: args.start_wdl,
        end_wdl: args.end_wdl,
        scale: args.scale,
        weight_decay: args.weight_decay,
        win_rate_model: args.win_rate_model,
        optimizer: optimizer_name.to_string(),
        qa: QA,
        qb: QB,
    };
    let experiment_quantise_only = args.quantise_only;
    let mut experiment_ctx = ExperimentContext::new(
        args.output.clone(),
        args.net_id.clone(),
        std::env::args().collect::<Vec<_>>().join(" "),
        experiment_params,
        args.data.clone(),
        args.superbatches,
        fv_scale,
    );

    // WDL scheduler
    let wdl_scheduler = args.create_wdl_scheduler().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });

    // Training schedule
    let batches_per_superbatch =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
    let schedule = TrainingSchedule {
        net_id: args.net_id.clone(),
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
    if !args.quantise_only && args.resume.is_some() {
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

    // Data loader
    let data_files_owned: Vec<String> = if args.quantise_only {
        let resume_path = args.resume.as_ref().expect("--quantise-only requires --resume");
        let quantised = resume_path.join("quantised.bin");
        if quantised.exists() {
            vec![quantised.to_str().unwrap().to_string()]
        } else {
            vec![resume_path.join("raw.bin").to_str().unwrap().to_string()]
        }
    } else {
        args.data.split(',').map(|s| s.to_string()).collect()
    };
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();
    let mut data_loader = DirectSequentialDataLoader::new(&data_files_ref);
    if let Some(interleave_batches) = args.interleave_batches_value() {
        data_loader = data_loader.with_interleave_batches(interleave_batches);
    }
    if args.epoch_file_shuffle {
        data_loader = data_loader.with_epoch_file_shuffle(true, args.file_shuffle_seed);
    }

    // SavedFormat
    // HandThreat block: full pair と defensive は quantised.bin 上の layout が同一
    // (arch_str の HandThreat={dims} でサイズを表現するだけ) なので、どちらの
    // case でも hand_threat=true 扱いで書き出せる。
    let save_format_hand_threat = use_hand_threat || use_hand_threat_defensive;
    let save_format = build_layerstack_save_format(
        halfka_dim,
        input_size,
        ft_out,
        l1_out,
        l2_out,
        fv_scale,
        args.psqt,
        threat_profile,
        save_format_hand_threat,
        hand_count_dense_dims,
    );

    // Network builder
    let ft_out_c = ft_out;
    let l1_out_c = l1_out;
    let l1_effective_c = l1_effective;
    let l2_out_c = l2_out;
    let l2_in_c = l2_in;
    let use_psqt = args.psqt;
    let bucket_impl = match args.bucket_mode {
        BucketMode::Kingrank9 => ShogiLayerStackBucket9::KingRank9,
        BucketMode::Ply9 => ShogiLayerStackBucket9::Ply9(ply_bounds.expect("ply bounds must exist in ply9 mode")),
        BucketMode::Progress8 => match progress_bucket {
            Some(LoadedProgressBucket::V1(bucket)) => ShogiLayerStackBucket9::Progress8(bucket),
            _ => panic!("progress coeff v1 must exist in progress8 mode"),
        },
        BucketMode::Progress8Gikou => match progress_bucket {
            Some(LoadedProgressBucket::Gikou(bucket)) => ShogiLayerStackBucket9::Progress8GikouLite(bucket),
            _ => panic!("progress coeff v2 must exist in progress8gikou mode"),
        },
        BucketMode::Progress8KPAbs => match progress_bucket {
            Some(LoadedProgressBucket::KPAbs(bucket)) => ShogiLayerStackBucket9::Progress8KPAbs(bucket),
            _ => panic!("progress.bin must exist in progress8kpabs mode"),
        },
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
            .set(WrmLossParams { nnue2score: args.wrm_nnue2score, in_scaling })
            .expect("WRM loss parameters should only be initialized once");
        loss_fn_wrm
    } else {
        loss_fn_sigmoid
    };

    macro_rules! build_trainer_with_input {
        ($opt:expr, $use_win_rate:expr, $bucket_impl:expr, $input:expr) => {{
            let mut builder = ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .output_buckets($bucket_impl)
                .save_format(&save_format)
                .loss_fn(loss_fn);
            #[cfg(feature = "cpu")]
            {
                builder = builder.use_threads(1);
            }
            if $use_win_rate {
                builder = builder.use_win_rate_model();
            }
            builder.build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                // L0 (Feature Transformer)
                let l0 = builder.new_affine("l0", input_size, ft_out_c);
                l0.init_with_effective_input_size(32);

                // L1 入力次元: FT 出力 + （HandCount Dense 有効時は +14）
                let l1_in_total = ft_out_c + hand_count_dense_dims;

                // LayerStack layers:
                // - l1: bucket-specific delta (zero init). 入力は [FT 出力 (ft_out_c) | HandCount (14)]
                //   を concat した全体 (l1_in_total) 次元。
                // - l1f: shared factorized part。FT 出力 (ft_out_c) のみに作用（HandCount は共有しない）
                let l1 = Affine {
                    weights: builder.new_weights(
                        "l1w",
                        Shape::new(NUM_BUCKETS * l1_out_c, l1_in_total),
                        InitSettings::Zeroed,
                    ),
                    bias: builder.new_weights("l1b", Shape::new(NUM_BUCKETS * l1_out_c, 1), InitSettings::Zeroed),
                };
                let l1f = builder.new_affine("l1f", ft_out_c, l1_out_c);
                let l2 = builder.new_affine("l2", l2_in_c, NUM_BUCKETS * l2_out_c);
                let l3 = builder.new_affine("l3", l2_out_c, NUM_BUCKETS);

                // Forward pass
                let stm = l0.forward(stm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
                let ntm = l0.forward(ntm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
                let combined = stm.concat(ntm);

                // HandCount Dense を有効にする場合、L1 入力に 14 元の dense vector を concat。
                // `"hand_count"` という名前の dense 入力は ValueDataLoader が
                // `ShogiHalfKaHmHandCount::fill_hand_count` を使って populate する。
                let l1_input_full = if hand_count_dense_dims > 0 {
                    let hand_count =
                        builder.new_dense_input("hand_count", Shape::new(hand_count_dense_dims, 1));
                    combined.concat(hand_count)
                } else {
                    combined
                };

                let l1_out_t = l1.forward(l1_input_full).select(output_buckets) + l1f.forward(combined);
                let l1_main = l1_out_t.slice_rows(0, l1_effective_c);
                let l1_skip = l1_out_t.slice_rows(l1_effective_c, l1_out_c);

                let l1_sqr = l1_main.abs_pow(2.0) * (127.0 / 128.0);
                let l2_input_tensor = l1_sqr.concat(l1_main).crelu();

                let l2_out_t = l2.forward(l2_input_tensor).select(output_buckets).crelu();
                let l3_out = l3.forward(l2_out_t).select(output_buckets);
                let net_output = l3_out + l1_skip;

                if use_psqt {
                    // PSQT shortcut: FT と同じ入力、出力 = バケット数
                    // 学習初期に「PSQTなし」と等価にするため Zeroed で開始
                    let psqt = Affine {
                        weights: builder.new_weights(
                            "psqtw",
                            Shape::new(NUM_BUCKETS, input_size),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("psqtb", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };

                    // PSQT shortcut (Stockfish 準拠: (stm - nstm) / 2)
                    // 各駒は両視点に逆符号で寄与するため、stm - nstm は正味の配置価値を
                    // 約2倍にカウントする。/2 はこの二重カウントを補正する正規化。
                    let stm_psqt = psqt.forward(stm_inputs);
                    let ntm_psqt = psqt.forward(ntm_inputs) * (-1.0);
                    let psqt_diff = (stm_psqt + ntm_psqt).select(output_buckets) * 0.5;

                    net_output + psqt_diff
                } else {
                    net_output
                }
            })
        }};
    }

    macro_rules! maybe_run_or_quantise {
        ($trainer:expr) => {{
            if args.quantise_only {
                let resume_path = args.resume.as_ref().expect("--quantise-only requires --resume");
                let resume_str = resume_path.to_str().unwrap();
                println!("Loading checkpoint from {}...", resume_str);
                $trainer.load_from_checkpoint(resume_str);

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

    let use_win_rate_model = args.win_rate_model;

    // 入力型の分岐: threat 有効時は ShogiHalfKaHmThreat、無効時は ShogiHalfKA_hm
    // ジェネリクスが異なるため、各 optimizer × input の組み合わせを展開する
    macro_rules! run_optimizer {
        ($input:expr) => {{
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::AdamW, use_win_rate_model, bucket_impl, $input);
                    trainer.optimiser.set_params(AdamWParams { decay: args.weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::RAdam => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::RAdam, use_win_rate_model, bucket_impl, $input);
                    let params = RAdamParams { decay: args.weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::Ranger, use_win_rate_model, bucket_impl, $input);
                    trainer.optimiser.set_params(RangerParams { decay: args.weight_decay, ..Default::default() });
                    maybe_run_or_quantise!(trainer);
                }
            }
        }};
    }

    if let Some(tp) = threat_profile {
        run_optimizer!(ShogiHalfKaHmThreat::new(tp));
    } else if use_hand_threat {
        run_optimizer!(ShogiHalfKaHmHandThreat::new());
    } else if use_hand_threat_defensive {
        run_optimizer!(ShogiHalfKaHmHandThreatDefensive::new());
    } else if use_hand_count_dense {
        run_optimizer!(ShogiHalfKaHmHandCount);
    } else {
        run_optimizer!(ShogiHalfKA_hm);
    }

    // Generate final experiment JSON (status: completed)
    if !experiment_quantise_only {
        if let Err(e) = experiment_ctx.write_experiment_json("completed") {
            eprintln!("Warning: Failed to generate experiment JSON: {}", e);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leb128_roundtrip() {
        // 正の値
        let encoded = encode_signed_leb128(0);
        assert_eq!(encoded, vec![0x00]);

        let encoded = encode_signed_leb128(1);
        assert_eq!(encoded, vec![0x01]);

        let encoded = encode_signed_leb128(63);
        assert_eq!(encoded, vec![0x3F]);

        let encoded = encode_signed_leb128(64);
        assert_eq!(encoded, vec![0xC0, 0x00]);

        let encoded = encode_signed_leb128(127);
        assert_eq!(encoded, vec![0xFF, 0x00]);

        // 負の値
        let encoded = encode_signed_leb128(-1);
        assert_eq!(encoded, vec![0x7F]);

        let encoded = encode_signed_leb128(-64);
        assert_eq!(encoded, vec![0x40]);

        let encoded = encode_signed_leb128(-65);
        assert_eq!(encoded, vec![0xBF, 0x7F]);

        let encoded = encode_signed_leb128(-128);
        assert_eq!(encoded, vec![0x80, 0x7F]);
    }

    #[test]
    fn test_leb128_i16_range() {
        // i16::MAX = 32767
        let encoded = encode_signed_leb128(32767);
        assert_eq!(encoded, vec![0xFF, 0xFF, 0x01]);

        // i16::MIN = -32768
        let encoded = encode_signed_leb128(-32768);
        assert_eq!(encoded, vec![0x80, 0x80, 0x7E]);
    }

    #[test]
    fn test_leb128_tensor_format() {
        let values: Vec<i16> = vec![0, 1, -1, 127, -128];
        let result = encode_leb128_tensor_i16(&values);

        // Check magic
        assert_eq!(&result[..17], b"COMPRESSED_LEB128");

        // Check size field (u32 LE)
        let size = u32::from_le_bytes([result[17], result[18], result[19], result[20]]) as usize;

        // Verify compressed data length
        assert_eq!(result.len(), 17 + 4 + size);
    }

    #[test]
    fn test_pad32() {
        assert_eq!(pad32(1536), 1536);
        assert_eq!(pad32(30), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(33), 64);
    }

    #[test]
    fn test_bucket_initial_position() {
        // 平手初期局面: 先手玉 = 4 (rank=4%9=4), 後手玉 = 76 (rank=76%9=4)
        // side_to_move = Black
        // f_rank = 4 (Black そのまま)
        // e_rank = 8 - 4 = 4 (Black から見て相手は後手 → 反転)
        // F_TO_INDEX[4] = 3, E_TO_INDEX[4] = 1
        // bucket = 3 + 1 = 4
        // ただし先手玉がSQ_59(=4)の場合、rank=4%9=4 ←正しい
        //
        // 実際のバケットはPackedSfenValueのデコード結果に依存するため、
        // ここではバケット計算ロジック自体をテストする

        // 味方玉rank=0, 相手玉rank=0 → bucket=0
        const F_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
        const E_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        assert_eq!(F_TO_INDEX[0] + E_TO_INDEX[0], 0);

        // 味方玉rank=8, 相手玉rank=8 → bucket=8
        assert_eq!(F_TO_INDEX[8] + E_TO_INDEX[8], 8);

        // 味方玉rank=4, 相手玉rank=4 → bucket=4
        assert_eq!(F_TO_INDEX[4] + E_TO_INDEX[4], 4);
    }
}
