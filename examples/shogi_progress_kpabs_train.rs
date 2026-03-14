/*
Approximate KP-absolute progress trainer from shuffled PackedSfenValue packs.

This utility trains a YaneuraOu-compatible `progress.bin` using `game_ply` as a
proxy target:

    y = clamp((game_ply - 1) / (ply_max - 1), 0, 1)

The learned model matches `progress8kpabs` inference:

    z = sum(weights[kp_abs_index])
    p = sigmoid(z)
    bucket = min(7, floor(p * 8))
*/

use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    mem::size_of,
    path::{Path, PathBuf},
};

use bullet_lib::{
    game::outputs::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs},
    shogi::PackedSfenValue,
};
use clap::Parser;

const PACK_RECORD_BYTES: usize = size_of::<PackedSfenValue>();
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

#[derive(Parser, Debug)]
#[command(name = "shogi_progress_kpabs_train")]
#[command(about = "Train an approximate KP-absolute progress.bin from shuffled shogi packs")]
struct Args {
    /// Comma-separated files or directories. Directories contribute only top-level *.bin files.
    #[arg(long)]
    data: String,

    /// Output progress.bin path
    #[arg(long)]
    output: PathBuf,

    /// Number of training positions to consume per epoch
    #[arg(long, visible_alias = "samples", default_value = "50000000")]
    max_positions: usize,

    /// Number of validation positions to consume before the training split
    #[arg(long, default_value = "2000000")]
    val_positions: usize,

    /// Batch size
    #[arg(long, default_value = "4096")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.0002")]
    lr: f32,

    /// Number of passes over the training split
    #[arg(long, default_value = "1")]
    epochs: usize,

    /// Target normalization maximum for y = clamp((ply-1)/(ply_max-1), 0, 1)
    #[arg(long, default_value = "256")]
    ply_max: u16,

    /// Progress report interval in batches
    #[arg(long, default_value = "100")]
    log_interval: usize,

    /// Use game-relative progress target: y = game_ply / total_ply_of_game.
    /// Requires game-order-preserved (non-shuffled) pack data.
    /// Game boundaries are detected by game_ply decreasing.
    #[arg(long)]
    game_relative: bool,
}

#[derive(Debug, Clone)]
struct PackInfo {
    path: PathBuf,
    records: u64,
}

struct PackCursor {
    reader: BufReader<File>,
    remaining_records: u64,
}

struct RoundRobinPackStream {
    cursors: Vec<PackCursor>,
    cursor: usize,
}

#[derive(Debug, Clone, Copy)]
struct EpochStats {
    samples: usize,
    batches: usize,
    mean_loss: f64,
    bucket_hist: [usize; 8],
}

#[derive(Debug, Clone, Copy)]
struct EvalStats {
    samples: usize,
    mean_loss: f64,
    bucket_hist: [usize; 8],
}

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    beta1_pow: f32,
    beta2_pow: f32,
}

impl AdamState {
    fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    fn step(&mut self, weights: &mut [f32], grad: &[f32], lr: f32) {
        self.beta1_pow *= ADAM_BETA1;
        self.beta2_pow *= ADAM_BETA2;
        let bias_correction1 = 1.0 - self.beta1_pow;
        let bias_correction2 = 1.0 - self.beta2_pow;

        for ((w, m), (v, &g)) in weights
            .iter_mut()
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut().zip(grad.iter()))
        {
            *m = ADAM_BETA1 * *m + (1.0 - ADAM_BETA1) * g;
            *v = ADAM_BETA2 * *v + (1.0 - ADAM_BETA2) * g * g;

            let m_hat = *m / bias_correction1.max(f32::MIN_POSITIVE);
            let v_hat = *v / bias_correction2.max(f32::MIN_POSITIVE);
            *w -= lr * m_hat / (v_hat.sqrt() + ADAM_EPS);
        }
    }
}

impl PackCursor {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let records = file.metadata()?.len() / PACK_RECORD_BYTES as u64;
        Ok(Self {
            reader: BufReader::new(file),
            remaining_records: records,
        })
    }

    fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.remaining_records == 0 {
            return Ok(None);
        }

        let mut psv = PackedSfenValue::default();
        match self.reader.read_exact(psv.as_bytes_mut()) {
            Ok(()) => {
                self.remaining_records -= 1;
                Ok(Some(psv))
            }
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
                self.remaining_records = 0;
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }
}

impl RoundRobinPackStream {
    fn open(packs: &[PackInfo]) -> io::Result<Self> {
        let mut cursors = Vec::with_capacity(packs.len());
        for pack in packs {
            cursors.push(PackCursor::open(&pack.path)?);
        }
        Ok(Self { cursors, cursor: 0 })
    }

    fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.cursors.is_empty() {
            return Ok(None);
        }

        let len = self.cursors.len();
        for _ in 0..len {
            let idx = self.cursor % len;
            self.cursor = (self.cursor + 1) % len;
            if let Some(psv) = self.cursors[idx].next_psv()? {
                return Ok(Some(psv));
            }
        }

        Ok(None)
    }

    fn skip(&mut self, count: usize) -> io::Result<usize> {
        let mut skipped = 0usize;
        while skipped < count {
            match self.next_psv()? {
                Some(_) => skipped += 1,
                None => break,
            }
        }
        Ok(skipped)
    }
}

fn collect_pack_infos(spec: &str) -> io::Result<Vec<PackInfo>> {
    let mut paths = Vec::new();

    for raw in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let path = PathBuf::from(raw);
        let meta = fs::metadata(&path).map_err(|err| {
            io::Error::new(err.kind(), format!("failed to read metadata for '{}': {err}", path.display()))
        })?;

        if meta.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                paths.push(path);
            } else {
                eprintln!("Ignoring non-bin file: {}", path.display());
            }
            continue;
        }

        if meta.is_dir() {
            let mut dir_paths = Vec::new();
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if !entry.file_type()?.is_file() {
                    continue;
                }
                if entry_path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    dir_paths.push(entry_path);
                }
            }
            dir_paths.sort();
            paths.extend(dir_paths);
            continue;
        }

        eprintln!("Ignoring unsupported path: {}", path.display());
    }

    paths.sort();
    paths.dedup();

    let mut packs = Vec::with_capacity(paths.len());
    for path in paths {
        let records = fs::metadata(&path)?.len() / PACK_RECORD_BYTES as u64;
        if records == 0 {
            eprintln!("Ignoring empty pack: {}", path.display());
            continue;
        }
        packs.push(PackInfo { path, records });
    }

    if packs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no valid *.bin packs were found from --data",
        ));
    }

    Ok(interleave_pack_groups(packs))
}

fn pack_group_key(path: &Path) -> String {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    if name.starts_with("hao_depth_9_shuffled_") {
        return "hao_depth_9_shuffled".to_string();
    }
    if name.starts_with("shuffled_") {
        return "shuffled".to_string();
    }

    path.file_stem()
        .and_then(|s| s.to_str())
        .map_or_else(|| "unknown".to_string(), |s| s.to_string())
}

fn interleave_pack_groups(packs: Vec<PackInfo>) -> Vec<PackInfo> {
    let total = packs.len();
    let mut groups: BTreeMap<String, VecDeque<PackInfo>> = BTreeMap::new();
    for pack in packs {
        groups.entry(pack_group_key(&pack.path)).or_default().push_back(pack);
    }

    let mut out = Vec::with_capacity(total);
    while out.len() < total {
        let mut progressed = false;
        for queue in groups.values_mut() {
            if let Some(pack) = queue.pop_front() {
                out.push(pack);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    out
}

/// game-relative モード用: 対局順保持データから各レコードの total_ply を事前計算する。
///
/// 対局境界の検出: game_ply が前のレコードの game_ply 以下になったら新しい対局の開始とみなす。
/// 各対局のレコードは先読みして total_ply（最大 game_ply）を取得し、全レコード分のマップを返す。
fn build_game_relative_targets(packs: &[PackInfo]) -> io::Result<Vec<f32>> {
    // 1st pass: 全レコードの game_ply を読み込み
    let mut all_plies: Vec<u16> = Vec::new();
    for pack in packs {
        let mut cursor = PackCursor::open(&pack.path)?;
        while let Some(psv) = cursor.next_psv()? {
            all_plies.push(psv.game_ply());
        }
    }

    if all_plies.is_empty() {
        return Ok(Vec::new());
    }

    // 2nd pass: 対局境界を検出して各レコードの教師値を計算
    // game_ply が前のレコード以下になったら新対局
    let mut targets = Vec::with_capacity(all_plies.len());

    let mut game_start = 0usize;
    for i in 1..=all_plies.len() {
        let is_boundary = i == all_plies.len() || all_plies[i] <= all_plies[i - 1];
        if is_boundary {
            // game_start..i が1つの対局
            let total_ply = all_plies[game_start..i].iter().copied().max().unwrap_or(1).max(1);
            for j in game_start..i {
                let y = all_plies[j] as f32 / total_ply as f32;
                targets.push(y.clamp(0.0, 1.0));
            }
            game_start = i;
        }
    }

    // 対局数と分布の概要を表示
    let num_games = {
        let mut count = 1usize;
        for i in 1..all_plies.len() {
            if all_plies[i] <= all_plies[i - 1] {
                count += 1;
            }
        }
        count
    };
    let avg_ply = all_plies.len() as f64 / num_games as f64;
    println!(
        "game-relative: {} records, {} games detected, avg {:.1} ply/game",
        all_plies.len(),
        num_games,
        avg_ply
    );

    Ok(targets)
}

fn progress_target_from_ply(game_ply: u16, ply_max: u16) -> f32 {
    if ply_max <= 1 {
        return 1.0;
    }
    let numerator = game_ply.saturating_sub(1) as f32;
    let denominator = (ply_max - 1) as f32;
    (numerator / denominator).clamp(0.0, 1.0)
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn progress_bucket(progress: f32) -> usize {
    ((progress * 8.0).floor() as i32).clamp(0, 7) as usize
}

fn top_bucket_info(hist: &[usize; 8]) -> (usize, f64) {
    let total: usize = hist.iter().sum();
    if total == 0 {
        return (0, 0.0);
    }

    let mut best_idx = 0usize;
    let mut best_count = 0usize;
    for (idx, &count) in hist.iter().enumerate() {
        if count > best_count {
            best_idx = idx;
            best_count = count;
        }
    }

    (best_idx, best_count as f64 / total as f64)
}

fn evaluate(
    weights: &[f32],
    packs: &[PackInfo],
    val_positions: usize,
    ply_max: u16,
    game_relative_targets: Option<&[f32]>,
) -> io::Result<EvalStats> {
    if val_positions == 0 {
        return Ok(EvalStats {
            samples: 0,
            mean_loss: 0.0,
            bucket_hist: [0; 8],
        });
    }

    let mut stream = RoundRobinPackStream::open(packs)?;
    let mut active = Vec::with_capacity(96);
    let mut hist = [0usize; 8];
    let mut loss_sum = 0.0f64;
    let mut samples = 0usize;

    while samples < val_positions {
        let Some(psv) = stream.next_psv()? else {
            break;
        };

        let y = if let Some(targets) = game_relative_targets {
            targets.get(samples).copied().unwrap_or(0.5)
        } else {
            progress_target_from_ply(psv.game_ply(), ply_max)
        };
        ShogiProgressKPAbs::collect_active_indices(&psv, &mut active);

        let mut z = 0.0f32;
        for &idx in &active {
            z += weights[idx];
        }
        let p = sigmoid(z);
        let err = p - y;
        loss_sum += f64::from(err * err);
        hist[progress_bucket(p)] += 1;
        samples += 1;
    }

    Ok(EvalStats {
        samples,
        mean_loss: if samples > 0 { loss_sum / samples as f64 } else { 0.0 },
        bucket_hist: hist,
    })
}

fn train_epoch(
    weights: &mut [f32],
    adam: &mut AdamState,
    packs: &[PackInfo],
    args: &Args,
    epoch: usize,
    game_relative_targets: Option<&[f32]>,
) -> io::Result<EpochStats> {
    let mut stream = RoundRobinPackStream::open(packs)?;
    let skipped = stream.skip(args.val_positions)?;
    if skipped < args.val_positions {
        eprintln!(
            "Warning: only skipped {} validation samples before training (requested {})",
            skipped, args.val_positions
        );
    }

    // game-relative の場合、val_positions 分だけオフセットした教師値を使う
    let train_targets = game_relative_targets.map(|t| {
        if args.val_positions < t.len() {
            &t[args.val_positions..]
        } else {
            &t[t.len()..]
        }
    });

    let mut grad = vec![0.0f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
    let mut active = Vec::with_capacity(96);
    let mut hist = [0usize; 8];
    let mut loss_sum = 0.0f64;
    let mut samples = 0usize;
    let mut batches = 0usize;

    while samples < args.max_positions {
        grad.fill(0.0);
        let mut batch_count = 0usize;
        let mut batch_loss = 0.0f64;

        while batch_count < args.batch_size && samples < args.max_positions {
            let Some(psv) = stream.next_psv()? else {
                break;
            };

            let y = if let Some(targets) = train_targets {
                targets.get(samples).copied().unwrap_or(0.5)
            } else {
                progress_target_from_ply(psv.game_ply(), args.ply_max)
            };
            ShogiProgressKPAbs::collect_active_indices(&psv, &mut active);

            let mut z = 0.0f32;
            for &idx in &active {
                z += weights[idx];
            }
            let p = sigmoid(z);
            let err = p - y;
            let grad_scale = 2.0 * err * p * (1.0 - p);

            for &idx in &active {
                grad[idx] += grad_scale;
            }

            batch_loss += f64::from(err * err);
            hist[progress_bucket(p)] += 1;
            batch_count += 1;
            samples += 1;
        }

        if batch_count == 0 {
            break;
        }

        let inv_batch = 1.0 / batch_count as f32;
        for g in &mut grad {
            *g *= inv_batch;
        }

        adam.step(weights, &grad, args.lr);
        batches += 1;
        loss_sum += batch_loss;

        if args.log_interval > 0 && (batches % args.log_interval == 0 || samples == args.max_positions) {
            println!(
                "epoch {} batch {} samples {} train_loss {:.6}",
                epoch,
                batches,
                samples,
                batch_loss / batch_count as f64
            );
        }
    }

    Ok(EpochStats {
        samples,
        batches,
        mean_loss: if samples > 0 { loss_sum / samples as f64 } else { 0.0 },
        bucket_hist: hist,
    })
}

fn write_progress_bin(path: &Path, weights: &[f32]) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let mut out = BufWriter::new(File::create(path)?);
    for &weight in weights {
        out.write_all(&(weight as f64).to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    if args.batch_size == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--batch-size must be >= 1"));
    }
    if args.epochs == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--epochs must be >= 1"));
    }

    let packs = collect_pack_infos(&args.data)?;
    let total_records: u64 = packs.iter().map(|p| p.records).sum();
    println!("Loaded {} pack files", packs.len());
    println!("Total available positions: {}", total_records);
    for pack in &packs {
        println!("  {} ({})", pack.path.display(), pack.records);
    }

    if total_records == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "all pack files were empty"));
    }

    let requested_total = args.val_positions as u64 + args.max_positions as u64;
    if requested_total > total_records {
        println!(
            "Warning: requested val+train positions ({}) exceed available positions ({})",
            requested_total, total_records
        );
    }

    let mut weights = vec![0.0f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
    let mut adam = AdamState::new(SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS);

    // game-relative モード: 事前に教師値を計算
    let game_relative_targets = if args.game_relative {
        Some(build_game_relative_targets(&packs)?)
    } else {
        None
    };
    let gr_ref = game_relative_targets.as_deref();

    if args.val_positions > 0 {
        let baseline = evaluate(&weights, &packs, args.val_positions, args.ply_max, gr_ref)?;
        println!(
            "baseline val_loss {:.6} samples {} top_bucket b{} ({:.2}%)",
            baseline.mean_loss,
            baseline.samples,
            top_bucket_info(&baseline.bucket_hist).0,
            top_bucket_info(&baseline.bucket_hist).1 * 100.0
        );
    }

    for epoch in 1..=args.epochs {
        let train = train_epoch(&mut weights, &mut adam, &packs, &args, epoch, gr_ref)?;
        let (train_top_bucket, train_top_share) = top_bucket_info(&train.bucket_hist);
        println!(
            "epoch {} train_loss {:.6} samples {} batches {} top_bucket b{} ({:.2}%)",
            epoch,
            train.mean_loss,
            train.samples,
            train.batches,
            train_top_bucket,
            train_top_share * 100.0
        );

        if args.val_positions > 0 {
            let val = evaluate(&weights, &packs, args.val_positions, args.ply_max, gr_ref)?;
            let (val_top_bucket, val_top_share) = top_bucket_info(&val.bucket_hist);
            println!(
                "epoch {} val_loss {:.6} samples {} top_bucket b{} ({:.2}%)",
                epoch,
                val.mean_loss,
                val.samples,
                val_top_bucket,
                val_top_share * 100.0
            );
        }
    }

    write_progress_bin(&args.output, &weights)?;
    let bytes = fs::metadata(&args.output)?.len();
    println!(
        "Wrote {} weights to {} ({} bytes)",
        SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
        args.output.display(),
        bytes
    );

    Ok(())
}
