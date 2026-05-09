/*
KP-absolute progress trainer: learns a YaneuraOu-compatible `progress.bin`
for LayerStack bucket selection (`--bucket-mode progress8kpabs`).

See docs/shogi/shogi_progress_kpabs_train.md for full documentation.

Model:
    z = sum(weights[kp_abs_index])
    p = sigmoid(z)
    bucket = min(7, floor(p * 8))

Two modes:
  - Default (approximate): y = clamp((game_ply - 1) / (ply_max - 1), 0, 1)
    Works with shuffled data.
  - --game-relative (recommended): y = game_ply / total_ply
    Requires game-order-preserved data. Detects game boundaries by game_ply decrease.
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
    /// Uses per-game batching (1 game = 1 gradient step) and file-streaming (no 2-pass).
    #[arg(long)]
    game_relative: bool,

    /// Maximum number of games for training (game-relative mode only, 0=unlimited)
    #[arg(long, default_value_t = 0)]
    max_games: usize,

    /// Number of validation games (game-relative mode only, 0=auto 5% of files)
    #[arg(long, default_value_t = 0)]
    val_games: usize,

    /// Progress report interval in games (game-relative mode only)
    #[arg(long, default_value_t = 1000)]
    log_interval_games: usize,

    /// Save a per-epoch checkpoint as `<output_stem>.e{N}.<ext>` after each epoch.
    /// The final `--output` file is still written after the last epoch.
    #[arg(long)]
    save_each_epoch: bool,
}

fn epoch_checkpoint_path(output: &Path, epoch: usize) -> PathBuf {
    let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("progress");
    let ext = output.extension().and_then(|s| s.to_str()).unwrap_or("bin");
    let new_name = format!("{stem}.e{epoch}.{ext}");
    output.with_file_name(new_name)
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
        Self { m: vec![0.0; size], v: vec![0.0; size], beta1_pow: 1.0, beta2_pow: 1.0 }
    }

    fn step(&mut self, weights: &mut [f32], grad: &[f32], lr: f32) {
        self.beta1_pow *= ADAM_BETA1;
        self.beta2_pow *= ADAM_BETA2;
        let bias_correction1 = 1.0 - self.beta1_pow;
        let bias_correction2 = 1.0 - self.beta2_pow;

        for ((w, m), (v, &g)) in weights.iter_mut().zip(self.m.iter_mut()).zip(self.v.iter_mut().zip(grad.iter())) {
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
        Ok(Self { reader: BufReader::new(file), remaining_records: records })
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
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ext == "bin" || ext == "pack" {
                paths.push(path);
            } else {
                eprintln!("Ignoring unsupported file: {}", path.display());
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
                let ext = entry_path.extension().and_then(|s| s.to_str()).unwrap_or("");
                if ext == "bin" || ext == "pack" {
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
            "no valid *.bin or *.pack files were found from --data",
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

    path.file_stem().and_then(|s| s.to_str()).map_or_else(|| "unknown".to_string(), |s| s.to_string())
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

/// ファイルから対局単位でレコードを返すイテレータ。
/// 対局境界は game_ply が前のレコード以下になったら新対局と判定。
struct GameIterator {
    cursor: PackCursor,
    buffer: Vec<PackedSfenValue>,
    prev_ply: Option<u16>,
    done: bool,
}

impl GameIterator {
    fn new(cursor: PackCursor) -> Self {
        Self { cursor, buffer: Vec::new(), prev_ply: None, done: false }
    }

    /// 次の対局のレコード列を返す。None = ファイル終端。
    fn next_game(&mut self) -> io::Result<Option<Vec<PackedSfenValue>>> {
        if self.done {
            return Ok(None);
        }

        loop {
            match self.cursor.next_psv()? {
                Some(psv) => {
                    let ply = psv.game_ply();
                    let is_boundary = self.prev_ply.is_some_and(|prev| ply <= prev);
                    self.prev_ply = Some(ply);

                    if is_boundary && !self.buffer.is_empty() {
                        // 前の対局を返し、新しい対局をバッファに開始
                        let game = std::mem::take(&mut self.buffer);
                        self.buffer.push(psv);
                        return Ok(Some(game));
                    }

                    self.buffer.push(psv);
                }
                None => {
                    self.done = true;
                    if !self.buffer.is_empty() {
                        return Ok(Some(std::mem::take(&mut self.buffer)));
                    }
                    return Ok(None);
                }
            }
        }
    }
}

/// 全ファイルを順次走査して対局を返すイテレータ。
struct MultiFileGameIterator {
    packs: Vec<PackInfo>,
    file_index: usize,
    current: Option<GameIterator>,
}

impl MultiFileGameIterator {
    fn new(packs: Vec<PackInfo>) -> Self {
        Self { packs, file_index: 0, current: None }
    }

    fn next_game(&mut self) -> io::Result<Option<Vec<PackedSfenValue>>> {
        loop {
            if let Some(ref mut gi) = self.current {
                if let Some(game) = gi.next_game()? {
                    return Ok(Some(game));
                }
            }
            // 次のファイルへ
            if self.file_index >= self.packs.len() {
                return Ok(None);
            }
            let cursor = PackCursor::open(&self.packs[self.file_index].path)?;
            self.current = Some(GameIterator::new(cursor));
            self.file_index += 1;
        }
    }

    fn file_index(&self) -> usize {
        self.file_index
    }

    fn file_count(&self) -> usize {
        self.packs.len()
    }
}

/// game-relative モードの 1 epoch 学習 (対局単位バッチ、ファイル単位ストリーム)
fn train_epoch_game_relative(
    weights: &mut [f32],
    adam: &mut AdamState,
    packs: &[PackInfo],
    lr: f32,
    max_games: usize,
    log_interval: usize,
    epoch: usize,
) -> io::Result<EpochStats> {
    let mut iter = MultiFileGameIterator::new(packs.to_vec());
    let mut grad = vec![0.0f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
    let mut active = Vec::with_capacity(96);
    let mut hist = [0usize; 8];
    let mut loss_sum = 0.0f64;
    let mut samples = 0usize;
    let mut games = 0usize;

    while max_games == 0 || games < max_games {
        let Some(game) = iter.next_game()? else {
            break;
        };

        let game_len = game.len();
        if game_len == 0 {
            continue;
        }

        // 対局単位で勾配を蓄積
        grad.fill(0.0);
        let mut game_loss = 0.0f64;

        for (i, psv) in game.iter().enumerate() {
            // 教師値: linspace(0, 1, game_len)
            let y = if game_len == 1 { 0.0f32 } else { i as f32 / (game_len - 1) as f32 };

            ShogiProgressKPAbs::collect_active_indices(psv, &mut active);

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

            game_loss += f64::from(err * err);
            hist[progress_bucket(p)] += 1;
        }

        // 対局内の局面数で正規化して Adam 更新
        let inv_len = 1.0 / game_len as f32;
        for g in &mut grad {
            *g *= inv_len;
        }
        adam.step(weights, &grad, lr);

        loss_sum += game_loss;
        samples += game_len;
        games += 1;

        if log_interval > 0 && games.is_multiple_of(log_interval) {
            println!(
                "epoch {} file {}/{} games {} samples {} avg_loss {:.6} last_game_loss {:.6}",
                epoch,
                iter.file_index(),
                iter.file_count(),
                games,
                samples,
                loss_sum / samples as f64,
                game_loss / game_len as f64,
            );
        }
    }

    Ok(EpochStats {
        samples,
        batches: games,
        mean_loss: if samples > 0 { loss_sum / samples as f64 } else { 0.0 },
        bucket_hist: hist,
    })
}

/// game-relative モードの検証
fn evaluate_game_relative(weights: &[f32], packs: &[PackInfo], max_games: usize) -> io::Result<EvalStats> {
    let mut iter = MultiFileGameIterator::new(packs.to_vec());
    let mut active = Vec::with_capacity(96);
    let mut hist = [0usize; 8];
    let mut loss_sum = 0.0f64;
    let mut samples = 0usize;
    let mut games = 0usize;

    while max_games == 0 || games < max_games {
        let Some(game) = iter.next_game()? else {
            break;
        };
        let game_len = game.len();
        if game_len == 0 {
            continue;
        }

        for (i, psv) in game.iter().enumerate() {
            let y = if game_len == 1 { 0.0f32 } else { i as f32 / (game_len - 1) as f32 };
            ShogiProgressKPAbs::collect_active_indices(psv, &mut active);
            let mut z = 0.0f32;
            for &idx in &active {
                z += weights[idx];
            }
            let p = sigmoid(z);
            let err = p - y;
            loss_sum += f64::from(err * err);
            hist[progress_bucket(p)] += 1;
        }

        samples += game_len;
        games += 1;
    }

    Ok(EvalStats { samples, mean_loss: if samples > 0 { loss_sum / samples as f64 } else { 0.0 }, bucket_hist: hist })
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
        return Ok(EvalStats { samples: 0, mean_loss: 0.0, bucket_hist: [0; 8] });
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

    Ok(EvalStats { samples, mean_loss: if samples > 0 { loss_sum / samples as f64 } else { 0.0 }, bucket_hist: hist })
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
    let train_targets = game_relative_targets
        .map(|t| if args.val_positions < t.len() { &t[args.val_positions..] } else { &t[t.len()..] });

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

        if args.log_interval > 0 && (batches.is_multiple_of(args.log_interval) || samples == args.max_positions) {
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

    if args.game_relative {
        // === game-relative モード ===
        // ファイル単位ストリーム + 対局単位バッチ (2-pass 不要)

        // train/val ファイル分割
        let split = if args.val_games > 0 || packs.len() < 20 {
            // val_games 指定時 or ファイル少数時: 先頭 5% を val に
            packs.len().max(2) / 20
        } else {
            packs.len() / 20
        }
        .max(1)
        .min(packs.len() - 1);

        let val_packs = packs[..split].to_vec();
        let train_packs = packs[split..].to_vec();
        println!("game-relative mode: {} train files, {} val files", train_packs.len(), val_packs.len());

        // baseline
        let val_max_games = if args.val_games > 0 { args.val_games } else { 5000 };
        let baseline = evaluate_game_relative(&weights, &val_packs, val_max_games)?;
        println!(
            "baseline val_loss {:.6} samples {} top_bucket b{} ({:.2}%)",
            baseline.mean_loss,
            baseline.samples,
            top_bucket_info(&baseline.bucket_hist).0,
            top_bucket_info(&baseline.bucket_hist).1 * 100.0
        );

        for epoch in 1..=args.epochs {
            let train = train_epoch_game_relative(
                &mut weights,
                &mut adam,
                &train_packs,
                args.lr,
                args.max_games,
                args.log_interval_games,
                epoch,
            )?;
            let (train_top_bucket, train_top_share) = top_bucket_info(&train.bucket_hist);
            println!(
                "epoch {} train_loss {:.6} samples {} games {} top_bucket b{} ({:.2}%)",
                epoch,
                train.mean_loss,
                train.samples,
                train.batches,
                train_top_bucket,
                train_top_share * 100.0
            );

            let val = evaluate_game_relative(&weights, &val_packs, val_max_games)?;
            let (val_top_bucket, val_top_share) = top_bucket_info(&val.bucket_hist);
            println!(
                "epoch {} val_loss {:.6} samples {} top_bucket b{} ({:.2}%)",
                epoch,
                val.mean_loss,
                val.samples,
                val_top_bucket,
                val_top_share * 100.0
            );

            if args.save_each_epoch {
                let ckpt = epoch_checkpoint_path(&args.output, epoch);
                write_progress_bin(&ckpt, &weights)?;
                println!("epoch {} checkpoint: {}", epoch, ckpt.display());
            }
        }
    } else {
        // === 近似版モード (従来通り) ===

        if args.val_positions > 0 {
            let baseline = evaluate(&weights, &packs, args.val_positions, args.ply_max, None)?;
            println!(
                "baseline val_loss {:.6} samples {} top_bucket b{} ({:.2}%)",
                baseline.mean_loss,
                baseline.samples,
                top_bucket_info(&baseline.bucket_hist).0,
                top_bucket_info(&baseline.bucket_hist).1 * 100.0
            );
        }

        for epoch in 1..=args.epochs {
            let train = train_epoch(&mut weights, &mut adam, &packs, &args, epoch, None)?;
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
                let val = evaluate(&weights, &packs, args.val_positions, args.ply_max, None)?;
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

            if args.save_each_epoch {
                let ckpt = epoch_checkpoint_path(&args.output, epoch);
                write_progress_bin(&ckpt, &weights)?;
                println!("epoch {} checkpoint: {}", epoch, ckpt.display());
            }
        }
    }

    write_progress_bin(&args.output, &weights)?;
    let bytes = fs::metadata(&args.output)?.len();
    println!("Wrote {} weights to {} ({} bytes)", SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, args.output.display(), bytes);

    Ok(())
}
