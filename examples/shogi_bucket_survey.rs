/*
Shogi bucket distribution survey

Compares multiple output-bucket strategies on PackedSfenValue data.

Usage:
    cargo run --release --example shogi_bucket_survey -- \
      --pack data/DLSuisho15b/hao_depth_9_shuffled_01.bin \
      --samples 200000
*/

use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    mem::size_of,
    path::PathBuf,
};

use bullet_lib::{
    game::outputs::{
        OutputBuckets, SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER, SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES,
        SHOGI_PROGRESS8_FEATURE_ORDER, SHOGI_PROGRESS8_NUM_FEATURES, ShogiKingRankBucket, ShogiProgressBucket8,
        ShogiProgressBucket8GikouLite, ShogiProgressKPAbs,
    },
    shogi::{Color, PackedSfenValue, ShogiBoard},
};
use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(name = "shogi_bucket_survey")]
#[command(about = "Survey bucket distributions for candidate shogi output-bucket schemes")]
struct Args {
    /// Comma-separated pack files
    #[arg(long)]
    pack: String,

    /// Number of samples to read in total
    #[arg(long, default_value = "50000")]
    samples: usize,

    /// Starting record offset for each file
    #[arg(long, default_value = "0")]
    offset: u64,

    /// Read every N-th record (1 = dense scan)
    #[arg(long, default_value = "1")]
    stride: u64,

    /// Split --samples evenly across pack files instead of sequential fill
    #[arg(long)]
    balanced: bool,

    /// Print per-pack (per-file) top-bucket summary
    #[arg(long)]
    per_pack: bool,

    /// Optional fixed boundaries for 9 ply buckets, e.g. "30,44,58,72,86,100,116,138"
    #[arg(long)]
    fixed_ply_bounds: Option<String>,

    /// Optional progress coeff JSON (coeff_v1) for progress8 histogram
    #[arg(long)]
    progress_coeff: Option<PathBuf>,

    /// Optional progress coeff JSON (coeff_v2) for progress8gikou histogram
    #[arg(long)]
    progress_coeff_v2: Option<PathBuf>,

    /// Optional progress.bin path for progress8kpabs histogram
    #[arg(long)]
    progress_kpabs: Option<PathBuf>,

    /// Optional CSV path to dump progress-feature training rows
    #[arg(long)]
    dump_progress_csv: Option<PathBuf>,

    /// Optional CSV path to dump progress-feature rows for coeff_v2 (gikou_lite_34)
    #[arg(long)]
    dump_progress_v2_csv: Option<PathBuf>,

    /// ply_max used in y_progress_target = clamp((game_ply-1)/(ply_max-1), 0, 1)
    #[arg(long, default_value = "256")]
    ply_max: u16,
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

#[derive(Clone, Copy)]
struct SampleMeta {
    psv: PackedSfenValue,
    pack_index: usize,
    record_index: u64,
    ply: u16,
    kingrank_bucket: u8,
    friend_zone3: u8,
    board_non_king_count: u8,
}

fn friend_zone3(board: &ShogiBoard) -> u8 {
    let side = board.side_to_move;
    let f_king = board.king_square(side);
    let f_rank = match side {
        Color::Black => f_king.rank() as usize,
        Color::White => 8 - f_king.rank() as usize,
    };
    match f_rank {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    }
}

fn board_non_king_count(board: &ShogiBoard) -> u8 {
    board
        .board
        .iter()
        .filter(|p| {
            p.piece_type != bullet_lib::shogi::PieceType::None && p.piece_type != bullet_lib::shogi::PieceType::King
        })
        .count() as u8
}

fn quantile_boundaries(values: &[u16], bins: usize) -> Vec<u16> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let mut out = Vec::with_capacity(bins.saturating_sub(1));
    for i in 1..bins {
        let idx = (i * n) / bins;
        out.push(sorted[idx.min(n - 1)]);
    }
    out
}

fn bucket_by_boundaries(value: u16, boundaries: &[u16]) -> usize {
    for (i, &b) in boundaries.iter().enumerate() {
        if value <= b {
            return i;
        }
    }
    boundaries.len()
}

fn csv_escape(text: &str) -> String {
    text.replace('"', "\"\"")
}

fn progress_target_from_ply(game_ply: u16, ply_max: u16) -> f32 {
    if ply_max <= 1 {
        return 1.0;
    }
    let num = game_ply.saturating_sub(1) as f32;
    let den = (ply_max - 1) as f32;
    (num / den).clamp(0.0, 1.0)
}

fn dump_progress_feature_csv(
    path: &PathBuf,
    packs: &[PathBuf],
    samples: &[SampleMeta],
    ply_max: u16,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "pack_path,record_index,game_ply,x_board_non_king,x_hand_total,x_major_board,x_promoted_board,x_stm_king_rank_rel,x_ntm_king_rank_rel,y_progress_target,sample_weight"
    )?;

    for s in samples {
        let x = ShogiProgressBucket8::extract_features(&s.psv);
        let y = progress_target_from_ply(s.ply, ply_max);
        let pack_path =
            packs.get(s.pack_index).map(|p| p.display().to_string()).unwrap_or_else(|| "<unknown-pack>".to_string());
        let pack_path_escaped = csv_escape(&pack_path);

        writeln!(
            out,
            "\"{}\",{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},1.0",
            pack_path_escaped, s.record_index, s.ply, x[0], x[1], x[2], x[3], x[4], x[5], y
        )?;
    }
    out.flush()?;
    Ok(())
}

fn dump_progress_feature_v2_csv(
    path: &PathBuf,
    packs: &[PathBuf],
    samples: &[SampleMeta],
    ply_max: u16,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(File::create(path)?);
    write!(out, "pack_path,record_index,game_ply")?;
    for name in SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER {
        write!(out, ",{}", name)?;
    }
    writeln!(out, ",y_progress_target,sample_weight")?;

    for s in samples {
        let x = ShogiProgressBucket8GikouLite::extract_features(&s.psv);
        let y = progress_target_from_ply(s.ply, ply_max);
        let pack_path =
            packs.get(s.pack_index).map(|p| p.display().to_string()).unwrap_or_else(|| "<unknown-pack>".to_string());
        let pack_path_escaped = csv_escape(&pack_path);

        write!(out, "\"{}\",{},{}", pack_path_escaped, s.record_index, s.ply)?;
        for v in x {
            write!(out, ",{:.6}", v)?;
        }
        writeln!(out, ",{:.6},1.0", y)?;
    }
    out.flush()?;
    Ok(())
}

fn print_hist(name: &str, hist: &[usize]) {
    let total: usize = hist.iter().sum();
    println!("\n== {name} ==");
    if total == 0 {
        println!("(no samples)");
        return;
    }

    let mut max_bucket = 0usize;
    let mut max_count = 0usize;
    for (i, &c) in hist.iter().enumerate() {
        if c > max_count {
            max_count = c;
            max_bucket = i;
        }
    }

    for (i, &c) in hist.iter().enumerate() {
        let pct = 100.0 * (c as f64) / (total as f64);
        println!("bucket {:>2}: {:>8} ({:>6.2}%)", i, c, pct);
    }

    let max_share = 100.0 * (max_count as f64) / (total as f64);
    println!("top bucket: {} ({:.2}%)", max_bucket, max_share);
}

fn top_bucket_info(hist: &[usize]) -> (usize, f64) {
    let total: usize = hist.iter().sum();
    if total == 0 {
        return (0, 0.0);
    }
    let mut max_bucket = 0usize;
    let mut max_count = 0usize;
    for (i, &c) in hist.iter().enumerate() {
        if c > max_count {
            max_count = c;
            max_bucket = i;
        }
    }
    let share = 100.0 * (max_count as f64) / (total as f64);
    (max_bucket, share)
}

fn parse_bounds_csv(text: &str) -> Result<Vec<u16>, String> {
    let mut out = Vec::new();
    for token in text.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        match t.parse::<u16>() {
            Ok(v) => out.push(v),
            Err(e) => return Err(format!("invalid boundary '{t}': {e}")),
        }
    }
    Ok(out)
}

fn load_progress_bucket_from_json(path: &PathBuf) -> Result<ShogiProgressBucket8, String> {
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
        .map_err(|e| format!("failed to read --progress-coeff-v2 '{}': {e}", path.display()))?;
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

fn read_samples(
    path: &PathBuf,
    pack_index: usize,
    offset: u64,
    stride: u64,
    max_samples: usize,
) -> io::Result<Vec<SampleMeta>> {
    let mut file = File::open(path)?;
    let record_size = size_of::<PackedSfenValue>() as u64;
    let file_records = file.metadata()?.len() / record_size;
    let mut out = Vec::with_capacity(max_samples);

    if stride == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--stride must be >= 1"));
    }
    if offset >= file_records {
        return Ok(out);
    }

    file.seek(SeekFrom::Start(offset * record_size))?;
    let mut buf = [0u8; 40];
    let mut record_index = offset;

    while out.len() < max_samples {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&buf);

        let board = psv.decode();
        let kingrank_bucket = ShogiKingRankBucket::<9>.bucket(&psv);
        out.push(SampleMeta {
            psv,
            pack_index,
            record_index,
            ply: psv.game_ply(),
            kingrank_bucket,
            friend_zone3: friend_zone3(&board),
            board_non_king_count: board_non_king_count(&board),
        });

        record_index = record_index.saturating_add(1);

        if stride > 1 {
            let skip_bytes = (stride - 1) * record_size;
            if file.seek(SeekFrom::Current(skip_bytes as i64)).is_err() {
                break;
            }
            record_index = record_index.saturating_add(stride - 1);
        }
    }

    Ok(out)
}

fn main() {
    let args = Args::parse();
    let packs: Vec<PathBuf> =
        args.pack.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect();

    if packs.is_empty() {
        eprintln!("No --pack files were provided.");
        std::process::exit(1);
    }

    let mut samples = Vec::with_capacity(args.samples);
    if args.balanced {
        let pack_count = packs.len();
        let per_pack = args.samples / pack_count;
        let extra = args.samples % pack_count;
        for (idx, path) in packs.iter().enumerate() {
            let target = per_pack + usize::from(idx < extra);
            if target == 0 {
                continue;
            }
            match read_samples(path, idx, args.offset, args.stride, target) {
                Ok(mut chunk) => {
                    println!("Loaded {} samples from {}", chunk.len(), path.display());
                    samples.append(&mut chunk);
                }
                Err(err) => {
                    eprintln!("Failed to read {}: {}", path.display(), err);
                }
            }
        }
    } else {
        let mut remaining = args.samples;
        for (idx, path) in packs.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            match read_samples(path, idx, args.offset, args.stride, remaining) {
                Ok(mut chunk) => {
                    remaining = remaining.saturating_sub(chunk.len());
                    println!("Loaded {} samples from {}", chunk.len(), path.display());
                    samples.append(&mut chunk);
                }
                Err(err) => {
                    eprintln!("Failed to read {}: {}", path.display(), err);
                }
            }
        }
    }

    if samples.is_empty() {
        eprintln!("No samples loaded.");
        std::process::exit(1);
    }

    if let Some(path) = &args.dump_progress_csv {
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Failed to create parent dir for --dump-progress-csv '{}': {e}", path.display());
                std::process::exit(1);
            }
        }
        if let Err(e) = dump_progress_feature_csv(path, &packs, &samples, args.ply_max) {
            eprintln!("Failed to write --dump-progress-csv '{}': {e}", path.display());
            std::process::exit(1);
        }
        println!("Dumped progress feature CSV: {}", path.display());
    }
    if let Some(path) = &args.dump_progress_v2_csv {
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Failed to create parent dir for --dump-progress-v2-csv '{}': {e}", path.display());
                std::process::exit(1);
            }
        }
        if let Err(e) = dump_progress_feature_v2_csv(path, &packs, &samples, args.ply_max) {
            eprintln!("Failed to write --dump-progress-v2-csv '{}': {e}", path.display());
            std::process::exit(1);
        }
        println!("Dumped progress v2 feature CSV: {}", path.display());
    }

    println!("\nTotal samples: {}", samples.len());
    let plys: Vec<u16> = samples.iter().map(|s| s.ply).collect();
    let counts: Vec<u16> = samples.iter().map(|s| s.board_non_king_count as u16).collect();
    let q9_bounds = quantile_boundaries(&plys, 9);
    let q3_bounds = quantile_boundaries(&plys, 3);
    let mat_q9_bounds = quantile_boundaries(&counts, 9);
    println!("Ply quantile boundaries (9 buckets): {:?}", q9_bounds);
    println!("Ply quantile boundaries (3 phases): {:?}", q3_bounds);
    println!("Board non-king quantile boundaries (9 buckets): {:?}", mat_q9_bounds);

    let fixed_bounds = if let Some(text) = &args.fixed_ply_bounds {
        match parse_bounds_csv(text) {
            Ok(v) if v.len() == 8 => Some(v),
            Ok(v) => {
                eprintln!("--fixed-ply-bounds must have 8 values for 9 buckets, got {}", v.len());
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Failed to parse --fixed-ply-bounds: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    if let Some(bounds) = &fixed_bounds {
        println!("Fixed ply boundaries (9 buckets): {:?}", bounds);
    }
    let progress_bucket = if let Some(path) = &args.progress_coeff {
        match load_progress_bucket_from_json(path) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Failed to load --progress-coeff: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let progress_bucket_v2 = if let Some(path) = &args.progress_coeff_v2 {
        match load_progress_bucket_v2_from_json(path) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Failed to load --progress-coeff-v2: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let progress_bucket_kpabs = if let Some(path) = &args.progress_kpabs {
        match ShogiProgressKPAbs::load_from_bin(path) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Failed to load --progress-kpabs: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    let mut hist_kingrank = vec![0usize; 9];
    let mut hist_ply_q9 = vec![0usize; 9];
    let mut hist_hybrid = vec![0usize; 9];
    let mut hist_ply_fixed = vec![0usize; 9];
    let mut hist_board_count_q9 = vec![0usize; 9];
    let mut hist_progress8 = vec![0usize; 8];
    let mut hist_progress8_gikou = vec![0usize; 8];
    let mut hist_progress8_kpabs = vec![0usize; 8];
    let mut per_pack_samples = vec![0usize; packs.len()];
    let mut per_pack_kingrank = vec![vec![0usize; 9]; packs.len()];
    let mut per_pack_progress8 = progress_bucket.map(|_| vec![vec![0usize; 8]; packs.len()]);
    let mut per_pack_progress8_gikou = progress_bucket_v2.map(|_| vec![vec![0usize; 8]; packs.len()]);
    let mut per_pack_progress8_kpabs = progress_bucket_kpabs.map(|_| vec![vec![0usize; 8]; packs.len()]);

    for s in &samples {
        let pack_idx = s.pack_index;
        per_pack_samples[pack_idx] += 1;
        per_pack_kingrank[pack_idx][s.kingrank_bucket as usize] += 1;

        hist_kingrank[s.kingrank_bucket as usize] += 1;
        hist_ply_q9[bucket_by_boundaries(s.ply, &q9_bounds)] += 1;
        hist_board_count_q9[bucket_by_boundaries(s.board_non_king_count as u16, &mat_q9_bounds)] += 1;

        let phase = bucket_by_boundaries(s.ply, &q3_bounds);
        let hybrid_bucket = phase * 3 + s.friend_zone3 as usize;
        hist_hybrid[hybrid_bucket] += 1;

        if let Some(bounds) = &fixed_bounds {
            hist_ply_fixed[bucket_by_boundaries(s.ply, bounds)] += 1;
        }
        if let Some(bucket) = progress_bucket {
            let b = bucket.bucket(&s.psv) as usize;
            hist_progress8[b] += 1;
            if let Some(per_pack) = per_pack_progress8.as_mut() {
                per_pack[pack_idx][b] += 1;
            }
        }
        if let Some(bucket) = progress_bucket_v2 {
            let b = bucket.bucket(&s.psv) as usize;
            hist_progress8_gikou[b] += 1;
            if let Some(per_pack) = per_pack_progress8_gikou.as_mut() {
                per_pack[pack_idx][b] += 1;
            }
        }
        if let Some(bucket) = progress_bucket_kpabs {
            let b = bucket.bucket(&s.psv) as usize;
            hist_progress8_kpabs[b] += 1;
            if let Some(per_pack) = per_pack_progress8_kpabs.as_mut() {
                per_pack[pack_idx][b] += 1;
            }
        }
    }

    print_hist("Current: KingRank 3x3", &hist_kingrank);
    print_hist("Candidate A: Ply Quantile 9", &hist_ply_q9);
    print_hist("Candidate C: BoardNonKingCount Quantile 9", &hist_board_count_q9);
    if fixed_bounds.is_some() {
        print_hist("Candidate A2: Ply Fixed-Boundary 9", &hist_ply_fixed);
    }
    if progress_bucket.is_some() {
        print_hist("Candidate D: Progress8 (logistic)", &hist_progress8);
    }
    if progress_bucket_v2.is_some() {
        print_hist("Candidate E: Progress8 Gikou-lite (logistic)", &hist_progress8_gikou);
    }
    if progress_bucket_kpabs.is_some() {
        print_hist("Candidate F: Progress8 KPAbs (logistic)", &hist_progress8_kpabs);
    }
    print_hist("Candidate B: (Ply Quantile 3) x (FriendKingZone 3)", &hist_hybrid);

    if args.per_pack {
        println!("\n== Per-Pack Top-Bucket Summary ==");
        if progress_bucket.is_some() && progress_bucket_v2.is_some() && progress_bucket_kpabs.is_some() {
            println!(
                "{:>3} {:>8} {:>16} {:>16} {:>18} {:>18}  pack",
                "idx", "samples", "kingrank_top", "progress8_top", "progress8gikou_top", "progress8kpabs_top"
            );
        } else if progress_bucket.is_some() && progress_bucket_v2.is_some() {
            println!(
                "{:>3} {:>8} {:>16} {:>16} {:>18}  pack",
                "idx", "samples", "kingrank_top", "progress8_top", "progress8gikou_top"
            );
        } else if progress_bucket.is_some() && progress_bucket_kpabs.is_some() {
            println!(
                "{:>3} {:>8} {:>16} {:>16} {:>18}  pack",
                "idx", "samples", "kingrank_top", "progress8_top", "progress8kpabs_top"
            );
        } else if progress_bucket_v2.is_some() && progress_bucket_kpabs.is_some() {
            println!(
                "{:>3} {:>8} {:>16} {:>18} {:>18}  pack",
                "idx", "samples", "kingrank_top", "progress8gikou_top", "progress8kpabs_top"
            );
        } else if progress_bucket.is_some() {
            println!("{:>3} {:>8} {:>16} {:>16}  pack", "idx", "samples", "kingrank_top", "progress8_top");
        } else if progress_bucket_v2.is_some() {
            println!("{:>3} {:>8} {:>16} {:>18}  pack", "idx", "samples", "kingrank_top", "progress8gikou_top");
        } else if progress_bucket_kpabs.is_some() {
            println!("{:>3} {:>8} {:>16} {:>18}  pack", "idx", "samples", "kingrank_top", "progress8kpabs_top");
        } else {
            println!("{:>3} {:>8} {:>16}  pack", "idx", "samples", "kingrank_top");
        }

        for (idx, path) in packs.iter().enumerate() {
            let samples = per_pack_samples[idx];
            if samples == 0 {
                continue;
            }

            let (kr_bucket, kr_share) = top_bucket_info(&per_pack_kingrank[idx]);
            if let (Some(progress_hist), Some(progress_hist_gikou), Some(progress_hist_kpabs)) =
                (&per_pack_progress8, &per_pack_progress8_gikou, &per_pack_progress8_kpabs)
            {
                let (pr_bucket, pr_share) = top_bucket_info(&progress_hist[idx]);
                let (pg_bucket, pg_share) = top_bucket_info(&progress_hist_gikou[idx]);
                let (pk_bucket, pk_share) = top_bucket_info(&progress_hist_kpabs[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>5} ({:>6.2}%) {:>7} ({:>6.2}%) {:>7} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pr_bucket,
                    pr_share,
                    pg_bucket,
                    pg_share,
                    pk_bucket,
                    pk_share,
                    path.display()
                );
            } else if let (Some(progress_hist), Some(progress_hist_gikou)) =
                (&per_pack_progress8, &per_pack_progress8_gikou)
            {
                let (pr_bucket, pr_share) = top_bucket_info(&progress_hist[idx]);
                let (pg_bucket, pg_share) = top_bucket_info(&progress_hist_gikou[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>5} ({:>6.2}%) {:>7} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pr_bucket,
                    pr_share,
                    pg_bucket,
                    pg_share,
                    path.display()
                );
            } else if let (Some(progress_hist), Some(progress_hist_kpabs)) =
                (&per_pack_progress8, &per_pack_progress8_kpabs)
            {
                let (pr_bucket, pr_share) = top_bucket_info(&progress_hist[idx]);
                let (pk_bucket, pk_share) = top_bucket_info(&progress_hist_kpabs[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>5} ({:>6.2}%) {:>7} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pr_bucket,
                    pr_share,
                    pk_bucket,
                    pk_share,
                    path.display()
                );
            } else if let (Some(progress_hist_gikou), Some(progress_hist_kpabs)) =
                (&per_pack_progress8_gikou, &per_pack_progress8_kpabs)
            {
                let (pg_bucket, pg_share) = top_bucket_info(&progress_hist_gikou[idx]);
                let (pk_bucket, pk_share) = top_bucket_info(&progress_hist_kpabs[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>7} ({:>6.2}%) {:>7} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pg_bucket,
                    pg_share,
                    pk_bucket,
                    pk_share,
                    path.display()
                );
            } else if let Some(progress_hist) = &per_pack_progress8 {
                let (pr_bucket, pr_share) = top_bucket_info(&progress_hist[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>5} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pr_bucket,
                    pr_share,
                    path.display()
                );
            } else if let Some(progress_hist_gikou) = &per_pack_progress8_gikou {
                let (pg_bucket, pg_share) = top_bucket_info(&progress_hist_gikou[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>7} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pg_bucket,
                    pg_share,
                    path.display()
                );
            } else if let Some(progress_hist_kpabs) = &per_pack_progress8_kpabs {
                let (pk_bucket, pk_share) = top_bucket_info(&progress_hist_kpabs[idx]);
                println!(
                    "{:>3} {:>8} {:>5} ({:>6.2}%) {:>7} ({:>6.2}%)  {}",
                    idx,
                    samples,
                    kr_bucket,
                    kr_share,
                    pk_bucket,
                    pk_share,
                    path.display()
                );
            } else {
                println!("{:>3} {:>8} {:>5} ({:>6.2}%)  {}", idx, samples, kr_bucket, kr_share, path.display());
            }
        }
    }
}
