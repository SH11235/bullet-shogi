/*
Shogi LayerStack NNUE Inference Test

学習済みモデルを読み込んで、局面を評価するテストスクリプト。
rshogi と bullet-shogi の推論結果を比較するために使用。

Usage:
    cargo run --release --example shogi_layerstack_eval -- \
        --checkpoint <PATH> \
        --pack <PACK_PATH>

Options:
    --checkpoint <PATH>  チェックポイントディレクトリのパス (例: checkpoints/v1/v1-69)
    --pack <PATH>        PackedSfenValue の pack ファイル
    --samples <N>        評価するサンプル数 (default: 10)
    --offset <N>         pack 内の開始レコード位置 (default: 0)
    --weights <PATH>     weights.bin のパス (省略時: checkpoint/optimiser_state/weights.bin)
    --l0 <SIZE>          L0 サイズ (default: 1536)
    --l1 <SIZE>          L1 サイズ (default: 16)
    --l2 <SIZE>          L2 サイズ (default: 32)
    --scale <N>          学習時の scale (default: 508)
*/

use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

use acyclib::{graph::like::GraphLike, graph::save::GraphWeights, trainer::dataloader::PreparedBatchDevice};
use bullet_lib::{
    game::{
        inputs::{ShogiHalfKA_hm, SparseInputType},
        outputs::{
            OutputBuckets, SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS, SHOGI_PROGRESS8_FEATURE_ORDER,
            SHOGI_PROGRESS8_NUM_FEATURES, ShogiLayerStackBucket9, ShogiProgressBucket8,
        },
    },
    nn::optimiser,
    value::ValueTrainerBuilder,
};
use clap::{Parser, ValueEnum};
use serde::Deserialize;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "shogi_layerstack_eval")]
#[command(about = "Shogi LayerStack NNUE inference test")]
struct Args {
    /// Checkpoint directory path
    #[arg(long)]
    checkpoint: PathBuf,

    /// PackedSfenValue pack file
    #[arg(long)]
    pack: PathBuf,

    /// L0 (Feature Transformer) size
    #[arg(long, default_value = "1536")]
    l0: usize,

    /// L1 size
    #[arg(long, default_value = "16")]
    l1: usize,

    /// L2 size
    #[arg(long, default_value = "32")]
    l2: usize,

    /// Training scale (for centipawn conversion)
    #[arg(long, default_value = "508")]
    scale: i32,

    /// Number of samples to evaluate
    #[arg(long, default_value = "10")]
    samples: usize,

    /// Starting record offset in pack file
    #[arg(long, default_value = "0")]
    offset: u64,

    /// Optional weights.bin path
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Print feature weight sums for debug
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Dump intermediate values (CPU forward pass with float weights)
    #[arg(long, default_value_t = false)]
    dump_intermediates: bool,

    /// Output bucket mode (kingrank9 / ply9)
    #[arg(long, value_enum, default_value = "kingrank9")]
    bucket_mode: BucketMode,

    /// Optional boundaries for ply9 buckets (8 comma-separated values)
    #[arg(long)]
    ply_bounds: Option<String>,

    /// Coefficient JSON (coeff_v1) path for progress8 mode
    #[arg(long)]
    progress_coeff: Option<PathBuf>,

    /// Optional output path to dump evaluated positions as SFEN (one per line)
    #[arg(long)]
    dump_sfens: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum BucketMode {
    #[default]
    Kingrank9,
    Ply9,
    Progress8,
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

fn piece_char(pt: bullet_lib::shogi::PieceType) -> Option<char> {
    use bullet_lib::shogi::PieceType;
    match pt {
        PieceType::Pawn => Some('P'),
        PieceType::Lance => Some('L'),
        PieceType::Knight => Some('N'),
        PieceType::Silver => Some('S'),
        PieceType::Gold => Some('G'),
        PieceType::Bishop => Some('B'),
        PieceType::Rook => Some('R'),
        PieceType::King => Some('K'),
        PieceType::ProPawn => Some('P'),
        PieceType::ProLance => Some('L'),
        PieceType::ProKnight => Some('N'),
        PieceType::ProSilver => Some('S'),
        PieceType::Horse => Some('B'),
        PieceType::Dragon => Some('R'),
        PieceType::None => None,
    }
}

fn is_promoted(pt: bullet_lib::shogi::PieceType) -> bool {
    use bullet_lib::shogi::PieceType;
    matches!(
        pt,
        PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver
            | PieceType::Horse
            | PieceType::Dragon
    )
}

fn hand_to_sfen(black_hand: &bullet_lib::shogi::Hand, white_hand: &bullet_lib::shogi::Hand) -> String {
    use bullet_lib::shogi::PieceType;

    let order = [
        (PieceType::Rook, 'R', 'r'),
        (PieceType::Bishop, 'B', 'b'),
        (PieceType::Gold, 'G', 'g'),
        (PieceType::Silver, 'S', 's'),
        (PieceType::Knight, 'N', 'n'),
        (PieceType::Lance, 'L', 'l'),
        (PieceType::Pawn, 'P', 'p'),
    ];

    let mut out = String::new();
    for (pt, bch, wch) in order {
        let bc = black_hand.count(pt) as usize;
        let wc = white_hand.count(pt) as usize;
        if bc > 0 {
            if bc > 1 {
                out.push_str(&bc.to_string());
            }
            out.push(bch);
        }
        if wc > 0 {
            if wc > 1 {
                out.push_str(&wc.to_string());
            }
            out.push(wch);
        }
    }
    if out.is_empty() { "-".to_string() } else { out }
}

fn board_to_sfen(board: &bullet_lib::shogi::ShogiBoard, ply: u16) -> String {
    let mut s = String::new();

    for rank in 0..9 {
        let mut empty = 0usize;
        for file in (0..9).rev() {
            let idx = file * 9 + rank;
            let pc = board.board[idx];
            if pc.piece_type == bullet_lib::shogi::PieceType::None {
                empty += 1;
                continue;
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
                empty = 0;
            }
            if is_promoted(pc.piece_type) {
                s.push('+');
            }
            if let Some(mut ch) = piece_char(pc.piece_type) {
                if pc.color == bullet_lib::shogi::Color::White {
                    ch = ch.to_ascii_lowercase();
                }
                s.push(ch);
            }
        }
        if empty > 0 {
            s.push_str(&empty.to_string());
        }
        if rank != 8 {
            s.push('/');
        }
    }

    let stm = if board.side_to_move == bullet_lib::shogi::Color::Black { 'b' } else { 'w' };
    let hand = hand_to_sfen(&board.black_hand, &board.white_hand);
    let move_no = if ply == 0 { 1 } else { ply };
    format!("{s} {stm} {hand} {move_no}")
}

// =============================================================================
// Main
// =============================================================================

const NUM_BUCKETS: usize = 9;

#[inline]
fn pad32(n: usize) -> usize {
    (n + 31) & !31
}

fn read_leb128_i16_block<R: Read>(reader: &mut R) -> io::Result<Vec<i16>> {
    let mut magic = [0u8; 17];
    reader.read_exact(&mut magic)?;
    if &magic != b"COMPRESSED_LEB128" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid LEB128 block magic in quantised.bin",
        ));
    }

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let payload_len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;

    let mut values = Vec::new();
    let mut i = 0usize;
    while i < payload.len() {
        let mut result = 0i64;
        let mut shift = 0u32;
        let last_byte = loop {
            if i >= payload.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated signed LEB128 payload",
                ));
            }
            let byte = payload[i];
            i += 1;

            result |= i64::from(byte & 0x7f) << shift;
            shift += 7;

            if (byte & 0x80) == 0 {
                break byte;
            }
            if shift >= 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "signed LEB128 value exceeds 64-bit range",
                ));
            }
        };

        // Sign-extend when sign bit is set.
        if shift < 64 && (last_byte & 0x40) != 0 {
            result |= !0i64 << shift;
        }

        if result < i64::from(i16::MIN) || result > i64::from(i16::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "decoded LEB128 value out of i16 range",
            ));
        }

        values.push(result as i16);
    }

    Ok(values)
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

fn resolve_bucket_impl(args: &Args) -> Result<ShogiLayerStackBucket9, String> {
    match args.bucket_mode {
        BucketMode::Kingrank9 => {
            if args.ply_bounds.is_some() {
                Err("--ply-bounds can only be used with --bucket-mode ply9".to_string())
            } else if args.progress_coeff.is_some() {
                Err("--progress-coeff can only be used with --bucket-mode progress8".to_string())
            } else {
                Ok(ShogiLayerStackBucket9::KingRank9)
            }
        }
        BucketMode::Ply9 => {
            if args.progress_coeff.is_some() {
                return Err("--progress-coeff can only be used with --bucket-mode progress8".to_string());
            }
            let bounds = match &args.ply_bounds {
                Some(text) => parse_ply_bounds_csv(text)?,
                None => SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS,
            };
            Ok(ShogiLayerStackBucket9::Ply9(bounds))
        }
        BucketMode::Progress8 => {
            if args.ply_bounds.is_some() {
                return Err("--ply-bounds can only be used with --bucket-mode ply9".to_string());
            }
            let path = args
                .progress_coeff
                .as_ref()
                .ok_or_else(|| "--bucket-mode progress8 requires --progress-coeff".to_string())?;
            let bucket = load_progress_bucket_from_json(path)?;
            Ok(ShogiLayerStackBucket9::Progress8(bucket))
        }
    }
}

fn main() {
    let args = Args::parse();
    let bucket_impl = resolve_bucket_impl(&args).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let l0_size = args.l0;
    let l1_size = args.l1;
    let l1_effective = l1_size - 1;
    let l2_input = l1_effective * 2;
    let l2_size = args.l2;
    let input_size = ShogiHalfKA_hm.num_inputs();
    let l1_input_dim = l0_size;

    println!("=== Shogi LayerStack NNUE Inference Test ===");
    println!("Checkpoint: {}", args.checkpoint.display());
    println!("Pack: {}", args.pack.display());
    println!("Architecture: L0={}, L1={}, L2={}", l0_size, l1_size, l2_size);
    println!("Scale: {}", args.scale);
    match bucket_impl {
        ShogiLayerStackBucket9::KingRank9 => println!("Bucket mode: kingrank9"),
        ShogiLayerStackBucket9::Ply9(bounds) => {
            println!("Bucket mode: ply9");
            println!("Ply bounds: {:?}", bounds);
        }
        ShogiLayerStackBucket9::Progress8(_) => {
            println!("Bucket mode: progress8");
            if let Some(path) = &args.progress_coeff {
                println!("Progress coeff: {}", path.display());
            }
        }
    }
    println!();

    // Build network (same as training)
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::Ranger)
        .use_threads(1)
        .inputs(ShogiHalfKA_hm)
        .output_buckets(bucket_impl)
        .save_format(&[]) // No save format needed for eval
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // L0 (Feature Transformer)
            let l0 = builder.new_affine("l0", input_size, l0_size);
            l0.init_with_effective_input_size(32);

            // LayerStack layers
            let l1 = builder.new_affine("l1", l1_input_dim, NUM_BUCKETS * l1_size);
            let l1f = builder.new_affine("l1f", l1_input_dim, l1_size);
            let l2 = builder.new_affine("l2", l2_input, NUM_BUCKETS * l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_BUCKETS);

            // Forward pass
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul() * (127.0 / 128.0);
            let combined = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(combined).select(output_buckets) + l1f.forward(combined);
            let l1_main = l1_out.slice_rows(0, l1_effective);
            let l1_skip = l1_out.slice_rows(l1_effective, l1_size);

            let l1_sqr = l1_main.abs_pow(2.0) * (127.0 / 128.0);
            let l1_concat = l1_sqr.concat(l1_main);
            let l2_input_tensor = l1_concat.crelu();

            let l2_out = l2.forward(l2_input_tensor).select(output_buckets).crelu();
            let l3_out = l3.forward(l2_out).select(output_buckets);
            l3_out + l1_skip
        });

    // Load weights from checkpoint (optimiser_state/weights.bin)
    let weights_path = args.weights.clone().unwrap_or_else(|| args.checkpoint.join("optimiser_state/weights.bin"));
    println!("Loading weights from: {}", weights_path.display());

    if !weights_path.exists() {
        eprintln!("Error: Weights file not found: {}", weights_path.display());
        std::process::exit(1);
    }

    trainer.optimiser.load_weights_from_file(weights_path.to_str().unwrap()).unwrap();
    println!("Weights loaded successfully!");
    println!();

    if args.debug {
        let weights = GraphWeights::from(trainer.optimiser.graph.primary());
        let l0 = weights.get("l0w");
        let l0b = weights.get("l0b");
        let l1 = weights.get("l1w");
        let l2 = weights.get("l2w");
        let l2b = weights.get("l2b");
        let l3 = weights.get("l3w");
        let l3b = weights.get("l3b");
        let l1f = weights.get("l1fw");
        let output_dim = l0_size;

        const PIECE_INPUTS: usize = 1629;
        const KB: usize = 44;
        const F_HAND_BISHOP: usize = 79;
        const E_HAND_BISHOP: usize = 82;
        const F_HAND_ROOK: usize = 85;
        const E_HAND_ROOK: usize = 88;
        const F_BISHOP: usize = 900;
        const E_BISHOP: usize = 981;
        const F_ROOK: usize = 1224;
        const E_ROOK: usize = 1305;

        let features = [
            ("F_HAND_BISHOP", F_HAND_BISHOP),
            ("E_HAND_BISHOP", E_HAND_BISHOP),
            ("F_HAND_ROOK", F_HAND_ROOK),
            ("E_HAND_ROOK", E_HAND_ROOK),
            ("F_BISHOP+64", F_BISHOP + 64),
            ("E_BISHOP+64", E_BISHOP + 64),
            ("F_ROOK+16", F_ROOK + 16),
            ("E_ROOK+16", E_ROOK + 16),
        ];

        println!("=== Feature weight sums (l0w) ===");
        for &(name, bp) in &features {
            let feature_idx = KB * PIECE_INPUTS + bp;
            let mut sum = 0.0f32;
            for out in 0..output_dim {
                // l0w is column-major [rows=output_dim, cols=input_dim]:
                // index = col * rows + row
                let idx = feature_idx * output_dim + out;
                sum += l0.values[idx];
            }
            println!("{name}: sum={sum:.4}");
        }
        println!();

        println!("l1w shape: rows={} cols={}", l1.shape.rows(), l1.shape.cols());
        println!("l1fw shape: rows={} cols={}", l1f.shape.rows(), l1f.shape.cols());
        println!();

        // quantised.bin と weights.bin の一致確認（L1の一部）
        let quantised_path = args.checkpoint.join("quantised.bin");
        if quantised_path.exists() {
            if let Ok(mut f) = std::fs::File::open(&quantised_path) {
                let mut buf4 = [0u8; 4];
                // Header
                let _ = f.read_exact(&mut buf4);
                let _ = f.read_exact(&mut buf4);
                let _ = f.read_exact(&mut buf4);
                let arch_len = u32::from_le_bytes(buf4) as usize;
                let mut arch = vec![0u8; arch_len];
                let _ = f.read_exact(&mut arch);

                // ft_hash
                let _ = f.read_exact(&mut buf4);

                // FT biases / weights はそれぞれ LEB128 ブロック
                let ft_biases_q = match read_leb128_i16_block(&mut f) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Warning: failed to parse FT bias block: {e}");
                        return;
                    }
                };
                let ft_weights_q = match read_leb128_i16_block(&mut f) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Warning: failed to parse FT weight block: {e}");
                        return;
                    }
                };

                if ft_biases_q.len() != l0_size {
                    eprintln!(
                        "Warning: FT bias length mismatch: got {}, expected {}",
                        ft_biases_q.len(),
                        l0_size
                    );
                    return;
                }
                if ft_weights_q.len() != input_size * l0_size {
                    eprintln!(
                        "Warning: FT weight length mismatch: got {}, expected {}",
                        ft_weights_q.len(),
                        input_size * l0_size
                    );
                    return;
                }

                println!("=== FT bias sample check (quantised.bin vs weights.bin) ===");
                for idx in 0..4 {
                    let q_expected = (l0b.values[idx] * 127.0f32).round() as i16;
                    let q_file = ft_biases_q[idx];
                    println!("ft_bias[{idx}]: float={:.6} q_expected={q_expected} q_file={q_file}", l0b.values[idx]);
                }
                println!();

                println!("=== FT weight sample check (quantised.bin vs weights.bin) ===");
                for &(name, bp) in &features[..4] {
                    let feature_idx = KB * PIECE_INPUTS + bp;
                    for out_idx in 0..2 {
                        let expected =
                            (l0.values[feature_idx * output_dim + out_idx] * 127.0f32).round() as i16;
                        let q_file = ft_weights_q[feature_idx * l0_size + out_idx];
                        println!(
                            "{name} out={out_idx}: float={:.6} q_expected={expected} q_file={q_file}",
                            l0.values[feature_idx * output_dim + out_idx]
                        );
                    }
                }
                println!();

                // FT ブロック消費後、LayerStack 本体を読む。

                // LayerStack は bucket ごとに保存される:
                // [fc_hash][l1b][l1w][l2b][l2w][l3b][l3w]
                let l1_bias_count = NUM_BUCKETS * l1_size;
                let mut l1_biases = vec![0i32; l1_bias_count];
                let mut l1_weights = vec![0i8; l1_bias_count * l1_input_dim];
                let l2_bias_count = NUM_BUCKETS * l2_size;
                let mut l2_biases = vec![0i32; l2_bias_count];
                let mut l2_weights = vec![0i8; l2_bias_count * l2_input];
                let mut l3_biases = vec![0i32; NUM_BUCKETS];
                let mut l3_weights = vec![0i8; NUM_BUCKETS * l2_size];

                let l1_padded_in = pad32(l1_input_dim);
                let l2_padded_in = pad32(l2_input);
                let out_padded_in = pad32(l2_size);

                for bucket in 0..NUM_BUCKETS {
                    // fc_hash
                    let _ = f.read_exact(&mut buf4);

                    // l1 biases
                    for out_idx in 0..l1_size {
                        let _ = f.read_exact(&mut buf4);
                        l1_biases[bucket * l1_size + out_idx] = i32::from_le_bytes(buf4);
                    }

                    // l1 weights (row-major with padded input)
                    let mut row = vec![0u8; l1_padded_in];
                    for out_idx in 0..l1_size {
                        let _ = f.read_exact(&mut row);
                        let global_out = bucket * l1_size + out_idx;
                        for in_idx in 0..l1_input_dim {
                            l1_weights[global_out * l1_input_dim + in_idx] = row[in_idx] as i8;
                        }
                    }

                    // l2 biases
                    for out_idx in 0..l2_size {
                        let _ = f.read_exact(&mut buf4);
                        l2_biases[bucket * l2_size + out_idx] = i32::from_le_bytes(buf4);
                    }

                    // l2 weights (row-major with padded input)
                    let mut l2_row = vec![0u8; l2_padded_in];
                    for out_idx in 0..l2_size {
                        let _ = f.read_exact(&mut l2_row);
                        let global_out = bucket * l2_size + out_idx;
                        for in_idx in 0..l2_input {
                            l2_weights[global_out * l2_input + in_idx] = l2_row[in_idx] as i8;
                        }
                    }

                    // l3 bias
                    let _ = f.read_exact(&mut buf4);
                    l3_biases[bucket] = i32::from_le_bytes(buf4);

                    // l3 weights (padded)
                    let mut out_row = vec![0u8; out_padded_in];
                    let _ = f.read_exact(&mut out_row);
                    for in_idx in 0..l2_size {
                        l3_weights[bucket * l2_size + in_idx] = out_row[in_idx] as i8;
                    }
                }

                println!("=== L1 bias sample check (quantised.bin vs weights.bin) ===");
                let l1b = weights.get("l1b");
                let l1fb = weights.get("l1fb");
                let bias_scale = 127.0f32 * 64.0f32; // QA * QB
                for idx in 0..4 {
                    let merged_b = l1b.values[idx] + l1fb.values[idx % l1_size];
                    let q_expected = (merged_b * bias_scale).round() as i32;
                    let q_file = l1_biases[idx];
                    println!("bias[{idx}]: merged_float={merged_b:.6} q_expected={q_expected} q_file={q_file}");
                }
                println!();

                println!("=== L1 weight sample check (quantised.bin vs weights.bin) ===");
                let qb = 64.0f32;
                let bucket = 8usize;
                let out_base = bucket * l1_size;
                for out_in_bucket in 0..2 {
                    let out_idx = out_base + out_in_bucket;
                    for in_idx in 0..4 {
                        let bucket_w = l1.values[in_idx * (NUM_BUCKETS * l1_size) + out_idx];
                        let shared_w = l1f.values[in_idx * l1_size + out_in_bucket];
                        let float_w = bucket_w + shared_w;
                        let q_expected = (float_w * qb).round() as i8;
                        let q_file = l1_weights[out_idx * l1_input_dim + in_idx];
                        println!(
                            "bucket={bucket} out={out_in_bucket} in={in_idx}: merged_float={float_w:.6} q_expected={q_expected} q_file={q_file}"
                        );
                    }
                }
                println!();

                println!("=== L2 bias sample check (quantised.bin vs weights.bin) ===");
                let bias_scale = 127.0f32 * 64.0f32;
                for idx in 0..4 {
                    let b_expected = (l2b.values[idx] * bias_scale).round() as i32;
                    let b_file = l2_biases[idx];
                    println!("l2_bias[{idx}]: float={:.6} q_expected={b_expected} q_file={b_file}", l2b.values[idx]);
                }
                println!();

                println!("=== L2 weight sample check (quantised.bin vs weights.bin) ===");
                let qb = 64.0f32;
                let bucket = 8usize;
                let out_base = bucket * l2_size;
                for out_in_bucket in 0..2 {
                    let out_idx = out_base + out_in_bucket;
                    for in_idx in 0..4 {
                        let w = l2.values[in_idx * (NUM_BUCKETS * l2_size) + out_idx];
                        let q_expected = (w * qb).round() as i8;
                        let q_file = l2_weights[out_idx * l2_input + in_idx];
                        println!(
                            "bucket={bucket} out={out_in_bucket} in={in_idx}: float={w:.6} q_expected={q_expected} q_file={q_file}"
                        );
                    }
                }
                println!();

                println!("=== L3 bias/weight sample check (quantised.bin vs weights.bin) ===");
                for bucket in 0..2 {
                    let b_expected = (l3b.values[bucket] * bias_scale).round() as i32;
                    let b_file = l3_biases[bucket];
                    println!(
                        "l3_bias[bucket={bucket}]: float={:.6} q_expected={b_expected} q_file={b_file}",
                        l3b.values[bucket]
                    );
                    for in_idx in 0..4 {
                        let w = l3.values[in_idx * NUM_BUCKETS + bucket];
                        let q_expected = (w * qb).round() as i8;
                        let q_file = l3_weights[bucket * l2_size + in_idx];
                        println!(
                            "l3_w[bucket={bucket} in={in_idx}]: float={w:.6} q_expected={q_expected} q_file={q_file}"
                        );
                    }
                }
                println!();
            }
        }
    }

    // Load samples from pack file
    let mut file = File::open(&args.pack).unwrap_or_else(|e| {
        eprintln!("Error: Failed to open pack file: {e}");
        std::process::exit(1);
    });

    let record_size = std::mem::size_of::<bullet_lib::shogi::PackedSfenValue>() as u64;
    let start = args.offset * record_size;
    if let Err(e) = file.seek(SeekFrom::Start(start)) {
        eprintln!("Error: Failed to seek pack file: {e}");
        std::process::exit(1);
    }

    println!("=== Evaluation Results ===");
    println!("{:>5} {:>6} {:>8} {:>12} {:>12} {:>10}", "Index", "Bucket", "Score", "Raw", "Centipawn", "Diff");
    println!("{}", "-".repeat(50));

    let mut sfen_writer = if let Some(path) = &args.dump_sfens {
        let file = File::create(path).unwrap_or_else(|e| {
            eprintln!("Error: Failed to create --dump-sfens file '{}': {e}", path.display());
            std::process::exit(1);
        });
        Some(io::BufWriter::new(file))
    } else {
        None
    };

    for idx in 0..args.samples {
        let mut buf = [0u8; 40];
        if file.read_exact(&mut buf).is_err() {
            break;
        }

        let mut psv = bullet_lib::shogi::PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&buf);

        let host_data = trainer.state.prepare(std::slice::from_ref(&psv), 1, 1.0, 1.0);

        #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
        let graph = &mut trainer.optimiser.graph;

        #[cfg(any(feature = "multigpu", feature = "cpu"))]
        let graph = trainer.optimiser.graph.primary_mut();

        let mut device_data = PreparedBatchDevice::new(graph.devices(), &host_data).unwrap();
        device_data.load_into_graph(graph).unwrap();
        graph.synchronise().unwrap();
        graph.forward().unwrap();

        let vals = trainer.get_output_values();
        let raw_output = match vals.as_slice() {
            [score] => *score,
            [loss, draw, win] => {
                let max = (*win).max(*draw).max(*loss);
                let win = (win - max).exp();
                let draw = (draw - max).exp();
                let loss = (loss - max).exp();
                (win + draw / 2.0) / (win + draw + loss)
            }
            _ => {
                eprintln!("Unexpected output size: {}", vals.len());
                break;
            }
        };

        let cp = (args.scale as f32) * raw_output;
        let target = psv.score() as f32;
        let diff = cp - target;
        let bucket = bucket_impl.bucket(&psv);
        let decoded = psv.decode();
        let sfen_line = board_to_sfen(&decoded, psv.game_ply());

        if let Some(writer) = sfen_writer.as_mut() {
            if let Err(e) = writeln!(writer, "{sfen_line}") {
                eprintln!("Error: Failed to write SFEN: {e}");
                std::process::exit(1);
            }
        }

        println!(
            "{:>5} {:>6} {:>8} {:>12.4} {:>12.1} {:>10.1}",
            args.offset + idx as u64,
            bucket,
            psv.score(),
            raw_output,
            cp,
            diff
        );
    }

    println!();
    println!("Note: Raw output is the network output before sigmoid (or winrate if WDL).");
    println!("      Centipawn = scale * raw_output");
    if let Some(path) = &args.dump_sfens {
        println!("Dumped SFENs: {}", path.display());
    }
    println!();
    println!("Compare these values with rshogi evaluation on the same records!");

    // Dump intermediates if requested
    if args.dump_intermediates {
        println!();
        println!("=== Float Intermediate Values Dump ===");
        let weights = GraphWeights::from(trainer.optimiser.graph.primary());
        dump_float_intermediates(&weights, l0_size, l1_size, l2_size, input_size, &args.pack, args.offset, bucket_impl);
    }
}

// =============================================================================
// Float Intermediate Dump
// =============================================================================

/// Float 中間値を保持する構造体（デバッグ用）
#[derive(Debug)]
struct FloatIntermediates {
    /// FT出力 (STM) [f32; 1536]
    pub ft_stm: Vec<f32>,
    /// FT出力 (NSTM) [f32; 1536]
    pub ft_nstm: Vec<f32>,
    /// Product Pooling 後 [f32; 1536]
    pub pp_out: Vec<f32>,
    /// L1 出力 (main部分) [f32; 15]
    pub l1_main: Vec<f32>,
    /// L1 出力 (bypass) - f32
    pub l1_bypass: f32,
    /// Dual Activation 後 [f32; 30]
    pub dual_act: Vec<f32>,
    /// L2 出力 [f32; 64]
    pub l2_out: Vec<f32>,
    /// 最終出力 (bypass 加算前)
    pub out_before_bypass: f32,
    /// 最終出力 (bypass 加算後)
    pub final_output: f32,
    /// 使用したバケット
    pub bucket: usize,
}

impl FloatIntermediates {
    fn print_summary(&self) {
        println!("=== LayerStack Intermediates (bullet-shogi float) ===");
        println!("Bucket: {}", self.bucket);
        println!();

        // FT出力の統計
        let ft_stm_sum: f32 = self.ft_stm.iter().sum();
        let ft_nstm_sum: f32 = self.ft_nstm.iter().sum();
        println!("FT STM [1536]: sum={:.2}", ft_stm_sum);
        println!("  first 8: {:?}", &self.ft_stm[..8].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
        println!("FT NSTM [1536]: sum={:.2}", ft_nstm_sum);

        // PP出力の統計
        let pp_sum: f32 = self.pp_out.iter().sum();
        let pp_max = self.pp_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pp_min = self.pp_out.iter().cloned().fold(f32::INFINITY, f32::min);
        println!();
        println!("PP out [1536]: sum={:.2}, min={:.4}, max={:.4}", pp_sum, pp_min, pp_max);
        println!("  first 8: {:?}", &self.pp_out[..8].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

        // L1出力
        println!();
        println!("L1 main [15]: {:?}", &self.l1_main.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
        println!("L1 bypass: {:.4}", self.l1_bypass);

        // Dual Activation
        println!();
        println!("Dual Act [30]:");
        println!(
            "  SqrCReLU [0..15]: {:?}",
            &self.dual_act[..15].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>()
        );
        println!(
            "  CReLU [15..30]: {:?}",
            &self.dual_act[15..30].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>()
        );

        // L2出力
        println!();
        let l2_sum: f32 = self.l2_out.iter().sum();
        println!("L2 out [64]: sum={:.2}", l2_sum);
        println!("  first 8: {:?}", &self.l2_out[..8].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

        // Output
        println!();
        println!("Out (before bypass): {:.4}", self.out_before_bypass);
        println!("Final output: {:.4}", self.final_output);
        println!("=============================================");
    }
}

/// Float forward pass with intermediate dump
#[allow(clippy::needless_range_loop, clippy::manual_memcpy)]
fn dump_float_intermediates(
    weights: &GraphWeights,
    l0_size: usize,
    l1_size: usize,
    l2_size: usize,
    _input_size: usize,
    pack_path: &PathBuf,
    offset: u64,
    bucket_impl: ShogiLayerStackBucket9,
) {
    // Load weights
    let l0w = weights.get("l0w");
    let l0b = weights.get("l0b");
    let l1w = weights.get("l1w");
    let l1b = weights.get("l1b");
    let l1fw = weights.get("l1fw");
    let l1fb = weights.get("l1fb");
    let l2w = weights.get("l2w");
    let l2b = weights.get("l2b");
    let l3w = weights.get("l3w");
    let l3b = weights.get("l3b");

    // Read one sample from pack file
    let mut file = File::open(pack_path).expect("Failed to open pack file");
    let record_size = std::mem::size_of::<bullet_lib::shogi::PackedSfenValue>() as u64;
    file.seek(SeekFrom::Start(offset * record_size)).expect("Failed to seek");
    let mut buf = [0u8; 40];
    file.read_exact(&mut buf).expect("Failed to read");
    let mut psv = bullet_lib::shogi::PackedSfenValue::default();
    psv.as_bytes_mut().copy_from_slice(&buf);

    println!("Evaluating sample at offset {}", offset);
    println!("Target score: {}", psv.score());
    println!();

    // Get active features for both perspectives
    let (stm_features, nstm_features) = get_active_features(&psv);
    println!("STM features: {} active", stm_features.len());
    println!("NSTM features: {} active", nstm_features.len());

    // Feature Transformer (STM)
    let mut ft_stm = vec![0.0f32; l0_size];
    for i in 0..l0_size {
        ft_stm[i] = l0b.values[i];
    }
    for &feat_idx in &stm_features {
        for i in 0..l0_size {
            // l0w: column-major [rows=l0_size, cols=input_size]
            let w_idx = feat_idx * l0_size + i;
            ft_stm[i] += l0w.values[w_idx];
        }
    }

    // Feature Transformer (NSTM)
    let mut ft_nstm = vec![0.0f32; l0_size];
    for i in 0..l0_size {
        ft_nstm[i] = l0b.values[i];
    }
    for &feat_idx in &nstm_features {
        for i in 0..l0_size {
            let w_idx = feat_idx * l0_size + i;
            ft_nstm[i] += l0w.values[w_idx];
        }
    }

    // ClippedReLU + pairwise_mul (Product Pooling の前半)
    // STM: [0..768] * [768..1536]
    // NSTM: [0..768] * [768..1536]
    let mut ft_stm_crelu = vec![0.0f32; l0_size];
    let mut ft_nstm_crelu = vec![0.0f32; l0_size];
    for i in 0..l0_size {
        ft_stm_crelu[i] = ft_stm[i].clamp(0.0, 1.0);
        ft_nstm_crelu[i] = ft_nstm[i].clamp(0.0, 1.0);
    }

    // Product Pooling: pairwise_mul * (127/128)
    let mut pp_out = vec![0.0f32; l0_size];
    for i in 0..(l0_size / 2) {
        pp_out[i] = ft_stm_crelu[i] * ft_stm_crelu[i + l0_size / 2] * (127.0 / 128.0);
        pp_out[i + l0_size / 2] = ft_nstm_crelu[i] * ft_nstm_crelu[i + l0_size / 2] * (127.0 / 128.0);
    }

    // Bucket calculation (must match selected output bucket mode)
    let bucket = bucket_impl.bucket(&psv) as usize;

    // L1: [l1_size, l0_size] but stored as [NUM_BUCKETS * l1_size, l0_size]
    let l1_effective = l1_size - 1;
    let mut l1_out = vec![0.0f32; l1_size];
    for i in 0..l1_size {
        let out_idx = bucket * l1_size + i;
        l1_out[i] = l1b.values[out_idx] + l1fb.values[i];
        for j in 0..l0_size {
            // column-major:
            // l1w:  shape [NUM_BUCKETS*l1_size, l0_size], idx = in * rows + out
            // l1fw: shape [l1_size, l0_size],            idx = in * rows + out
            let w_idx = j * (NUM_BUCKETS * l1_size) + out_idx;
            let wf_idx = j * l1_size + i;
            l1_out[i] += pp_out[j] * (l1w.values[w_idx] + l1fw.values[wf_idx]);
        }
    }

    let l1_main: Vec<f32> = l1_out[..l1_effective].to_vec();
    let l1_bypass = l1_out[l1_effective];

    // Dual Activation: SqrCReLU + CReLU
    // SqrCReLU: x² * (127/128), clamp [0, 1]
    // CReLU: clamp [0, 1]
    let mut dual_act = vec![0.0f32; l1_effective * 2];
    for i in 0..l1_effective {
        let sqr = l1_main[i] * l1_main[i] * (127.0 / 128.0);
        dual_act[i] = sqr.clamp(0.0, 1.0);
        dual_act[i + l1_effective] = l1_main[i].clamp(0.0, 1.0);
    }

    // L2: [NUM_BUCKETS * l2_size, l1_effective * 2]
    let mut l2_out = vec![0.0f32; l2_size];
    for i in 0..l2_size {
        let out_idx = bucket * l2_size + i;
        l2_out[i] = l2b.values[out_idx];
        for j in 0..(l1_effective * 2) {
            let w_idx = j * (NUM_BUCKETS * l2_size) + out_idx;
            l2_out[i] += dual_act[j] * l2w.values[w_idx];
        }
    }
    // CReLU
    for i in 0..l2_size {
        l2_out[i] = l2_out[i].clamp(0.0, 1.0);
    }

    // Output: [NUM_BUCKETS, l2_size]
    let mut out_before_bypass = l3b.values[bucket];
    for i in 0..l2_size {
        let w_idx = i * NUM_BUCKETS + bucket;
        out_before_bypass += l2_out[i] * l3w.values[w_idx];
    }

    let final_output = out_before_bypass + l1_bypass;

    let intermediates = FloatIntermediates {
        ft_stm,
        ft_nstm,
        pp_out,
        l1_main,
        l1_bypass,
        dual_act,
        l2_out,
        out_before_bypass,
        final_output,
        bucket,
    };

    intermediates.print_summary();

    // Print expected quantized values for comparison with rshogi
    println!();
    println!("=== Expected Quantized Values (for rshogi comparison) ===");
    println!(
        "PP out (×127): {:?}",
        intermediates.pp_out[..8].iter().map(|x| (*x * 127.0).round() as i32).collect::<Vec<_>>()
    );
    println!(
        "L1 main (×8128): {:?}",
        intermediates.l1_main.iter().map(|x| (*x * 8128.0).round() as i32).collect::<Vec<_>>()
    );
    println!("L1 bypass (×8128): {}", (intermediates.l1_bypass * 8128.0).round() as i32);
    println!(
        "Dual Act (×127): SqrCReLU={:?}",
        intermediates.dual_act[..15].iter().map(|x| (*x * 127.0).round() as i32).collect::<Vec<_>>()
    );
    println!(
        "L2 out (×127): {:?}",
        intermediates.l2_out[..8].iter().map(|x| (*x * 127.0).round() as i32).collect::<Vec<_>>()
    );
    println!("Final (×508 for cp): {}", (intermediates.final_output * 508.0).round() as i32);
}

/// Get active features for a position
fn get_active_features(psv: &bullet_lib::shogi::PackedSfenValue) -> (Vec<usize>, Vec<usize>) {
    let mut stm_features = Vec::new();
    let mut nstm_features = Vec::new();

    ShogiHalfKA_hm.map_features(psv, |stm_idx, nstm_idx| {
        stm_features.push(stm_idx);
        nstm_features.push(nstm_idx);
    });

    debug_assert_eq!(stm_features.len(), nstm_features.len());
    (stm_features, nstm_features)
}
