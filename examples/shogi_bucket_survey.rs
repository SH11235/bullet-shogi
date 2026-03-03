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
    io::{self, Read, Seek, SeekFrom},
    mem::size_of,
    path::PathBuf,
};

use bullet_lib::{
    game::outputs::{OutputBuckets, ShogiKingRankBucket},
    shogi::{Color, PackedSfenValue, ShogiBoard},
};
use clap::Parser;

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

    /// Optional fixed boundaries for 9 ply buckets, e.g. "30,44,58,72,86,100,116,138"
    #[arg(long)]
    fixed_ply_bounds: Option<String>,
}

#[derive(Clone, Copy)]
struct SampleMeta {
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

fn read_samples(path: &PathBuf, offset: u64, stride: u64, max_samples: usize) -> io::Result<Vec<SampleMeta>> {
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

    while out.len() < max_samples {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&buf);

        let board = psv.decode();
        let kingrank_bucket = ShogiKingRankBucket::<9>.bucket(&psv);
        out.push(SampleMeta {
            ply: psv.game_ply(),
            kingrank_bucket,
            friend_zone3: friend_zone3(&board),
            board_non_king_count: board_non_king_count(&board),
        });

        if stride > 1 {
            let skip_bytes = (stride - 1) * record_size;
            if file.seek(SeekFrom::Current(skip_bytes as i64)).is_err() {
                break;
            }
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
    let mut remaining = args.samples;
    for path in &packs {
        if remaining == 0 {
            break;
        }
        match read_samples(path, args.offset, args.stride, remaining) {
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

    if samples.is_empty() {
        eprintln!("No samples loaded.");
        std::process::exit(1);
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

    let mut hist_kingrank = vec![0usize; 9];
    let mut hist_ply_q9 = vec![0usize; 9];
    let mut hist_hybrid = vec![0usize; 9];
    let mut hist_ply_fixed = vec![0usize; 9];
    let mut hist_board_count_q9 = vec![0usize; 9];

    for s in &samples {
        hist_kingrank[s.kingrank_bucket as usize] += 1;
        hist_ply_q9[bucket_by_boundaries(s.ply, &q9_bounds)] += 1;
        hist_board_count_q9[bucket_by_boundaries(s.board_non_king_count as u16, &mat_q9_bounds)] += 1;

        let phase = bucket_by_boundaries(s.ply, &q3_bounds);
        let hybrid_bucket = phase * 3 + s.friend_zone3 as usize;
        hist_hybrid[hybrid_bucket] += 1;

        if let Some(bounds) = &fixed_bounds {
            hist_ply_fixed[bucket_by_boundaries(s.ply, bounds)] += 1;
        }
    }

    print_hist("Current: KingRank 3x3", &hist_kingrank);
    print_hist("Candidate A: Ply Quantile 9", &hist_ply_q9);
    print_hist("Candidate C: BoardNonKingCount Quantile 9", &hist_board_count_q9);
    if fixed_bounds.is_some() {
        print_hist("Candidate A2: Ply Fixed-Boundary 9", &hist_ply_fixed);
    }
    print_hist("Candidate B: (Ply Quantile 3) x (FriendKingZone 3)", &hist_hybrid);
}
