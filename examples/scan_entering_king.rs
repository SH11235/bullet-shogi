//! Detect "entering king" (入玉) signature in PSV files by decoding king positions.
//!
//! Usage:
//!   cargo run --release --example scan_entering_king -- \
//!     --files <comma-separated paths> \
//!     [--max-positions 1000000]
//!
//! Reports per-file:
//!   - mean game_ply
//!   - black king rank distribution (rank 0=top/white side)
//!   - % positions where black king has entered white side (rank ≤ 2)
//!   - % positions where white king has entered black side (rank ≥ 6)
//!   - % both kings have entered (両入玉)
//!
//! Square indexing in bullet-shogi:
//!   index = file*9 + rank, file 0..8 (1筋..9筋), rank 0..8 (1段..9段, 0=top)
//!   Black starts at rank 6,7,8; white at rank 0,1,2.
//!   Entering king = king has crossed into opponent's three-rank territory.

use std::{
    fs::File,
    io::{self, BufReader, Read},
    mem::size_of,
    path::PathBuf,
};

use bullet_lib::shogi::PackedSfenValue;
use clap::Parser;

const PSV_SIZE: usize = size_of::<PackedSfenValue>();

#[derive(Parser, Debug)]
#[command(name = "scan_entering_king")]
struct Args {
    /// Comma-separated PSV files
    #[arg(long)]
    files: String,

    /// Max positions to scan per file
    #[arg(long, default_value_t = 1_000_000)]
    max_positions: usize,
}

#[derive(Default)]
struct Stats {
    n: u64,
    ply_sum: u64,
    bk_rank_hist: [u64; 9],
    wk_rank_hist: [u64; 9],
    /// black king rank ≤ 2 (entered white territory)
    bk_entered: u64,
    /// white king rank ≥ 6 (entered black territory)
    wk_entered: u64,
    /// both kings entered
    both_entered: u64,
    /// black king rank ≤ 4 (crossed river / midline)
    bk_past_mid: u64,
    /// white king rank ≥ 4
    wk_past_mid: u64,
}

fn scan_file(path: &PathBuf, max_positions: usize) -> io::Result<Stats> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut buf = [0u8; PSV_SIZE];
    let mut stats = Stats::default();

    while stats.n < max_positions as u64 {
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let psv: PackedSfenValue = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const PackedSfenValue) };
        let board = psv.decode();
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            continue;
        }

        let bk_rank = board.black_king_sq.rank() as usize; // 0=top
        let wk_rank = board.white_king_sq.rank() as usize;

        stats.n += 1;
        stats.ply_sum += board.ply as u64;
        if bk_rank < 9 {
            stats.bk_rank_hist[bk_rank] += 1;
        }
        if wk_rank < 9 {
            stats.wk_rank_hist[wk_rank] += 1;
        }
        let bk_entered = bk_rank <= 2;
        let wk_entered = wk_rank >= 6;
        if bk_entered {
            stats.bk_entered += 1;
        }
        if wk_entered {
            stats.wk_entered += 1;
        }
        if bk_entered && wk_entered {
            stats.both_entered += 1;
        }
        if bk_rank <= 4 {
            stats.bk_past_mid += 1;
        }
        if wk_rank >= 4 {
            stats.wk_past_mid += 1;
        }
    }
    Ok(stats)
}

fn print_report(label: &str, s: &Stats) {
    if s.n == 0 {
        println!("{label}: empty");
        return;
    }
    let n = s.n as f64;
    let pct = |x: u64| 100.0 * x as f64 / n;
    println!("\n=== {label} ===");
    println!("  positions: {}", s.n);
    println!("  mean game_ply: {:.1}", s.ply_sum as f64 / n);
    println!(
        "  black king rank hist (0=top, 8=bottom): {:?}",
        s.bk_rank_hist.iter().map(|c| (100.0 * *c as f64 / n).round() as u32).collect::<Vec<_>>()
    );
    println!(
        "  white king rank hist (0=top, 8=bottom): {:?}",
        s.wk_rank_hist.iter().map(|c| (100.0 * *c as f64 / n).round() as u32).collect::<Vec<_>>()
    );
    println!("  black king entered (rank ≤ 2):  {:>6.2}%", pct(s.bk_entered));
    println!("  white king entered (rank ≥ 6):  {:>6.2}%", pct(s.wk_entered));
    println!("  both kings entered (両入玉):     {:>6.2}%", pct(s.both_entered));
    println!("  black king past midline (≤ 4):  {:>6.2}%", pct(s.bk_past_mid));
    println!("  white king past midline (≥ 4):  {:>6.2}%", pct(s.wk_past_mid));
    println!("  either king entered: {:>6.2}%", pct(s.bk_entered + s.wk_entered - s.both_entered));
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let paths: Vec<PathBuf> = args.files.split(',').map(|s| PathBuf::from(s.trim())).filter(|p| p.is_file()).collect();
    if paths.is_empty() {
        eprintln!("no input files");
        std::process::exit(1);
    }
    println!("Scanning {} file(s), max {} positions/file", paths.len(), args.max_positions);

    for p in &paths {
        let label = p.file_name().and_then(|s| s.to_str()).unwrap_or("unknown");
        let stats = scan_file(p, args.max_positions)?;
        print_report(label, &stats);
    }
    Ok(())
}
