//! Merge scores from two PackedSfenValue files by averaging.
//!
//! Usage:
//!   cargo run --release -p bullet_lib --example merge_scores -- \
//!     --dir-a data/DLSuisho15b \
//!     --dir-b data/aobazero_kd_20240329 \
//!     --prefix-a shuffled_ \
//!     --prefix-b aobazero_ \
//!     --output-dir data/ensemble_suisho_aoba \
//!     --output-prefix ensemble_ \
//!     --threads 4
//!
//! Processes matching file pairs (by numeric suffix) in parallel.
//! Each record: 40 bytes (PackedSfen[32] + score[2] + move16[2] + game_ply[2] + result[1] + pad[1])
//! Output keeps PackedSfen, move16, game_ply, result, pad from file A; score = avg(A, B).

use clap::Parser;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const RECORD_SIZE: usize = 40;
const SFEN_SIZE: usize = 32;
const SCORE_OFFSET: usize = 32;
/// Buffer: read/write 10000 records at a time (~400KB)
const BUF_RECORDS: usize = 10000;

#[derive(Parser)]
#[command(name = "merge_scores")]
struct Args {
    /// Directory containing input files A (e.g. DLSuisho15b)
    #[arg(long)]
    dir_a: PathBuf,

    /// Directory containing input files B (e.g. aobazero_kd_20240329)
    #[arg(long)]
    dir_b: PathBuf,

    /// Filename prefix for files in dir-a (e.g. "shuffled_")
    #[arg(long)]
    prefix_a: String,

    /// Filename prefix for files in dir-b (e.g. "aobazero_")
    #[arg(long)]
    prefix_b: String,

    /// Output directory
    #[arg(long)]
    output_dir: PathBuf,

    /// Output filename prefix (e.g. "ensemble_")
    #[arg(long, default_value = "ensemble_")]
    output_prefix: String,

    /// Number of file pairs to process in parallel
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Verify that PackedSfen matches between A and B (slower but safer)
    #[arg(long, default_value = "true")]
    verify: bool,
}

/// Discover matching file pairs by numeric suffix.
/// Returns Vec<(suffix_string, path_a, path_b)> sorted by suffix.
fn discover_pairs(dir_a: &Path, dir_b: &Path, prefix_a: &str, prefix_b: &str) -> Vec<(String, PathBuf, PathBuf)> {
    let mut pairs = Vec::new();

    let entries_a: Vec<_> = fs::read_dir(dir_a)
        .expect("Cannot read dir-a")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with(prefix_a) && name.ends_with(".bin")
        })
        .collect();

    for entry in &entries_a {
        let name_a = entry.file_name().to_string_lossy().to_string();
        // Extract suffix: e.g. "shuffled_01.bin" -> "01"
        let suffix = &name_a[prefix_a.len()..name_a.len() - 4]; // strip prefix and ".bin"
        let name_b = format!("{}{}.bin", prefix_b, suffix);
        let path_b = dir_b.join(&name_b);

        if path_b.exists() {
            pairs.push((suffix.to_string(), entry.path(), path_b));
        } else {
            eprintln!("Warning: no matching file for {} (expected {})", name_a, path_b.display());
        }
    }

    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

/// Process a single file pair: read A and B, write averaged scores to output.
/// Returns the number of records processed.
fn process_pair(path_a: &Path, path_b: &Path, output: &Path, verify: bool, total_records: &AtomicU64) -> u64 {
    let file_a = fs::File::open(path_a).unwrap_or_else(|e| panic!("Cannot open {}: {}", path_a.display(), e));
    let file_b = fs::File::open(path_b).unwrap_or_else(|e| panic!("Cannot open {}: {}", path_b.display(), e));
    let file_out = fs::File::create(output).unwrap_or_else(|e| panic!("Cannot create {}: {}", output.display(), e));

    let size_a = file_a.metadata().unwrap().len();
    let size_b = file_b.metadata().unwrap().len();
    assert_eq!(
        size_a,
        size_b,
        "File size mismatch: {} ({}) vs {} ({})",
        path_a.display(),
        size_a,
        path_b.display(),
        size_b
    );
    assert_eq!(size_a as usize % RECORD_SIZE, 0, "File size not a multiple of record size: {}", path_a.display());

    let num_records = size_a as usize / RECORD_SIZE;
    let mut reader_a = BufReader::with_capacity(RECORD_SIZE * BUF_RECORDS, file_a);
    let mut reader_b = BufReader::with_capacity(RECORD_SIZE * BUF_RECORDS, file_b);
    let mut writer = BufWriter::with_capacity(RECORD_SIZE * BUF_RECORDS, file_out);

    let mut buf_a = vec![0u8; RECORD_SIZE * BUF_RECORDS];
    let mut buf_b = vec![0u8; RECORD_SIZE * BUF_RECORDS];
    let mut records_done: u64 = 0;
    let mut mismatches: u64 = 0;

    loop {
        let n_a = read_exact_or_eof(&mut reader_a, &mut buf_a);
        let n_b = read_exact_or_eof(&mut reader_b, &mut buf_b);
        assert_eq!(n_a, n_b, "Read size mismatch at record {}", records_done);

        if n_a == 0 {
            break;
        }

        let n_records = n_a / RECORD_SIZE;
        for i in 0..n_records {
            let off = i * RECORD_SIZE;
            let rec_a = &buf_a[off..off + RECORD_SIZE];
            let rec_b = &buf_b[off..off + RECORD_SIZE];

            // Verify PackedSfen match
            if verify && rec_a[..SFEN_SIZE] != rec_b[..SFEN_SIZE] {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!("PackedSfen mismatch at record {} in {}", records_done + i as u64, path_a.display());
                }
            }

            // Average scores
            let score_a = i16::from_le_bytes([rec_a[SCORE_OFFSET], rec_a[SCORE_OFFSET + 1]]);
            let score_b = i16::from_le_bytes([rec_b[SCORE_OFFSET], rec_b[SCORE_OFFSET + 1]]);
            let avg = average_scores(score_a, score_b);

            // Write: copy record A, replace score
            buf_a[off + SCORE_OFFSET] = avg.to_le_bytes()[0];
            buf_a[off + SCORE_OFFSET + 1] = avg.to_le_bytes()[1];
        }

        writer.write_all(&buf_a[..n_a]).expect("Write failed");
        records_done += n_records as u64;
        total_records.fetch_add(n_records as u64, Ordering::Relaxed);
    }

    if mismatches > 0 {
        eprintln!(
            "WARNING: {} PackedSfen mismatches in {} ({} records total)",
            mismatches,
            path_a.display(),
            num_records
        );
    }

    assert_eq!(
        records_done, num_records as u64,
        "Record count mismatch: expected {}, got {}",
        num_records, records_done
    );

    records_done
}

/// Average two scores. Handles sentinel values (±32000).
fn average_scores(a: i16, b: i16) -> i16 {
    // Use i32 to avoid overflow
    let sum = a as i32 + b as i32;
    // Round toward zero (truncate)
    (sum / 2) as i16
}

/// Read as many bytes as possible (up to buf.len()), return count read.
/// Handles partial reads from BufReader.
fn read_exact_or_eof(reader: &mut impl Read, buf: &mut [u8]) -> usize {
    let mut total = 0;
    while total < buf.len() {
        match reader.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) => panic!("Read error: {}", e),
        }
    }
    // Ensure we read a multiple of RECORD_SIZE
    assert_eq!(total % RECORD_SIZE, 0, "Read {} bytes, not a multiple of {}", total, RECORD_SIZE);
    total
}

fn main() {
    let args = Args::parse();

    // Create output directory
    fs::create_dir_all(&args.output_dir).expect("Cannot create output directory");

    // Discover file pairs
    let pairs = discover_pairs(&args.dir_a, &args.dir_b, &args.prefix_a, &args.prefix_b);
    if pairs.is_empty() {
        eprintln!("No matching file pairs found!");
        std::process::exit(1);
    }

    println!("Found {} file pairs:", pairs.len());
    for (suffix, pa, pb) in &pairs {
        let size = fs::metadata(pa).unwrap().len();
        let records = size / RECORD_SIZE as u64;
        println!(
            "  {} + {} -> {}{}.bin  ({} records, {:.1} GB)",
            pa.file_name().unwrap().to_string_lossy(),
            pb.file_name().unwrap().to_string_lossy(),
            args.output_prefix,
            suffix,
            records,
            size as f64 / 1e9,
        );
    }
    println!();
    println!("Threads: {}, Verify: {}", args.threads, args.verify);
    println!();

    let total_records = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    // Process pairs in parallel using a thread pool
    let pairs_with_output: Vec<_> = pairs
        .iter()
        .map(|(suffix, pa, pb)| {
            let output = args.output_dir.join(format!("{}{}.bin", args.output_prefix, suffix));
            (pa.clone(), pb.clone(), output)
        })
        .collect();

    // Simple thread pool: process `threads` pairs at a time
    let chunks: Vec<_> = pairs_with_output.chunks(args.threads).collect();
    let mut completed = 0;

    for chunk in chunks {
        let handles: Vec<_> = chunk
            .iter()
            .map(|(pa, pb, out)| {
                let pa = pa.clone();
                let pb = pb.clone();
                let out = out.clone();
                let verify = args.verify;
                let total = Arc::clone(&total_records);
                std::thread::spawn(move || {
                    let n = process_pair(&pa, &pb, &out, verify, &total);
                    (out, n)
                })
            })
            .collect();

        for h in handles {
            let (out, n) = h.join().expect("Thread panicked");
            completed += 1;
            let elapsed = start.elapsed().as_secs_f64();
            let total = total_records.load(Ordering::Relaxed);
            println!(
                "[{}/{}] {} done ({} records, {:.0} records/sec overall)",
                completed,
                pairs_with_output.len(),
                out.file_name().unwrap().to_string_lossy(),
                n,
                total as f64 / elapsed,
            );
        }
    }

    let elapsed = start.elapsed();
    let total = total_records.load(Ordering::Relaxed);
    println!();
    println!(
        "All done: {} records in {:.1}s ({:.0} records/sec)",
        total,
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64(),
    );
}
