/// Compare bucket distributions of multiple progress.bin files on given data.
///
/// Usage:
///   cargo run --release --example compare_progress_buckets -- \
///     --data /path/to/data.pack \
///     --progress /path/to/progress1.bin,label1 \
///     --progress /path/to/progress2.bin,label2 \
///     --max-positions 200000
///
use std::{
    fs::File,
    io::{self, BufReader, Read},
    mem::size_of,
    path::{Path, PathBuf},
};

use bullet_lib::{
    game::outputs::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs},
    shogi::PackedSfenValue,
};
use clap::Parser;

const PSV_SIZE: usize = size_of::<PackedSfenValue>();
const NUM_BUCKETS: usize = 8;

#[derive(Parser, Debug)]
#[command(name = "compare_progress_buckets")]
struct Args {
    /// Data files (comma-separated or repeated)
    #[arg(long)]
    data: String,

    /// progress.bin files with labels: path,label (can be repeated)
    #[arg(long, action = clap::ArgAction::Append)]
    progress: Vec<String>,

    /// Max positions to evaluate per data file
    #[arg(long, default_value = "200000")]
    max_positions: usize,
}

struct ProgressWeights {
    label: String,
    weights: Vec<f32>,
}

fn load_progress_weights(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("size mismatch: got {} bytes, expected {}", bytes.len(), expected),
        ));
    }
    Ok(bytes.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b) as f32).collect())
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn compute_progress(weights: &[f32], psv: &PackedSfenValue) -> f32 {
    let mut indices = Vec::with_capacity(96);
    ShogiProgressKPAbs::collect_active_indices(psv, &mut indices);

    let mut sum = 0.0f32;
    for &idx in &indices {
        if idx < weights.len() {
            sum += weights[idx];
        }
    }
    sigmoid(sum)
}

struct BucketStats {
    buckets: [usize; NUM_BUCKETS],
    total: usize,
    progress_sum: f64,
}

impl BucketStats {
    fn new() -> Self {
        Self { buckets: [0; NUM_BUCKETS], total: 0, progress_sum: 0.0 }
    }

    fn add(&mut self, progress: f32) {
        let b = ((progress * NUM_BUCKETS as f32).floor() as i32).clamp(0, NUM_BUCKETS as i32 - 1) as usize;
        self.buckets[b] += 1;
        self.total += 1;
        self.progress_sum += progress as f64;
    }

    fn mean_progress(&self) -> f64 {
        if self.total > 0 { self.progress_sum / self.total as f64 } else { 0.0 }
    }

    fn std_pct(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let ideal = 100.0 / NUM_BUCKETS as f64;
        let var: f64 = self
            .buckets
            .iter()
            .map(|&c| {
                let pct = 100.0 * c as f64 / self.total as f64;
                (pct - ideal).powi(2)
            })
            .sum::<f64>()
            / NUM_BUCKETS as f64;
        var.sqrt()
    }

    fn effective_buckets(&self) -> usize {
        let threshold = (self.total as f64 * 0.05) as usize;
        self.buckets.iter().filter(|&&c| c > threshold).count()
    }
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    // Parse progress files
    let mut progress_list: Vec<ProgressWeights> = Vec::new();
    for spec in &args.progress {
        let parts: Vec<&str> = spec.splitn(2, ',').collect();
        let path = PathBuf::from(parts[0]);
        let label = if parts.len() > 1 {
            parts[1].to_string()
        } else {
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
        };
        println!("Loading {}: {}", label, path.display());
        let weights = load_progress_weights(&path)?;
        progress_list.push(ProgressWeights { label, weights });
    }

    if progress_list.is_empty() {
        eprintln!("No --progress files specified");
        return Ok(());
    }

    // Parse data files
    let data_files: Vec<PathBuf> =
        args.data.split(',').map(|s| PathBuf::from(s.trim())).filter(|p| p.exists()).collect();

    if data_files.is_empty() {
        eprintln!("No valid data files found");
        return Ok(());
    }

    // For each data file, compute bucket distributions
    for data_path in &data_files {
        let data_label = data_path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown");

        println!("\n{}", "=".repeat(78));
        println!("  Dataset: {} ({} positions)", data_label, args.max_positions);
        println!("{}", "=".repeat(78));

        let file = File::open(data_path)?;
        let mut reader = BufReader::new(file);
        let mut buf = vec![0u8; PSV_SIZE];

        let mut stats: Vec<BucketStats> = progress_list.iter().map(|_| BucketStats::new()).collect();
        let mut count = 0usize;

        while count < args.max_positions {
            match reader.read_exact(&mut buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let psv: PackedSfenValue = unsafe { std::ptr::read(buf.as_ptr() as *const PackedSfenValue) };

            for (i, pw) in progress_list.iter().enumerate() {
                let p = compute_progress(&pw.weights, &psv);
                stats[i].add(p);
            }

            count += 1;
        }

        // Print header
        print!("{:>8}", "bucket");
        for pw in &progress_list {
            print!("  {:>20}", pw.label);
        }
        println!();
        println!("{}", "-".repeat(10 + 22 * progress_list.len()));

        // Print bucket rows
        for b in 0..NUM_BUCKETS {
            print!("  b{}    ", b);
            for (i, _) in progress_list.iter().enumerate() {
                let c = stats[i].buckets[b];
                let pct = if stats[i].total > 0 { 100.0 * c as f64 / stats[i].total as f64 } else { 0.0 };
                print!("  {:>8} ({:5.1}%)", c, pct);
            }
            println!();
        }

        println!("{}", "-".repeat(10 + 22 * progress_list.len()));

        // Summary
        print!("{:>8}", "mean_p");
        for s in &stats {
            print!("  {:>20.4}", s.mean_progress());
        }
        println!();

        print!("{:>8}", "std%");
        for s in &stats {
            print!("  {:>20.2}", s.std_pct());
        }
        println!();

        print!("{:>8}", "eff_bkt");
        for s in &stats {
            print!("  {:>20}", s.effective_buckets());
        }
        println!();
    }

    Ok(())
}
