use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, IoSliceMut, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use structopt::StructOpt;

use crate::{
    Rand,
    interleave::{InterleaveMode, InterleaveOptions},
};

#[derive(StructOpt)]
pub struct ShuffleOptions {
    #[structopt(required = true, short, long)]
    pub input: PathBuf,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
    #[structopt(required = true, short, long)]
    pub mem_used_mb: usize,
    #[structopt(long, default_value = "block", parse(try_from_str))]
    pub interleave_mode: InterleaveMode,
    #[structopt(long, default_value = "8")]
    pub interleave_block_mb: usize,
    #[structopt(long)]
    pub seed: Option<u64>,
    /// Record size in bytes (default: 32 for ChessBoard, use 40 for PackedSfenValue)
    #[structopt(long, default_value = "32")]
    pub record_size: usize,
}
const MIN_TMP_FILES: usize = 4;
const BYTES_PER_MB: usize = 1_048_576;
const TMP_DIR: &str = "./tmp";

impl ShuffleOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        let record_size = self.record_size;
        ensure!(record_size > 0, "record_size must be at least 1");
        let input_size = fs::metadata(self.input.clone()).with_context(|| "Input file is invalid.")?.len() as usize;
        ensure!(
            input_size.is_multiple_of(record_size),
            "Input file size ({input_size}) is not a multiple of record size ({record_size})"
        );

        let bytes_used = self.mem_used_mb.checked_mul(BYTES_PER_MB).context("memory limit overflow")?;
        ensure!(bytes_used > 0, "mem_used_mb must be at least 1");

        // Test path before doing useless work
        validate_output_path(Path::new(&self.output))
            .with_context(|| format!("Invalid output path: {}", self.output.display()))?;

        println!("# [Shuffling Data] (record_size={})", record_size);
        let time = Instant::now();
        let base_seed = self.seed.unwrap_or_else(Rand::random_seed);

        if input_size <= bytes_used {
            let mut raw_bytes = std::fs::read(&self.input).with_context(|| "Failed to read input.")?;

            shuffle_positions(&mut raw_bytes, record_size, Rand::derive_seed(base_seed, 1));

            let mut file = File::create(&self.output).with_context(|| "Provide a correct path!")?;
            file.write_all(&raw_bytes)?;
        } else {
            let temp_dir = Path::new(TMP_DIR);
            if !Path::exists(temp_dir) {
                fs::create_dir(temp_dir).with_context(|| "Temp dir could not be created.")?;
            }
            let num_tmp_files = input_size.div_ceil(bytes_used).max(MIN_TMP_FILES);
            let temp_files = (0..num_tmp_files)
                .map(|idx| {
                    let output_file = format!(
                        "{}/part_{}.bin",
                        temp_dir.to_str().with_context(|| "Failed to convert path to string.")?,
                        idx + 1
                    );
                    Ok(PathBuf::from(output_file))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            self.split_file(&temp_files, input_size, base_seed)?;

            println!("# [Finished splitting data. Interleaving...]");
            let interleave_seed = Rand::derive_seed(base_seed, temp_files.len() as u64 + 1);
            let interleave = InterleaveOptions::new(
                temp_files.to_vec(),
                self.output.clone(),
                self.interleave_mode,
                self.interleave_block_mb,
                Some(interleave_seed),
                record_size,
            );
            interleave.run()?;

            if fs::remove_dir_all(temp_dir).is_err() {
                println!("Error automatically removing temp files");
            }
        }

        println!("> Took {:.2} seconds.", time.elapsed().as_secs_f32());

        Ok(())
    }

    fn split_file(&self, temp_files: &[PathBuf], input_size: usize, base_seed: u64) -> anyhow::Result<()> {
        let record_size = self.record_size;
        let mut input = BufReader::new(File::open(self.input.clone()).with_context(|| "Failed to open file.")?);
        let temp_files = temp_files
            .iter()
            .map(|f| File::create(f).with_context(|| "Tmp file could not be created."))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let total_positions = input_size / record_size;
        let ideal_positions_per_file = total_positions / temp_files.len();
        let mut positions_per_file = vec![ideal_positions_per_file; temp_files.len()];
        let remaining_positions = total_positions % temp_files.len();
        for size in positions_per_file.iter_mut().take(remaining_positions) {
            *size += 1;
        }

        for (idx, mut file) in temp_files.iter().enumerate() {
            println!("# [Shuffling temp file {} / {}]", idx + 1, temp_files.len());
            println!("    -> Reading into ram");

            let buffer_size = positions_per_file[idx] * record_size;
            let mut buffer = vec![0u8; buffer_size];

            // performs better than a read_exact
            let chunk_size = 1024 * 1024;
            let mut offset = 0;

            while offset < buffer_size {
                let remaining = buffer_size - offset;
                let current_chunk = remaining.min(chunk_size);
                let mut iovec = [IoSliceMut::new(&mut buffer[offset..offset + current_chunk])];
                let bytes_read = input.read_vectored(&mut iovec)?;

                if bytes_read == 0 {
                    break;
                }

                offset += bytes_read;
            }

            println!("    -> Shuffling in memory");

            shuffle_positions(&mut buffer[..buffer_size], record_size, Rand::derive_seed(base_seed, idx as u64 + 1));

            println!("    -> Writing to temp file");
            file.write_all(&buffer[..buffer_size])?;
        }

        Ok(())
    }
}

fn shuffle_positions(data: &mut [u8], record_size: usize, seed: u64) {
    assert_eq!(data.len() % record_size, 0);

    let len = data.len() / record_size;
    let mut rng = Rand::with_seed(seed);

    for i in (1..len).rev() {
        let idx = rng.rand() as usize % (i + 1);
        if idx != i {
            // Swap records at positions idx and i
            let (lo, hi) = if idx < i { (idx, i) } else { (i, idx) };
            let (left, right) = data.split_at_mut(hi * record_size);
            left[lo * record_size..lo * record_size + record_size].swap_with_slice(&mut right[..record_size]);
        }
    }
}

/// Test if we can write to the output path
fn validate_output_path(path: &Path) -> anyhow::Result<()> {
    match OpenOptions::new().write(true).create(true).truncate(false).open(path) {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("Cannot create file at specified path: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_RECORD_SIZE: usize = 32;

    fn make_records(values: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * TEST_RECORD_SIZE);
        for &value in values {
            let mut record = [value; TEST_RECORD_SIZE];
            record[0] = value;
            bytes.extend_from_slice(&record);
        }
        bytes
    }

    fn record_ids(data: &[u8]) -> Vec<u8> {
        data.chunks_exact(TEST_RECORD_SIZE).map(|chunk| chunk[0]).collect()
    }

    #[test]
    fn shuffle_positions_is_reproducible() {
        let mut a = make_records(&[1, 2, 3, 4, 5]);
        let mut b = make_records(&[1, 2, 3, 4, 5]);

        shuffle_positions(&mut a, TEST_RECORD_SIZE, 123);
        shuffle_positions(&mut b, TEST_RECORD_SIZE, 123);

        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_positions_preserves_all_records() {
        let mut data = make_records(&[1, 2, 3, 4, 5]);

        shuffle_positions(&mut data, TEST_RECORD_SIZE, 456);

        let mut ids = record_ids(&data);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }
}
