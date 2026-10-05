use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, ensure};
use clap::Args;

use crate::{
    Rand,
    interleave::{InterleaveMode, InterleaveOptions},
    output::{AtomicOutput, ensure_distinct_output},
};

#[derive(Args)]
pub struct ShuffleOptions {
    #[arg(required = true, short, long)]
    pub input: PathBuf,
    #[arg(required = true, short, long)]
    pub output: PathBuf,
    #[arg(required = true, short, long)]
    pub mem_used_mb: usize,
    #[arg(long, default_value = "block")]
    pub interleave_mode: InterleaveMode,
    #[arg(long, default_value = "8")]
    pub interleave_block_mb: usize,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Record size in bytes (default: 32 for ChessBoard, use 40 for PackedSfenValue)
    #[arg(long, default_value = "32")]
    pub record_size: usize,
}
const MIN_TMP_FILES: usize = 4;
const BYTES_PER_MB: usize = 1_048_576;

impl ShuffleOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        let record_size = self.record_size;
        ensure!(record_size > 0, "record_size must be at least 1");
        let input_size = usize::try_from(fs::metadata(&self.input).context("Input file is invalid.")?.len())
            .context("input file size exceeds addressable memory")?;
        ensure!(
            input_size.is_multiple_of(record_size),
            "Input file size ({input_size}) is not a multiple of record size ({record_size})"
        );

        let bytes_used = self.mem_used_mb.checked_mul(BYTES_PER_MB).context("memory limit overflow")?;
        ensure!(bytes_used > 0, "mem_used_mb must be at least 1");
        ensure!(bytes_used >= record_size, "memory limit must include at least one record");

        ensure_distinct_output(std::slice::from_ref(&self.input), &self.output)?;
        let mut output = AtomicOutput::new(&self.output)?;

        println!("# [Shuffling Data] (record_size={})", record_size);
        let time = Instant::now();
        let base_seed = self.seed.unwrap_or_else(Rand::random_seed);

        if input_size <= bytes_used {
            let mut input = File::open(&self.input).context("Failed to read input.")?;
            let read_limit = input_size.checked_add(1).context("input size overflow")?;
            let mut raw_bytes = Vec::with_capacity(read_limit);
            std::io::Read::by_ref(&mut input).take(read_limit as u64).read_to_end(&mut raw_bytes)?;
            ensure!(
                raw_bytes.len() == input_size && input.metadata()?.len() == input_size as u64,
                "input size changed while shuffling: {}",
                self.input.display()
            );

            shuffle_positions(&mut raw_bytes, record_size, Rand::derive_seed(base_seed, 1));

            output.file().write_all(&raw_bytes)?;
            output.commit()?;
        } else {
            drop(output);
            let temp_dir = tempfile::Builder::new().prefix(".bullet-shuffle-").tempdir_in(".")?;
            let num_tmp_files = input_size.div_ceil(bytes_used).max(MIN_TMP_FILES);
            let temp_files =
                (0..num_tmp_files).map(|idx| temp_dir.path().join(format!("part_{}.bin", idx + 1))).collect::<Vec<_>>();

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

            if temp_dir.close().is_err() {
                println!("Error automatically removing temp files");
            }
        }

        println!("> Took {:.2} seconds.", time.elapsed().as_secs_f32());

        Ok(())
    }

    fn split_file(&self, temp_files: &[PathBuf], input_size: usize, base_seed: u64) -> anyhow::Result<()> {
        let record_size = self.record_size;
        let mut input = BufReader::new(File::open(&self.input).with_context(|| "Failed to open file.")?);
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

            input.read_exact(&mut buffer).context("input ended before all expected records were read")?;

            println!("    -> Shuffling in memory");

            shuffle_positions(&mut buffer[..buffer_size], record_size, Rand::derive_seed(base_seed, idx as u64 + 1));

            println!("    -> Writing to temp file");
            file.write_all(&buffer[..buffer_size])?;
        }
        ensure!(
            input.fill_buf()?.is_empty() && input.get_ref().metadata()?.len() == input_size as u64,
            "input size changed while shuffling: {}",
            self.input.display()
        );

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
        data.as_chunks::<TEST_RECORD_SIZE>().0.iter().map(|chunk| chunk[0]).collect()
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

    #[test]
    fn truncated_input_cannot_be_padded_with_zero_records() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let temporary = directory.path().join("part.bin");
        fs::write(&input, make_records(&[1, 2])).unwrap();
        let options = ShuffleOptions {
            input,
            output: directory.path().join("output.bin"),
            mem_used_mb: 1,
            interleave_mode: InterleaveMode::Block,
            interleave_block_mb: 8,
            seed: Some(123),
            record_size: TEST_RECORD_SIZE,
        };
        let error = options.split_file(std::slice::from_ref(&temporary), 3 * TEST_RECORD_SIZE, 123).unwrap_err();
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(fs::metadata(temporary).unwrap().len(), 0);
        assert!(!options.output.exists());
    }
}
