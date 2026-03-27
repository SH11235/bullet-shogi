use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, IoSliceMut, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, ensure};
use bulletformat::ChessBoard;
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
}

const CHESS_BOARD_SIZE: usize = std::mem::size_of::<ChessBoard>();
const MIN_TMP_FILES: usize = 4;
const BYTES_PER_MB: usize = 1_048_576;
const TMP_DIR: &str = "./tmp";

impl ShuffleOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        let input_size = fs::metadata(self.input.clone()).with_context(|| "Input file is invalid.")?.len() as usize;
        assert_eq!(0, input_size % CHESS_BOARD_SIZE);

        let bytes_used = self.mem_used_mb.checked_mul(BYTES_PER_MB).context("memory limit overflow")?;
        ensure!(bytes_used > 0, "mem_used_mb must be at least 1");

        // Test path before doing useless work
        validate_output_path(Path::new(&self.output))
            .with_context(|| format!("Invalid output path: {}", self.output.display()))?;

        println!("# [Shuffling Data]");
        let time = Instant::now();
        let base_seed = self.seed.unwrap_or_else(Rand::random_seed);

        if input_size <= bytes_used {
            let mut raw_bytes = std::fs::read(&self.input).with_context(|| "Failed to read input.")?;

            shuffle_positions(&mut raw_bytes, Rand::derive_seed(base_seed, 1));

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
        let mut input = BufReader::new(File::open(self.input.clone()).with_context(|| "Failed to open file.")?);
        let temp_files = temp_files
            .iter()
            .map(|f| File::create(f).with_context(|| "Tmp file could not be created."))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let total_positions = input_size / CHESS_BOARD_SIZE;
        let ideal_positions_per_file = total_positions / temp_files.len();
        let mut positions_per_file = vec![ideal_positions_per_file; temp_files.len()];
        let remaining_positions = total_positions % temp_files.len();
        for size in positions_per_file.iter_mut().take(remaining_positions) {
            *size += 1;
        }

        for (idx, mut file) in temp_files.iter().enumerate() {
            println!("# [Shuffling temp file {} / {}]", idx + 1, temp_files.len());
            println!("    -> Reading into ram");

            let buffer_size = positions_per_file[idx] * CHESS_BOARD_SIZE;
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

            shuffle_positions(&mut buffer[..buffer_size], Rand::derive_seed(base_seed, idx as u64 + 1));

            println!("    -> Writing to temp file");
            file.write_all(&buffer[..buffer_size])?;
        }

        Ok(())
    }
}

fn shuffle_positions(data: &mut [u8], seed: u64) {
    assert_eq!(data.len() % CHESS_BOARD_SIZE, 0);

    let len = data.len() / CHESS_BOARD_SIZE;
    let mut rng = Rand::with_seed(seed);

    let records = unsafe {
        // SAFETY: `[u8; CHESS_BOARD_SIZE]` has alignment 1, and `data.len()` is a multiple
        // of `CHESS_BOARD_SIZE`, so the slice can be reinterpreted as fixed-size records.
        std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<[u8; CHESS_BOARD_SIZE]>(), len)
    };

    for i in (0..len).rev() {
        let idx = rng.rand() as usize % (i + 1);
        records.swap(idx, i);
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

    fn make_records(values: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * CHESS_BOARD_SIZE);
        for &value in values {
            let mut record = [value; CHESS_BOARD_SIZE];
            record[0] = value;
            bytes.extend_from_slice(&record);
        }
        bytes
    }

    fn record_ids(data: &[u8]) -> Vec<u8> {
        data.chunks_exact(CHESS_BOARD_SIZE).map(|chunk| chunk[0]).collect()
    }

    #[test]
    fn shuffle_positions_is_reproducible() {
        let mut a = make_records(&[1, 2, 3, 4, 5]);
        let mut b = make_records(&[1, 2, 3, 4, 5]);

        shuffle_positions(&mut a, 123);
        shuffle_positions(&mut b, 123);

        assert_eq!(a, b);
    }

    #[test]
    fn shuffle_positions_preserves_all_records() {
        let mut data = make_records(&[1, 2, 3, 4, 5]);

        shuffle_positions(&mut data, 456);

        let mut ids = record_ids(&data);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }
}
