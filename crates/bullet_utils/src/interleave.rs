use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::Rand;
use anyhow::{Context, ensure};
use structopt::StructOpt;

const BYTES_PER_MB: usize = 1_048_576;
const PROGRESS_INTERVAL_RECORDS: usize = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterleaveMode {
    Record,
    Block,
    Concat,
}

impl FromStr for InterleaveMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "record" => Ok(Self::Record),
            "block" => Ok(Self::Block),
            "concat" => Ok(Self::Concat),
            _ => Err(format!("invalid interleave mode '{s}' (expected one of: record, block, concat)")),
        }
    }
}

#[derive(StructOpt)]
pub struct InterleaveOptions {
    #[structopt(required = true, min_values = 2)]
    pub inputs: Vec<PathBuf>,
    #[structopt(required = true, short, long)]
    pub output: PathBuf,
    #[structopt(long, default_value = "block", parse(try_from_str))]
    pub mode: InterleaveMode,
    #[structopt(long, default_value = "8")]
    pub block_mb: usize,
    #[structopt(long)]
    pub seed: Option<u64>,
    /// Record size in bytes (default: 32)
    #[structopt(long, default_value = "32")]
    pub record_size: usize,
}

struct Stream {
    remaining_records: usize,
    reader: BufReader<File>,
}

impl InterleaveOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        match self.mode {
            InterleaveMode::Record => self.run_record(self.seed.unwrap_or_else(Rand::random_seed)),
            InterleaveMode::Block => self.run_block(self.seed.unwrap_or_else(Rand::random_seed)),
            InterleaveMode::Concat => self.run_concat(),
        }
    }

    pub fn new(
        inputs: Vec<PathBuf>,
        output: PathBuf,
        mode: InterleaveMode,
        block_mb: usize,
        seed: Option<u64>,
        record_size: usize,
    ) -> Self {
        Self { inputs, output, mode, block_mb, seed, record_size }
    }

    fn run_record(&self, seed: u64) -> anyhow::Result<()> {
        let size = self.record_size;
        println!("Writing to {:#?}", self.output);
        println!("Reading from:\n{:#?}", self.inputs);

        let (mut streams, total_records) = self.collect_streams()?;
        let expected_bytes = total_records.checked_mul(size).context("interleave size overflow")?;
        let target = File::create(&self.output).with_context(|| "Failed to create output file")?;
        let mut writer = BufWriter::new(target);
        let mut remaining = total_records;
        let mut rng = Rand::with_seed(seed);
        let mut prev = remaining / PROGRESS_INTERVAL_RECORDS;
        let mut value = vec![0u8; size];

        while remaining > 0 {
            let spot = rng.rand() as usize % remaining;
            let idx = pick_weighted_index(&streams, spot, |stream| stream.remaining_records);
            let stream = &mut streams[idx];

            stream.reader.read_exact(&mut value)?;
            writer.write_all(&value)?;

            remaining -= 1;
            stream.remaining_records -= 1;
            if stream.remaining_records == 0 {
                streams.swap_remove(idx);
            }

            report_progress(total_records, remaining, &mut prev);
        }

        writer.flush()?;
        validate_output_size(&self.output, expected_bytes)?;
        if total_records > 0 {
            println!();
        }

        Ok(())
    }

    fn run_block(&self, seed: u64) -> anyhow::Result<()> {
        let size = self.record_size;
        ensure!(self.block_mb > 0, "block size must be at least 1 MB");
        let block_bytes = self.block_mb.checked_mul(BYTES_PER_MB).context("block size overflow")?;
        let block_records = (block_bytes / size).max(1);
        self.run_block_with_records(block_records, seed)
    }

    fn run_block_with_records(&self, block_records: usize, seed: u64) -> anyhow::Result<()> {
        let size = self.record_size;
        ensure!(block_records > 0, "block size must include at least one record");

        println!("Writing to {:#?}", self.output);
        println!("Reading from:\n{:#?}", self.inputs);

        let (mut streams, total_records) = self.collect_streams()?;
        let expected_bytes = total_records.checked_mul(size).context("interleave size overflow")?;
        let target = File::create(&self.output).with_context(|| "Failed to create output file")?;
        let mut writer = BufWriter::new(target);
        let mut buffer = vec![0u8; block_records.checked_mul(size).context("block buffer overflow")?];
        let mut remaining = total_records;
        let mut rng = Rand::with_seed(seed);
        let mut prev = remaining / PROGRESS_INTERVAL_RECORDS;

        while remaining > 0 {
            let spot = rng.rand() as usize % remaining;
            let idx = pick_weighted_index(&streams, spot, |stream| stream.remaining_records);
            let stream = &mut streams[idx];
            let records_to_copy = stream.remaining_records.min(block_records);
            let bytes_to_copy = records_to_copy * size;

            stream.reader.read_exact(&mut buffer[..bytes_to_copy])?;
            writer.write_all(&buffer[..bytes_to_copy])?;

            remaining -= records_to_copy;
            stream.remaining_records -= records_to_copy;
            if stream.remaining_records == 0 {
                streams.swap_remove(idx);
            }

            report_progress(total_records, remaining, &mut prev);
        }

        writer.flush()?;
        validate_output_size(&self.output, expected_bytes)?;
        if total_records > 0 {
            println!();
        }

        Ok(())
    }

    fn run_concat(&self) -> anyhow::Result<()> {
        println!("Writing to {:#?}", self.output);
        println!("Reading from:\n{:#?}", self.inputs);

        let target = File::create(&self.output).with_context(|| "Failed to create output file")?;
        let mut writer = BufWriter::new(target);
        let mut expected_bytes = 0usize;

        for path in &self.inputs {
            let file =
                File::open(path).with_context(|| format!("Failed to open {path}", path = path.to_string_lossy()))?;
            let bytes = file.metadata()?.len() as usize;
            expected_bytes = expected_bytes.checked_add(bytes).context("concat size overflow")?;
            let mut reader = BufReader::new(file);
            io::copy(&mut reader, &mut writer)?;
        }

        writer.flush()?;
        validate_output_size(&self.output, expected_bytes)?;

        Ok(())
    }

    fn collect_streams(&self) -> anyhow::Result<(Vec<Stream>, usize)> {
        let size = self.record_size;
        let mut streams = Vec::new();
        let mut total_records = 0usize;

        for path in &self.inputs {
            let file =
                File::open(path).with_context(|| format!("Failed to open {path}", path = path.to_string_lossy()))?;
            let bytes = file.metadata()?.len() as usize;
            ensure!(bytes % size == 0, "input file size is not a multiple of {size}: {}", path.display());

            let records = bytes / size;
            if records > 0 {
                total_records = total_records.checked_add(records).context("interleave record count overflow")?;
                streams.push(Stream { remaining_records: records, reader: BufReader::new(file) });
            }
        }

        Ok((streams, total_records))
    }
}

fn pick_weighted_index<T, F>(items: &[T], mut spot: usize, weight_of: F) -> usize
where
    F: Fn(&T) -> usize,
{
    for (idx, item) in items.iter().enumerate() {
        let weight = weight_of(item);
        if spot < weight {
            return idx;
        }
        spot -= weight;
    }

    unreachable!("weighted spot must land on an item")
}

fn report_progress(total_records: usize, remaining_records: usize, prev: &mut usize) {
    if remaining_records / PROGRESS_INTERVAL_RECORDS < *prev {
        *prev = remaining_records / PROGRESS_INTERVAL_RECORDS;
        let written = total_records - remaining_records;
        print!("Written {written} / {total_records} ({:.2})\r", written as f32 / total_records as f32 * 100.0);
        let _ = std::io::stdout().flush();
    }
}

fn validate_output_size(path: &Path, expected_bytes: usize) -> anyhow::Result<()> {
    let actual_bytes = fs::metadata(path)?.len() as usize;
    ensure!(
        actual_bytes == expected_bytes,
        "Output file size {actual_bytes} does not match expected size {expected_bytes}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_dir(label: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        dir.push(format!("bullet_utils_{label}_{}_{}", std::process::id(), unique));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    const SIZE: usize = 32;

    fn record(byte: u8) -> [u8; SIZE] {
        let mut record = [byte; SIZE];
        record[0] = byte;
        record
    }

    fn write_records(path: &Path, values: &[u8]) {
        let mut bytes = Vec::with_capacity(values.len() * SIZE);
        for &value in values {
            bytes.extend_from_slice(&record(value));
        }
        fs::write(path, bytes).unwrap();
    }

    fn read_records(path: &Path) -> Vec<[u8; SIZE]> {
        let bytes = fs::read(path).unwrap();
        assert_eq!(0, bytes.len() % SIZE);
        bytes
            .chunks_exact(SIZE)
            .map(|chunk| {
                let mut record = [0u8; SIZE];
                record.copy_from_slice(chunk);
                record
            })
            .collect()
    }

    fn sorted_record_ids(path: &Path) -> Vec<u8> {
        let mut ids = read_records(path).into_iter().map(|record| record[0]).collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn weighted_picker_uses_correct_boundaries() {
        let weights = [1usize, 1usize];
        assert_eq!(pick_weighted_index(&weights, 0, |weight| *weight), 0);
        assert_eq!(pick_weighted_index(&weights, 1, |weight| *weight), 1);
    }

    #[test]
    fn record_mode_is_reproducible() {
        let dir = unique_dir("record_mode");
        let input_a = dir.join("a.bin");
        let input_b = dir.join("b.bin");
        let output_a = dir.join("out_a.bin");
        let output_b = dir.join("out_b.bin");
        write_records(&input_a, &[1, 2, 3]);
        write_records(&input_b, &[4, 5, 6]);

        InterleaveOptions::new(
            vec![input_a.clone(), input_b.clone()],
            output_a.clone(),
            InterleaveMode::Record,
            8,
            Some(42),
            SIZE,
        )
        .run()
        .unwrap();
        InterleaveOptions::new(
            vec![input_a.clone(), input_b.clone()],
            output_b.clone(),
            InterleaveMode::Record,
            8,
            Some(42),
            SIZE,
        )
        .run()
        .unwrap();

        assert_eq!(fs::read(&output_a).unwrap(), fs::read(&output_b).unwrap());
        assert_eq!(sorted_record_ids(&output_a), vec![1, 2, 3, 4, 5, 6]);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn block_mode_preserves_all_records() {
        let dir = unique_dir("block_mode");
        let input_a = dir.join("a.bin");
        let input_b = dir.join("b.bin");
        let output = dir.join("out.bin");
        write_records(&input_a, &[1, 2, 3, 4]);
        write_records(&input_b, &[5, 6, 7, 8]);

        let options = InterleaveOptions::new(
            vec![input_a.clone(), input_b.clone()],
            output.clone(),
            InterleaveMode::Block,
            8,
            Some(7),
            SIZE,
        );
        options.run_block_with_records(2, 7).unwrap();

        assert_eq!(sorted_record_ids(&output), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concat_mode_matches_plain_concatenation() {
        let dir = unique_dir("concat_mode");
        let input_a = dir.join("a.bin");
        let input_b = dir.join("b.bin");
        let output = dir.join("out.bin");
        write_records(&input_a, &[1, 2]);
        write_records(&input_b, &[3, 4]);

        InterleaveOptions::new(
            vec![input_a.clone(), input_b.clone()],
            output.clone(),
            InterleaveMode::Concat,
            8,
            Some(1),
            SIZE,
        )
        .run()
        .unwrap();

        let mut expected = fs::read(&input_a).unwrap();
        expected.extend_from_slice(&fs::read(&input_b).unwrap());
        assert_eq!(fs::read(&output).unwrap(), expected);

        fs::remove_dir_all(dir).unwrap();
    }
}
