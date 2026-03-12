use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    mem::MaybeUninit,
    path::PathBuf,
    slice,
};

use super::DataLoader;

/// ### Safety
/// This indicates that the type can be validly transmuted from
/// *any* sequence of bytes of the same size as the struct.
pub unsafe trait CanBeDirectlySequentiallyLoaded: Copy + 'static {}

#[derive(Clone)]
pub struct DirectSequentialDataLoader {
    file_paths: Vec<String>,
    interleave_batches: usize,
    shuffle_each_epoch: bool,
    shuffle_seed: u64,
}

impl DirectSequentialDataLoader {
    pub fn new(file_paths: &[&str]) -> Self {
        let file_paths = file_paths.iter().map(|path| path.to_string()).collect::<Vec<_>>();

        for path in &file_paths {
            let path_buf: PathBuf = path.parse().unwrap();
            assert!(path_buf.exists(), "File not found: {path}");
        }

        Self {
            file_paths,
            // Default behaviour: read each file end-to-end before moving to the next.
            interleave_batches: usize::MAX,
            shuffle_each_epoch: false,
            shuffle_seed: 0,
        }
    }

    /// Number of batches to read from one file before switching to the next file.
    ///
    /// `1` means strict round-robin by batch, large values approach sequential loading.
    pub fn with_interleave_batches(mut self, interleave_batches: usize) -> Self {
        assert!(interleave_batches > 0, "interleave_batches must be >= 1");
        self.interleave_batches = interleave_batches;
        self
    }

    /// Shuffle file order every epoch (deterministically from `seed` + epoch index).
    pub fn with_epoch_file_shuffle(mut self, enabled: bool, seed: u64) -> Self {
        self.shuffle_each_epoch = enabled;
        self.shuffle_seed = seed;
        self
    }

    pub fn map_file_sizes<F: FnMut(&str, u64)>(&self, mut f: F) {
        for file in self.file_paths.iter() {
            f(file, std::fs::metadata(file).unwrap().len());
        }
    }
}

#[derive(Clone)]
struct FileInfo {
    path: String,
    records: usize,
    batches: usize,
}

#[derive(Clone, Copy, Default)]
struct FileCursor {
    records_read: usize,
    batches_read: usize,
}

#[derive(Clone, Copy, Default)]
struct ReadChunkOutcome {
    records: usize,
    batches: usize,
    stop: bool,
}

#[derive(Clone)]
struct StartSkipState {
    cursors: Vec<FileCursor>,
    order_cursor: usize,
    first_chunk_cap: Option<(usize, usize)>,
}

fn batch_to_record_offset(
    consumed_batches: usize,
    total_batches: usize,
    total_records: usize,
    batch_size: usize,
) -> usize {
    if consumed_batches >= total_batches {
        total_records
    } else {
        consumed_batches.saturating_mul(batch_size).min(total_records)
    }
}

fn epoch_seed(base_seed: u64, epoch_idx: usize) -> u64 {
    let mut x = base_seed ^ (epoch_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0xA076_1D64_78BD_642F;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn fisher_yates_shuffle(data: &mut [usize], seed: u64) {
    let mut state = seed;
    for i in (1..data.len()).rev() {
        let j = (xorshift64(&mut state) as usize) % (i + 1);
        data.swap(i, j);
    }
}

fn build_epoch_file_order(num_files: usize, shuffle_each_epoch: bool, seed: u64, epoch_idx: usize) -> Vec<usize> {
    let mut order = (0..num_files).collect::<Vec<_>>();
    if shuffle_each_epoch && num_files > 1 {
        fisher_yates_shuffle(&mut order, epoch_seed(seed, epoch_idx));
    }
    order
}

fn next_available_slot(
    order: &[usize],
    infos: &[FileInfo],
    cursors: &[FileCursor],
    cursor: usize,
) -> Option<(usize, usize)> {
    if order.is_empty() {
        return None;
    }

    for delta in 0..order.len() {
        let slot = (cursor + delta) % order.len();
        let file_idx = order[slot];
        if cursors[file_idx].batches_read < infos[file_idx].batches {
            return Some((slot, file_idx));
        }
    }

    None
}

fn apply_start_skip(
    infos: &[FileInfo],
    order: &[usize],
    start_point_batches: usize,
    interleave_batches: usize,
) -> StartSkipState {
    let mut cursors = vec![FileCursor::default(); infos.len()];
    if order.is_empty() || start_point_batches == 0 {
        return StartSkipState { cursors, order_cursor: 0, first_chunk_cap: None };
    }

    let mut to_skip = start_point_batches;
    let mut cursor = 0usize;
    let mut first_chunk_cap = None;

    while to_skip > 0 {
        let (slot, file_idx) =
            next_available_slot(order, infos, &cursors, cursor).expect("invalid start point while applying skip");
        let remaining_batches = infos[file_idx].batches - cursors[file_idx].batches_read;
        let slot_chunk = remaining_batches.min(interleave_batches);

        if to_skip < slot_chunk {
            // start point lands inside this slot's chunk.
            cursors[file_idx].batches_read += to_skip;
            first_chunk_cap = Some((file_idx, slot_chunk - to_skip));
            cursor = slot;
            to_skip = 0;
        } else {
            // consumed this slot's chunk completely; move to next slot.
            cursors[file_idx].batches_read += slot_chunk;
            to_skip -= slot_chunk;
            cursor = (slot + 1) % order.len();
        }
    }

    StartSkipState { cursors, order_cursor: cursor, first_chunk_cap }
}

fn read_records_limited<T: CanBeDirectlySequentiallyLoaded, F: FnMut(&[T]) -> bool>(
    file: &mut File,
    buf: &mut [T],
    batch_size: usize,
    max_records: usize,
    f: &mut F,
) -> ReadChunkOutcome {
    if max_records == 0 {
        return ReadChunkOutcome::default();
    }

    let mut emitted_records = 0usize;
    let mut emitted_batches = 0usize;
    let mut remaining = max_records;

    while remaining > 0 {
        let to_read_records = remaining.min(buf.len());
        let to_read_bytes = to_read_records * size_of::<T>();
        let byte_buf = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), to_read_bytes) };

        let mut filled = 0usize;
        while filled < to_read_bytes {
            let count = file.read(&mut byte_buf[filled..]).unwrap_or(0);
            if count == 0 {
                break;
            }
            filled += count;
        }

        if filled == 0 {
            break;
        }

        assert_eq!(filled % size_of::<T>(), 0);
        let len = filled / size_of::<T>();
        emitted_records += len;
        remaining = remaining.saturating_sub(len);

        for batch in buf[..len].chunks(batch_size) {
            emitted_batches += 1;
            if f(batch) {
                return ReadChunkOutcome { records: emitted_records, batches: emitted_batches, stop: true };
            }
        }

        if len < to_read_records {
            break;
        }
    }

    ReadChunkOutcome { records: emitted_records, batches: emitted_batches, stop: false }
}

impl<T: CanBeDirectlySequentiallyLoaded> DataLoader<T> for DirectSequentialDataLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        let data_size = std::mem::size_of::<T>() as u64;

        let mut file_size = 0;

        self.map_file_sizes(|file, this_size| {
            if this_size % data_size != 0 {
                panic!("File [{file}] does not have a multiple of {data_size} size!");
            }

            file_size += this_size;
        });

        Some(file_size / data_size)
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, start_batch: usize, batch_size: usize, mut f: F) {
        assert!(batch_size > 0, "batch_size must be > 0");
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size_bytes = size_of::<T>();
        let batches_per_load = (buffer_size / data_size_bytes / batch_size).max(1);
        let cap = batch_size * batches_per_load;
        let data_size_u64 = data_size_bytes as u64;

        let infos: Vec<FileInfo> = self
            .file_paths
            .iter()
            .map(|file| {
                let this_size = std::fs::metadata(file).unwrap().len();
                if this_size % data_size_u64 != 0 {
                    panic!("File [{file}] does not have a multiple of {data_size_u64} size!");
                }

                let records_u64 = this_size / data_size_u64;
                let records = usize::try_from(records_u64).expect("file is too large to index on this platform");
                let batches_u64 = records_u64.div_ceil(batch_size as u64);
                let batches = usize::try_from(batches_u64).expect("batch count is too large to index on this platform");

                FileInfo { path: file.clone(), records, batches }
            })
            .collect();

        let batches_per_epoch: usize = infos.iter().map(|info| info.batches).sum();
        assert!(batches_per_epoch > 0, "No readable records in data files.");

        let mut epoch_idx = start_batch / batches_per_epoch;
        let mut epoch_start_point = start_batch % batches_per_epoch;

        let mut buf = unsafe { zeroed_boxed_slice::<T>(cap.max(batch_size)) };

        'dataloading: loop {
            let order = build_epoch_file_order(infos.len(), self.shuffle_each_epoch, self.shuffle_seed, epoch_idx);
            if self.shuffle_each_epoch {
                let order_text = order.iter().map(|&idx| infos[idx].path.as_str()).collect::<Vec<_>>().join(",");
                println!("DataLoader epoch {epoch_idx}: file order = {order_text}");
            }

            let StartSkipState { mut cursors, mut order_cursor, mut first_chunk_cap } =
                apply_start_skip(&infos, &order, epoch_start_point, self.interleave_batches);

            let mut loader_files = infos.iter().map(|info| File::open(&info.path).unwrap()).collect::<Vec<_>>();

            for (idx, file) in loader_files.iter_mut().enumerate() {
                let records_to_skip = batch_to_record_offset(
                    cursors[idx].batches_read,
                    infos[idx].batches,
                    infos[idx].records,
                    batch_size,
                );
                if records_to_skip > 0 {
                    file.seek(SeekFrom::Start((records_to_skip as u64) * data_size_u64)).unwrap();
                }
                cursors[idx].records_read = records_to_skip;
            }

            epoch_start_point = 0;

            while let Some((slot, file_idx)) = next_available_slot(&order, &infos, &cursors, order_cursor) {
                order_cursor = (slot + 1) % order.len();

                let info = &infos[file_idx];
                let cursor = &mut cursors[file_idx];

                let remaining_batches = info.batches - cursor.batches_read;
                if remaining_batches == 0 {
                    continue;
                }

                let mut chunk_batches = remaining_batches.min(self.interleave_batches);
                if let Some((pending_file_idx, pending_cap)) = first_chunk_cap {
                    if pending_file_idx == file_idx {
                        chunk_batches = chunk_batches.min(pending_cap);
                    }
                    first_chunk_cap = None;
                }
                let remaining_records = info.records - cursor.records_read;
                let max_records = remaining_records.min(chunk_batches.saturating_mul(batch_size));

                let outcome =
                    read_records_limited(&mut loader_files[file_idx], &mut buf, batch_size, max_records, &mut f);
                cursor.records_read += outcome.records;
                cursor.batches_read += outcome.batches;

                if outcome.stop {
                    break 'dataloading;
                }

                if outcome.records == 0 {
                    // EOF/truncated read: mark exhausted to avoid endless loop.
                    cursor.records_read = info.records;
                    cursor.batches_read = info.batches;
                }
            }

            epoch_idx += 1;
        }
    }
}

unsafe fn zeroed_boxed_slice<T: CanBeDirectlySequentiallyLoaded>(cap: usize) -> Box<[T]> {
    let mut buf = Box::<[T]>::new_uninit_slice(cap);

    // safe as `T` can be any bit pattern, including 0s
    let zeroed = unsafe {
        let mut t: MaybeUninit<T> = MaybeUninit::uninit();
        let tslice: &mut [MaybeUninit<u8>] = slice::from_raw_parts_mut(t.as_mut_ptr().cast(), size_of::<T>());

        for elem in tslice {
            elem.write(0);
        }

        t.assume_init()
    };

    for elem in buf.iter_mut() {
        elem.write(zeroed);
    }

    unsafe { buf.assume_init() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_info(batches: usize) -> FileInfo {
        FileInfo { path: "dummy".to_string(), records: batches * 16, batches }
    }

    #[test]
    fn start_skip_sequential_inside_file_keeps_same_slot() {
        let infos = vec![file_info(10), file_info(10)];
        let order = vec![0usize, 1usize];
        let state = apply_start_skip(&infos, &order, 5, usize::MAX);

        assert_eq!(state.cursors[0].batches_read, 5);
        assert_eq!(state.cursors[1].batches_read, 0);
        assert_eq!(state.order_cursor, 0);
        assert_eq!(state.first_chunk_cap, Some((0, 5)));
    }

    #[test]
    fn start_skip_sequential_boundary_moves_to_next_slot() {
        let infos = vec![file_info(10), file_info(10)];
        let order = vec![0usize, 1usize];
        let state = apply_start_skip(&infos, &order, 10, usize::MAX);

        assert_eq!(state.cursors[0].batches_read, 10);
        assert_eq!(state.cursors[1].batches_read, 0);
        assert_eq!(state.order_cursor, 1);
        assert_eq!(state.first_chunk_cap, None);
    }

    #[test]
    fn start_skip_interleaved_inside_slot_chunk() {
        let infos = vec![file_info(10), file_info(10)];
        let order = vec![0usize, 1usize];
        let state = apply_start_skip(&infos, &order, 2, 4);

        assert_eq!(state.cursors[0].batches_read, 2);
        assert_eq!(state.cursors[1].batches_read, 0);
        assert_eq!(state.order_cursor, 0);
        assert_eq!(state.first_chunk_cap, Some((0, 2)));
    }
}
