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

        Self { file_paths, shuffle_each_epoch: false, shuffle_seed: 0 }
    }

    pub fn map_file_sizes<F: FnMut(&str, u64)>(&self, mut f: F) {
        for file in self.file_paths.iter() {
            f(file, std::fs::metadata(file).unwrap().len());
        }
    }

    /// Inter-file round-robin batch interleaving (現状 no-op)。
    ///
    /// `DataLoader::map_chunks` は batch_size を引数に取らないため、ここで batch
    /// 単位 interleave を表現する手段が無い。引数が default `usize::MAX` (= 切替
    /// 無し) 以外で渡された場合は stderr に WARNING を出し、要求が静かに失われない
    /// ようにしている。
    pub fn with_interleave_batches(self, interleave_batches: usize) -> Self {
        if interleave_batches != usize::MAX {
            eprintln!(
                "[DirectSequentialDataLoader] WARNING: with_interleave_batches({interleave_batches}) is currently a no-op; \
                 files will be read sequentially end-to-end."
            );
        }
        self
    }

    /// 各エポックでファイルの読み込み順を `seed + epoch_idx` から決定論的に
    /// シャッフルする。`enabled = false` または `file_paths.len() <= 1` では
    /// no-op。シャッフルは Fisher-Yates、内部 PRNG は xorshift64。
    pub fn with_epoch_file_shuffle(mut self, enabled: bool, seed: u64) -> Self {
        self.shuffle_each_epoch = enabled;
        self.shuffle_seed = seed;
        self
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

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, mut f: F) {
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size = std::mem::size_of::<T>() as u64;
        let cap = buffer_size / data_size as usize;

        // 1 epoch = 全 file の records 合計
        let mut positions_per_epoch: u64 = 0;
        self.map_file_sizes(|_, this_size| positions_per_epoch += this_size / data_size);
        let positions_per_epoch = positions_per_epoch as usize;
        assert!(positions_per_epoch > 0, "No readable records in data files.");

        // start_position から (epoch_idx, in-epoch offset) を導出
        let mut epoch_idx = start_position / positions_per_epoch;
        let mut start_in_epoch = start_position % positions_per_epoch;

        let mut buf = unsafe { zeroed_boxed_slice::<T>(cap) };

        'dataloading: loop {
            let order =
                build_epoch_file_order(self.file_paths.len(), self.shuffle_each_epoch, self.shuffle_seed, epoch_idx);

            if self.shuffle_each_epoch {
                let order_text = order.iter().map(|&idx| self.file_paths[idx].as_str()).collect::<Vec<_>>().join(",");
                eprintln!("[DirectSequentialDataLoader] epoch {epoch_idx}: file order = {order_text}");
            }

            // 当該 epoch 内で start_in_epoch records をスキップしてから順次読む。
            // 次 epoch 以降の skip 量は 0 に reset。
            let mut to_skip = start_in_epoch;
            start_in_epoch = 0;

            for &file_idx in &order {
                let file_path = &self.file_paths[file_idx];
                let this_size = std::fs::metadata(file_path).unwrap().len();
                let this_positions = (this_size / data_size) as usize;

                if to_skip >= this_positions {
                    to_skip -= this_positions;
                    continue;
                }

                let mut loader_file = File::open(file_path).unwrap();
                if to_skip > 0 {
                    loader_file.seek(SeekFrom::Current((to_skip * data_size as usize) as i64)).unwrap();
                    to_skip = 0;
                }

                loop {
                    let count = loader_file
                        .read(unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), cap * size_of::<T>()) })
                        .unwrap_or(0);

                    if count == 0 {
                        break;
                    }

                    assert_eq!(count % size_of::<T>(), 0);
                    let len = count / size_of::<T>();

                    if f(&buf[..len]) {
                        break 'dataloading;
                    }
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
