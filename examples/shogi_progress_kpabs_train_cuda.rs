/*
KP-absolute progress trainer (CUDA, mini-batched).

Differences from `shogi_progress_kpabs_train`:
  - Mini-batches K games per Adam step (`--games-per-step`, default 1024)
    instead of "1 game = 1 step". Convex problem so the optimum is the same;
    convergence trajectory differs.
  - GPU forward / gradient scatter / Adam step (bullet-gpu raw FFI + NVRTC).
  - `--lr-scale {none, sqrt}` to compensate for batch averaging.
  - `--init-from <progress.bin>` to warm-start from an existing weight file
    (for resume-style runs).

Output is the same YaneuraOu-compatible 1,003,104-byte progress.bin.
Only the game-relative training mode is implemented.
*/

use std::{
    collections::{BTreeMap, VecDeque},
    ffi::c_void,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    mem::size_of,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use bullet_compiler::tensor::{DType, TValue};
use bullet_gpu::{
    buffer::Buffer,
    runtime::{
        Device, Dim3, Kernel, Module, Stream,
        cuda::{Cuda, CudaError},
    },
};
use bullet_lib::{
    game::outputs::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs},
    shogi::PackedSfenValue,
};
use clap::Parser;

/// Newtype around `bullet_gpu::runtime::cuda::CudaError` so it can flow
/// through `Box<dyn std::error::Error>` via `?`. The upstream type does not
/// implement `std::error::Error` (it's marker-typed for the GPU trait), so
/// we wrap it locally and route through this newtype with `cu(...)?`.
#[derive(Debug)]
struct CudaErr(CudaError);

impl std::fmt::Display for CudaErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::error::Error for CudaErr {}

/// Lift a `Result<_, CudaError>` into the boxed-error world used by `main`.
fn cu<T>(r: Result<T, CudaError>) -> Result<T, CudaErr> {
    r.map_err(CudaErr)
}

const PACK_RECORD_BYTES: usize = size_of::<PackedSfenValue>();
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Maximum active KP-absolute indices per position.
/// Shogi has at most 38 non-king pieces (board + hand) and each contributes
/// 2 indices (sq_bk and sq_wk variants), giving an upper bound of 76. We pad
/// to 80 for headroom; unused slots are filled with -1 sentinel.
const MAX_INDS_PER_POS: usize = 80;

#[derive(Parser, Debug)]
#[command(name = "shogi_progress_kpabs_train_cuda")]
#[command(about = "Train KP-absolute progress.bin on GPU with K-games-per-step mini-batching")]
struct Args {
    /// Comma-separated files or directories. Directories contribute only top-level *.bin files.
    #[arg(long)]
    data: String,

    /// Output progress.bin path
    #[arg(long)]
    output: PathBuf,

    /// Warm-start weights from an existing progress.bin
    #[arg(long)]
    init_from: Option<PathBuf>,

    /// Mini-batch size in games per Adam step
    #[arg(long, default_value_t = 1024)]
    games_per_step: usize,

    /// Maximum number of games for training (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_games: usize,

    /// Number of validation games (0 = auto 5% of files)
    #[arg(long, default_value_t = 0)]
    val_games: usize,

    /// Validation file ratio when --val-games is 0
    #[arg(long, default_value_t = 0.05)]
    val_files_ratio: f32,

    /// Number of passes over the training split
    #[arg(long, default_value_t = 1)]
    epochs: usize,

    /// Base learning rate (before scaling)
    #[arg(long, default_value_t = 1e-3)]
    lr: f32,

    /// Learning rate scaling for batch size: none keeps lr, sqrt multiplies by sqrt(K)
    #[arg(long, default_value = "sqrt")]
    lr_scale: LrScaleMode,

    /// Progress report interval in steps (= Adam updates)
    #[arg(long, default_value_t = 100)]
    log_interval_steps: usize,

    /// Save a per-epoch checkpoint as `<output_stem>.e{N}.<ext>` after each epoch
    #[arg(long)]
    save_each_epoch: bool,

    /// CUDA device ordinal
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Number of CPU threads decoding PSV records and building batches in parallel
    #[arg(long, default_value_t = 4)]
    reader_threads: usize,

    /// Maximum number of pre-built batches buffered ahead of the GPU
    #[arg(long, default_value_t = 4)]
    prefetch_depth: usize,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum LrScaleMode {
    None,
    Sqrt,
}

#[derive(Debug, Clone)]
struct PackInfo {
    path: PathBuf,
    records: u64,
}

struct PackCursor {
    reader: BufReader<File>,
    remaining_records: u64,
}

impl PackCursor {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let records = file.metadata()?.len() / PACK_RECORD_BYTES as u64;
        Ok(Self { reader: BufReader::new(file), remaining_records: records })
    }

    fn next_psv(&mut self) -> io::Result<Option<PackedSfenValue>> {
        if self.remaining_records == 0 {
            return Ok(None);
        }
        let mut psv = PackedSfenValue::default();
        match self.reader.read_exact(psv.as_bytes_mut()) {
            Ok(()) => {
                self.remaining_records -= 1;
                Ok(Some(psv))
            }
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
                self.remaining_records = 0;
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }
}

struct GameIterator {
    cursor: PackCursor,
    buffer: Vec<PackedSfenValue>,
    prev_ply: Option<u16>,
    done: bool,
}

impl GameIterator {
    fn new(cursor: PackCursor) -> Self {
        Self { cursor, buffer: Vec::new(), prev_ply: None, done: false }
    }

    fn next_game(&mut self) -> io::Result<Option<Vec<PackedSfenValue>>> {
        if self.done {
            return Ok(None);
        }
        loop {
            match self.cursor.next_psv()? {
                Some(psv) => {
                    let ply = psv.game_ply();
                    let is_boundary = self.prev_ply.is_some_and(|prev| ply <= prev);
                    self.prev_ply = Some(ply);
                    if is_boundary && !self.buffer.is_empty() {
                        let game = std::mem::take(&mut self.buffer);
                        self.buffer.push(psv);
                        return Ok(Some(game));
                    }
                    self.buffer.push(psv);
                }
                None => {
                    self.done = true;
                    if !self.buffer.is_empty() {
                        return Ok(Some(std::mem::take(&mut self.buffer)));
                    }
                    return Ok(None);
                }
            }
        }
    }
}

fn collect_pack_infos(spec: &str) -> io::Result<Vec<PackInfo>> {
    let mut paths = Vec::new();
    for raw in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let path = PathBuf::from(raw);
        let meta = fs::metadata(&path).map_err(|err| {
            io::Error::new(err.kind(), format!("failed to read metadata for '{}': {err}", path.display()))
        })?;
        if meta.is_file() {
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if ext == "bin" || ext == "pack" {
                paths.push(path);
            }
            continue;
        }
        if meta.is_dir() {
            let mut dir_paths = Vec::new();
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let p = entry.path();
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
                if ext == "bin" || ext == "pack" {
                    dir_paths.push(p);
                }
            }
            dir_paths.sort();
            paths.extend(dir_paths);
        }
    }
    paths.sort();
    paths.dedup();

    let mut packs = Vec::with_capacity(paths.len());
    for p in paths {
        let records = fs::metadata(&p)?.len() / PACK_RECORD_BYTES as u64;
        if records == 0 {
            continue;
        }
        packs.push(PackInfo { path: p, records });
    }
    if packs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no valid *.bin/*.pack files found"));
    }
    Ok(interleave_pack_groups(packs))
}

fn pack_group_key(path: &Path) -> String {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    if name.starts_with("hao_depth_9_shuffled_") {
        return "hao_depth_9_shuffled".to_string();
    }
    if name.starts_with("shuffled_") {
        return "shuffled".to_string();
    }
    path.file_stem().and_then(|s| s.to_str()).map_or_else(|| "unknown".to_string(), |s| s.to_string())
}

fn interleave_pack_groups(packs: Vec<PackInfo>) -> Vec<PackInfo> {
    let total = packs.len();
    let mut groups: BTreeMap<String, VecDeque<PackInfo>> = BTreeMap::new();
    for pack in packs {
        groups.entry(pack_group_key(&pack.path)).or_default().push_back(pack);
    }
    let mut out = Vec::with_capacity(total);
    while out.len() < total {
        let mut progressed = false;
        for queue in groups.values_mut() {
            if let Some(pack) = queue.pop_front() {
                out.push(pack);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    out
}

fn write_progress_bin(path: &Path, weights: &[f32]) -> io::Result<()> {
    if weights.len() != SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "weight length mismatch"));
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &w in weights {
        writer.write_all(&(w as f64).to_le_bytes())?;
    }
    writer.flush()
}

fn read_progress_bin(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("progress.bin size mismatch: got {}, expected {}", bytes.len(), expected),
        ));
    }
    Ok(bytes.chunks_exact(size_of::<f64>()).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect())
}

fn epoch_checkpoint_path(output: &Path, epoch: usize) -> PathBuf {
    let stem = output.file_stem().and_then(|s| s.to_str()).unwrap_or("progress");
    let ext = output.extension().and_then(|s| s.to_str()).unwrap_or("bin");
    output.with_file_name(format!("{stem}.e{epoch}.{ext}"))
}

fn top_bucket_info(hist: &[u64; 8]) -> (usize, f64) {
    let total: u128 = hist.iter().map(|&c| c as u128).sum();
    if total == 0 {
        return (0, 0.0);
    }
    let (idx, &count) = hist.iter().enumerate().max_by_key(|(_, c)| **c).unwrap();
    (idx, count as f64 / total as f64)
}

// ---------- CUDA kernels ----------

const KERNELS_SRC: &str = r#"
extern "C" __global__ void k_forward(
    const int*  __restrict__ indices,    // [n_pos * MAX_INDS]
    const float* __restrict__ weights,   // [num_weights]
    float* __restrict__ preds,           // [n_pos]
    const int n_pos,
    const int max_inds)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_pos) return;
    float z = 0.0f;
    const int base = pos * max_inds;
    for (int j = 0; j < max_inds; ++j) {
        int idx = indices[base + j];
        if (idx >= 0) z += weights[idx];
    }
    preds[pos] = 1.0f / (1.0f + expf(-z));
}

extern "C" __global__ void k_grad_loss_hist(
    const int*  __restrict__ indices,
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    const float* __restrict__ per_pos_norm,
    float* __restrict__ grad,
    double* __restrict__ loss_acc,                // f64 to avoid precision loss when summing many positions
    unsigned long long* __restrict__ hist,        // u64 to avoid overflow on full-epoch counts
    const int n_pos,
    const int max_inds)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_pos) return;
    float p = preds[pos];
    float y = targets[pos];
    float err = p - y;
    float norm = per_pos_norm[pos];
    float gscale = 2.0f * err * p * (1.0f - p) * norm;

    const int base = pos * max_inds;
    for (int j = 0; j < max_inds; ++j) {
        int idx = indices[base + j];
        if (idx >= 0) atomicAdd(&grad[idx], gscale);
    }

    atomicAdd(loss_acc, (double)err * (double)err);

    int b = (int)(p * 8.0f);
    if (b < 0) b = 0;
    if (b > 7) b = 7;
    atomicAdd(&hist[b], 1ULL);
}

extern "C" __global__ void k_adam_step(
    float* __restrict__ weights,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bc1,                     // 1 - beta1^t
    const float bc2,                     // 1 - beta2^t
    const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / fmaxf(bc1, 1e-30f);
    float v_hat = vi / fmaxf(bc2, 1e-30f);
    weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    grad[i] = 0.0f;
}

extern "C" __global__ void k_eval_loss_hist(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    double* __restrict__ loss_acc,
    unsigned long long* __restrict__ hist,
    const int n_pos)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_pos) return;
    float p = preds[pos];
    float y = targets[pos];
    float err = p - y;
    atomicAdd(loss_acc, (double)err * (double)err);
    int b = (int)(p * 8.0f);
    if (b < 0) b = 0;
    if (b > 7) b = 7;
    atomicAdd(&hist[b], 1ULL);
}
"#;

/// Compute grid dim for a 1-D launch covering `n` items with `threads` threads per block.
fn grid_dim_for(n: usize, threads: u32) -> Dim3 {
    let blocks = ((n as u32).max(1)).div_ceil(threads);
    Dim3 { x: blocks, y: 1, z: 1 }
}

// ---------- Host-side batch builder ----------

struct Batch {
    /// Flat indices, [total_positions * MAX_INDS_PER_POS], -1 for padding
    indices: Vec<i32>,
    /// Per-position target y = i / (game_len - 1), in [0,1]
    targets: Vec<f32>,
    /// Per-position normalization: 1 / (game_len * num_games_in_batch)
    per_pos_norm: Vec<f32>,
    n_positions: usize,
    n_games: usize,
}

impl Batch {
    fn new() -> Self {
        Self { indices: Vec::new(), targets: Vec::new(), per_pos_norm: Vec::new(), n_positions: 0, n_games: 0 }
    }

    fn push_game(&mut self, game: &[PackedSfenValue], scratch: &mut Vec<usize>) {
        let game_len = game.len();
        if game_len == 0 {
            return;
        }
        self.n_games += 1;
        for (i, psv) in game.iter().enumerate() {
            let y = if game_len == 1 { 0.0f32 } else { i as f32 / (game_len - 1) as f32 };
            ShogiProgressKPAbs::collect_active_indices(psv, scratch);
            let mut row = [-1i32; MAX_INDS_PER_POS];
            for (j, &idx) in scratch.iter().take(MAX_INDS_PER_POS).enumerate() {
                row[j] = idx as i32;
            }
            self.indices.extend_from_slice(&row);
            self.targets.push(y);
            // per-pos norm filled in finalize() once we know n_games
            self.per_pos_norm.push(1.0 / game_len as f32);
            self.n_positions += 1;
        }
    }

    /// Multiply per_pos_norm by 1/n_games to finish averaging.
    fn finalize(&mut self) {
        let inv_k = 1.0 / self.n_games.max(1) as f32;
        for n in &mut self.per_pos_norm {
            *n *= inv_k;
        }
    }
}

// ---------- GPU trainer ----------

/// Owned device allocation for an arbitrary dtype not supported by `Buffer`
/// (we need `f64` for loss accumulation and `u64` for the prediction histogram,
/// both of which use atomic operations on raw device pointers, but bullet-gpu's
/// `Buffer<G>` only models F32/I32). Allocation goes through `Device::malloc`
/// directly and the pointer is freed in `Drop`.
struct RawBuf {
    device: Arc<Device<Cuda>>,
    ptr: <Cuda as bullet_gpu::runtime::Gpu>::DevicePtr,
    bytes: usize,
}

impl RawBuf {
    fn zeroed(device: &Arc<Device<Cuda>>, bytes: usize) -> Result<Self, CudaErr> {
        let ptr = cu(device.malloc(bytes))?;
        unsafe {
            cu(device.memset(ptr, bytes, 0))?;
        }
        Ok(Self { device: device.clone(), ptr, bytes })
    }

    fn ptr(&self) -> <Cuda as bullet_gpu::runtime::Gpu>::DevicePtr {
        self.ptr
    }
}

impl Drop for RawBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.free(self.ptr);
        }
    }
}

struct GpuTrainer {
    device: Arc<Device<Cuda>>,
    stream: Arc<Stream<Cuda>>,
    // The module is kept alive via the kernels (each `Kernel` holds `Arc<Module>`).
    f_forward: Kernel<Cuda>,
    f_grad: Kernel<Cuda>,
    f_adam: Kernel<Cuda>,
    f_eval: Kernel<Cuda>,
    weights: Arc<Buffer<Cuda>>,
    m: Arc<Buffer<Cuda>>,
    v: Arc<Buffer<Cuda>>,
    grad: Arc<Buffer<Cuda>>,
    loss_acc: RawBuf,
    hist: RawBuf,
    /// pre-allocated scratch buffers (resized on demand)
    indices_dev: Option<Arc<Buffer<Cuda>>>,
    targets_dev: Option<Arc<Buffer<Cuda>>>,
    norm_dev: Option<Arc<Buffer<Cuda>>>,
    preds_dev: Option<Arc<Buffer<Cuda>>>,
    /// Adam state
    beta1_pow: f32,
    beta2_pow: f32,
}

impl GpuTrainer {
    fn new(device_id: usize, init_weights: Option<&[f32]>) -> Result<Self, CudaErr> {
        let device = cu(Device::<Cuda>::new(device_id as i32))?;
        let stream = cu(device.new_stream())?;

        // `Module::new` runs NVRTC compile + module load in one shot. The arch
        // option is automatically derived from `device.props().arch()` (e.g.
        // `--gpu-architecture=sm_86`), which covers the sm_60+ requirement of
        // the `atomicAdd(double*)` we use for loss accumulation.
        // Note: this targets the running device's exact SM. On environments
        // where the installed NVRTC is older than the running GPU's SM, NVRTC
        // can reject the unknown sm_xx target. If that happens, the fallback
        // is a forward-compatible virtual arch (e.g. `compute_60` PTX) which
        // currently requires a custom Module builder bypassing `Module::new`.
        let module = cu(Module::new(device.clone(), KERNELS_SRC))?;

        let f_forward = cu(module.clone().get_kernel("k_forward"))?;
        let f_grad = cu(module.clone().get_kernel("k_grad_loss_hist"))?;
        let f_adam = cu(module.clone().get_kernel("k_adam_step"))?;
        let f_eval = cu(module.get_kernel("k_eval_loss_hist"))?;

        let n = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
        let weights = if let Some(init) = init_weights {
            assert_eq!(init.len(), n);
            cu(Buffer::from_host(&device, &TValue::F32(init.to_vec())))?
        } else {
            cu(Buffer::zeroed(&device, DType::F32, n))?
        };
        let m = cu(Buffer::zeroed(&device, DType::F32, n))?;
        let v = cu(Buffer::zeroed(&device, DType::F32, n))?;
        let grad = cu(Buffer::zeroed(&device, DType::F32, n))?;
        let loss_acc = RawBuf::zeroed(&device, std::mem::size_of::<f64>())?;
        let hist = RawBuf::zeroed(&device, 8 * std::mem::size_of::<u64>())?;

        Ok(Self {
            device,
            stream,
            f_forward,
            f_grad,
            f_adam,
            f_eval,
            weights,
            m,
            v,
            grad,
            loss_acc,
            hist,
            indices_dev: None,
            targets_dev: None,
            norm_dev: None,
            preds_dev: None,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        })
    }

    fn zero_loss_hist(&mut self) -> Result<(), CudaErr> {
        // memset to 0 is valid for both f64 0.0 and u64 0.
        unsafe {
            cu(self.device.memset(self.loss_acc.ptr(), self.loss_acc.bytes, 0))?;
            cu(self.device.memset(self.hist.ptr(), self.hist.bytes, 0))?;
        }
        Ok(())
    }

    fn ensure_scratch(&mut self, n_pos: usize) -> Result<(), CudaErr> {
        let need_idx = n_pos * MAX_INDS_PER_POS;
        if self.indices_dev.as_ref().is_none_or(|b| b.size() < need_idx) {
            self.indices_dev = Some(cu(Buffer::zeroed(&self.device, DType::I32, need_idx))?);
        }
        if self.targets_dev.as_ref().is_none_or(|b| b.size() < n_pos) {
            self.targets_dev = Some(cu(Buffer::zeroed(&self.device, DType::F32, n_pos))?);
            self.norm_dev = Some(cu(Buffer::zeroed(&self.device, DType::F32, n_pos))?);
            self.preds_dev = Some(cu(Buffer::zeroed(&self.device, DType::F32, n_pos))?);
        }
        Ok(())
    }

    /// Upload a host slice into the prefix of a device scratch buffer via
    /// `Stream::memcpy_h2d`. The safe `Buffer::copy_from_host_async` would
    /// require the host buffer to exactly match the device buffer size, but
    /// the scratch buffers are sized to the largest batch ever seen and we
    /// only write the active prefix.
    ///
    /// SAFETY: the destination buffer's allocation must be at least
    /// `size_of_val(src)` bytes; the host slice must remain valid until the
    /// stream is synced. Both are guaranteed by `ensure_scratch` and the
    /// surrounding `step`/`eval_forward` flow which syncs before returning.
    unsafe fn upload_prefix<T>(&self, dst: &Arc<Buffer<Cuda>>, src: &[T]) -> Result<(), CudaErr> {
        let bytes = std::mem::size_of_val(src);
        assert!(
            bytes <= dst.bytes(),
            "upload_prefix: host slice ({bytes} bytes) exceeds device buffer ({} bytes)",
            dst.bytes()
        );
        let guard = cu(dst.acquire(self.stream.clone()))?;
        unsafe {
            cu(self.stream.memcpy_h2d(src.as_ptr().cast::<c_void>(), guard.ptr(), bytes))?;
        }
        // Drop the guard; the actual copy has been queued. We rely on the
        // caller syncing the stream before re-using the host slice.
        drop(guard);
        Ok(())
    }

    fn step(&mut self, batch: &Batch, lr: f32) -> Result<(), CudaErr> {
        let n_pos = batch.n_positions;
        if n_pos == 0 {
            return Ok(());
        }
        self.ensure_scratch(n_pos)?;

        let indices_buf = self.indices_dev.as_ref().unwrap().clone();
        let targets_buf = self.targets_dev.as_ref().unwrap().clone();
        let norm_buf = self.norm_dev.as_ref().unwrap().clone();
        let preds_buf = self.preds_dev.as_ref().unwrap().clone();

        // Stream-async upload of the active prefixes into the persistent
        // scratch buffers. The sync at the end of `step` waits for these.
        unsafe {
            self.upload_prefix(&indices_buf, &batch.indices[..n_pos * MAX_INDS_PER_POS])?;
            self.upload_prefix(&targets_buf, &batch.targets[..n_pos])?;
            self.upload_prefix(&norm_buf, &batch.per_pos_norm[..n_pos])?;
        }

        let g_idx = cu(indices_buf.acquire(self.stream.clone()))?;
        let g_tgt = cu(targets_buf.acquire(self.stream.clone()))?;
        let g_nrm = cu(norm_buf.acquire(self.stream.clone()))?;
        let g_pred = cu(preds_buf.acquire(self.stream.clone()))?;
        let g_w = cu(self.weights.acquire(self.stream.clone()))?;
        let g_grad = cu(self.grad.acquire(self.stream.clone()))?;
        let g_m = cu(self.m.acquire(self.stream.clone()))?;
        let g_v = cu(self.v.acquire(self.stream.clone()))?;

        let n_pos_i32 = n_pos as i32;
        let max_inds_i32 = MAX_INDS_PER_POS as i32;
        let grid_pos = grid_dim_for(n_pos, 256);

        // Forward: k_forward(indices, weights, preds, n_pos, max_inds)
        unsafe {
            let p_idx = g_idx.ptr();
            let p_w = g_w.ptr();
            let p_pred = g_pred.ptr();
            let mut args: [*mut c_void; 5] = [
                (&p_idx as *const _ as *mut c_void),
                (&p_w as *const _ as *mut c_void),
                (&p_pred as *const _ as *mut c_void),
                (&n_pos_i32 as *const _ as *mut c_void),
                (&max_inds_i32 as *const _ as *mut c_void),
            ];
            cu(self.f_forward.launch(&self.stream, grid_pos, 256, args.as_mut_ptr(), 0))?;
        }

        // Grad + loss + hist accumulator (atomic adds into self.grad / loss_acc / hist).
        unsafe {
            let p_idx = g_idx.ptr();
            let p_pred = g_pred.ptr();
            let p_tgt = g_tgt.ptr();
            let p_nrm = g_nrm.ptr();
            let p_grad = g_grad.ptr();
            let p_loss = self.loss_acc.ptr();
            let p_hist = self.hist.ptr();
            let mut args: [*mut c_void; 9] = [
                (&p_idx as *const _ as *mut c_void),
                (&p_pred as *const _ as *mut c_void),
                (&p_tgt as *const _ as *mut c_void),
                (&p_nrm as *const _ as *mut c_void),
                (&p_grad as *const _ as *mut c_void),
                (&p_loss as *const _ as *mut c_void),
                (&p_hist as *const _ as *mut c_void),
                (&n_pos_i32 as *const _ as *mut c_void),
                (&max_inds_i32 as *const _ as *mut c_void),
            ];
            cu(self.f_grad.launch(&self.stream, grid_pos, 256, args.as_mut_ptr(), 0))?;
        }

        // Adam step
        self.beta1_pow *= ADAM_BETA1;
        self.beta2_pow *= ADAM_BETA2;
        let bc1 = 1.0 - self.beta1_pow;
        let bc2 = 1.0 - self.beta2_pow;
        let beta1 = ADAM_BETA1;
        let beta2 = ADAM_BETA2;
        let eps = ADAM_EPS;
        let n_w_i32 = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS as i32;
        let grid_w = grid_dim_for(SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, 256);

        unsafe {
            let p_w = g_w.ptr();
            let p_m = g_m.ptr();
            let p_v = g_v.ptr();
            let p_grad = g_grad.ptr();
            let mut args: [*mut c_void; 11] = [
                (&p_w as *const _ as *mut c_void),
                (&p_m as *const _ as *mut c_void),
                (&p_v as *const _ as *mut c_void),
                (&p_grad as *const _ as *mut c_void),
                (&lr as *const _ as *mut c_void),
                (&beta1 as *const _ as *mut c_void),
                (&beta2 as *const _ as *mut c_void),
                (&eps as *const _ as *mut c_void),
                (&bc1 as *const _ as *mut c_void),
                (&bc2 as *const _ as *mut c_void),
                (&n_w_i32 as *const _ as *mut c_void),
            ];
            cu(self.f_adam.launch(&self.stream, grid_w, 256, args.as_mut_ptr(), 0))?;
        }

        // Sync to ensure all kernels finish before guards are dropped and
        // before the host-side batch buffer is freed by the caller.
        cu(self.stream.sync())?;
        drop((g_idx, g_tgt, g_nrm, g_pred, g_w, g_grad, g_m, g_v));
        Ok(())
    }

    fn eval_forward(&mut self, batch: &Batch) -> Result<(), CudaErr> {
        let n_pos = batch.n_positions;
        if n_pos == 0 {
            return Ok(());
        }
        self.ensure_scratch(n_pos)?;

        let indices_buf = self.indices_dev.as_ref().unwrap().clone();
        let targets_buf = self.targets_dev.as_ref().unwrap().clone();
        let preds_buf = self.preds_dev.as_ref().unwrap().clone();

        unsafe {
            self.upload_prefix(&indices_buf, &batch.indices[..n_pos * MAX_INDS_PER_POS])?;
            self.upload_prefix(&targets_buf, &batch.targets[..n_pos])?;
        }

        let g_idx = cu(indices_buf.acquire(self.stream.clone()))?;
        let g_tgt = cu(targets_buf.acquire(self.stream.clone()))?;
        let g_pred = cu(preds_buf.acquire(self.stream.clone()))?;
        let g_w = cu(self.weights.acquire(self.stream.clone()))?;

        let n_pos_i32 = n_pos as i32;
        let max_inds_i32 = MAX_INDS_PER_POS as i32;
        let grid_pos = grid_dim_for(n_pos, 256);

        unsafe {
            let p_idx = g_idx.ptr();
            let p_w = g_w.ptr();
            let p_pred = g_pred.ptr();
            let mut args: [*mut c_void; 5] = [
                (&p_idx as *const _ as *mut c_void),
                (&p_w as *const _ as *mut c_void),
                (&p_pred as *const _ as *mut c_void),
                (&n_pos_i32 as *const _ as *mut c_void),
                (&max_inds_i32 as *const _ as *mut c_void),
            ];
            cu(self.f_forward.launch(&self.stream, grid_pos, 256, args.as_mut_ptr(), 0))?;
        }

        unsafe {
            let p_pred = g_pred.ptr();
            let p_tgt = g_tgt.ptr();
            let p_loss = self.loss_acc.ptr();
            let p_hist = self.hist.ptr();
            let mut args: [*mut c_void; 5] = [
                (&p_pred as *const _ as *mut c_void),
                (&p_tgt as *const _ as *mut c_void),
                (&p_loss as *const _ as *mut c_void),
                (&p_hist as *const _ as *mut c_void),
                (&n_pos_i32 as *const _ as *mut c_void),
            ];
            cu(self.f_eval.launch(&self.stream, grid_pos, 256, args.as_mut_ptr(), 0))?;
        }

        cu(self.stream.sync())?;
        drop((g_idx, g_tgt, g_pred, g_w));
        Ok(())
    }

    fn read_loss_hist(&mut self) -> Result<(f64, [u64; 8]), CudaErr> {
        let mut loss = [0.0f64; 1];
        let mut hist = [0u64; 8];
        unsafe {
            cu(self.device.memcpy_d2h(
                self.loss_acc.ptr(),
                loss.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of::<f64>(),
            ))?;
            cu(self.device.memcpy_d2h(
                self.hist.ptr(),
                hist.as_mut_ptr().cast::<c_void>(),
                8 * std::mem::size_of::<u64>(),
            ))?;
        }
        Ok((loss[0], hist))
    }

    fn read_weights(&mut self) -> Result<Vec<f32>, CudaErr> {
        let host = cu(self.weights.to_host())?;
        match host {
            TValue::F32(v) => Ok(v),
            other => Err(CudaErr(CudaError::Message(format!("unexpected weights dtype: {:?}", other.dtype())))),
        }
    }

    fn synchronize(&self) -> Result<(), CudaErr> {
        cu(self.stream.sync())?;
        Ok(())
    }
}

// ---------- training / eval loops ----------

struct EpochStats {
    samples: usize,
    games: usize,
    steps: usize,
    mean_loss: f64,
    bucket_hist: [u64; 8],
}

// ---------- Multi-threaded prefetch producer ----------
//
// Each worker pops one file at a time from a shared queue, decodes its PSV
// records, splits them into games (game boundary = ply decrease), and pushes
// games_per_step-sized batches into an mpsc channel. The main (GPU) thread
// receives batches via `next_batch`. Order across workers is non-deterministic,
// but each batch is internally consistent because every game is produced from
// a single file by a single worker.

struct PrefetchProducer {
    workers: Vec<JoinHandle<io::Result<()>>>,
    rx: mpsc::Receiver<Batch>,
    files_done: Arc<AtomicUsize>,
    file_count: usize,
}

impl PrefetchProducer {
    fn spawn(
        packs: Vec<PackInfo>,
        reader_threads: usize,
        games_per_step: usize,
        prefetch_depth: usize,
        max_games: Option<usize>,
    ) -> Self {
        let file_count = packs.len();
        let queue: Arc<Mutex<VecDeque<PackInfo>>> = Arc::new(Mutex::new(VecDeque::from(packs)));
        let (tx, rx) = mpsc::sync_channel::<Batch>(prefetch_depth.max(1));
        let games_consumed = Arc::new(AtomicUsize::new(0));
        let files_done = Arc::new(AtomicUsize::new(0));

        let worker_count = reader_threads.max(1);
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let queue = Arc::clone(&queue);
            let tx = tx.clone();
            let games_consumed = Arc::clone(&games_consumed);
            let files_done = Arc::clone(&files_done);
            workers.push(thread::spawn(move || -> io::Result<()> {
                let mut scratch: Vec<usize> = Vec::with_capacity(96);
                let mut batch = Batch::new();

                loop {
                    let pack = match queue.lock().unwrap().pop_front() {
                        Some(p) => p,
                        None => break,
                    };
                    let cursor = PackCursor::open(&pack.path)?;
                    let mut gi = GameIterator::new(cursor);

                    while let Some(game) = gi.next_game()? {
                        if game.is_empty() {
                            continue;
                        }

                        if let Some(limit) = max_games {
                            // claim a slot in the global game budget
                            let prev = games_consumed.fetch_add(1, Ordering::Relaxed);
                            if prev >= limit {
                                // budget exhausted; flush any partial batch and stop the worker
                                if batch.n_games > 0 {
                                    batch.finalize();
                                    let _ = tx.send(std::mem::replace(&mut batch, Batch::new()));
                                }
                                files_done.fetch_add(1, Ordering::Relaxed);
                                return Ok(());
                            }
                        }

                        batch.push_game(&game, &mut scratch);
                        if batch.n_games >= games_per_step {
                            batch.finalize();
                            if tx.send(std::mem::replace(&mut batch, Batch::new())).is_err() {
                                files_done.fetch_add(1, Ordering::Relaxed);
                                return Ok(());
                            }
                        }
                    }

                    files_done.fetch_add(1, Ordering::Relaxed);
                }

                // flush trailing partial batch (when the queue empties before reaching games_per_step)
                if batch.n_games > 0 {
                    batch.finalize();
                    let _ = tx.send(batch);
                }
                Ok(())
            }));
        }
        // drop the original sender so the channel closes once all workers exit
        drop(tx);

        Self { workers, rx, files_done, file_count }
    }

    fn next_batch(&self) -> Option<Batch> {
        self.rx.recv().ok()
    }

    fn files_done(&self) -> usize {
        self.files_done.load(Ordering::Relaxed)
    }

    fn file_count(&self) -> usize {
        self.file_count
    }

    fn join(self) -> io::Result<()> {
        for handle in self.workers {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(io::Error::other("reader worker panicked")),
            }
        }
        Ok(())
    }
}

fn train_one_epoch(
    trainer: &mut GpuTrainer,
    packs: &[PackInfo],
    args: &Args,
    lr: f32,
    epoch: usize,
    total_epochs: usize,
) -> Result<EpochStats, Box<dyn std::error::Error>> {
    trainer.zero_loss_hist()?;
    let max_games = if args.max_games > 0 { Some(args.max_games) } else { None };
    let producer = PrefetchProducer::spawn(
        packs.to_vec(),
        args.reader_threads,
        args.games_per_step,
        args.prefetch_depth,
        max_games,
    );

    let mut games_total = 0usize;
    let mut samples_total = 0usize;
    let mut steps = 0usize;
    let start = Instant::now();

    while let Some(batch) = producer.next_batch() {
        if batch.n_games == 0 {
            continue;
        }
        games_total += batch.n_games;
        samples_total += batch.n_positions;
        trainer.step(&batch, lr)?;
        steps += 1;

        if args.log_interval_steps > 0 && steps.is_multiple_of(args.log_interval_steps) {
            trainer.synchronize()?;
            let (loss_sum, _) = trainer.read_loss_hist()?;
            let avg = if samples_total > 0 { loss_sum / samples_total as f64 } else { 0.0 };
            let elapsed = start.elapsed().as_secs_f64();
            let games_per_sec = games_total as f64 / elapsed.max(1e-9);
            println!(
                "epoch {}/{} files_done {}/{} steps {} games {} samples {} avg_loss {:.6} games/s {:.0}",
                epoch,
                total_epochs,
                producer.files_done(),
                producer.file_count(),
                steps,
                games_total,
                samples_total,
                avg,
                games_per_sec
            );
        }
    }

    producer.join()?;
    trainer.synchronize()?;
    let (loss_sum, hist) = trainer.read_loss_hist()?;
    let mean_loss = if samples_total > 0 { loss_sum / samples_total as f64 } else { 0.0 };
    Ok(EpochStats { samples: samples_total, games: games_total, steps, mean_loss, bucket_hist: hist })
}

fn evaluate_split(
    trainer: &mut GpuTrainer,
    packs: &[PackInfo],
    max_games: usize,
    games_per_step: usize,
    reader_threads: usize,
    prefetch_depth: usize,
) -> Result<EpochStats, Box<dyn std::error::Error>> {
    trainer.zero_loss_hist()?;
    let cap = if max_games > 0 { Some(max_games) } else { None };
    let producer = PrefetchProducer::spawn(packs.to_vec(), reader_threads, games_per_step, prefetch_depth, cap);

    let mut games_total = 0usize;
    let mut samples_total = 0usize;
    let mut steps = 0usize;

    while let Some(batch) = producer.next_batch() {
        if batch.n_games == 0 {
            continue;
        }
        games_total += batch.n_games;
        samples_total += batch.n_positions;
        trainer.eval_forward(&batch)?;
        steps += 1;
    }

    producer.join()?;
    trainer.synchronize()?;
    let (loss_sum, hist) = trainer.read_loss_hist()?;
    let mean_loss = if samples_total > 0 { loss_sum / samples_total as f64 } else { 0.0 };
    Ok(EpochStats { samples: samples_total, games: games_total, steps, mean_loss, bucket_hist: hist })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.epochs == 0 {
        return Err("--epochs must be >= 1".into());
    }
    if args.games_per_step == 0 {
        return Err("--games-per-step must be >= 1".into());
    }

    let packs = collect_pack_infos(&args.data)?;
    let total_records: u64 = packs.iter().map(|p| p.records).sum();
    println!("Loaded {} pack files, total available positions: {}", packs.len(), total_records);

    // Train/Val split (file-based)
    let val_count = ((packs.len() as f32) * args.val_files_ratio).round().max(1.0) as usize;
    let val_count = val_count.min(packs.len() / 2).max(1);
    let split = packs.len().saturating_sub(val_count);
    let train_packs = packs[..split].to_vec();
    let val_packs = packs[split..].to_vec();
    println!("split: {} train files, {} val files", train_packs.len(), val_packs.len());

    // LR scaling
    let scaled_lr = match args.lr_scale {
        LrScaleMode::None => args.lr,
        LrScaleMode::Sqrt => args.lr * (args.games_per_step as f32).sqrt(),
    };
    println!(
        "lr base {} scale {:?} -> effective lr {} (games_per_step={})",
        args.lr, args.lr_scale, scaled_lr, args.games_per_step
    );

    // Initial weights
    let init = if let Some(p) = &args.init_from {
        println!("initializing weights from {}", p.display());
        Some(read_progress_bin(p)?)
    } else {
        None
    };

    let mut trainer = GpuTrainer::new(args.device, init.as_deref())
        .map_err(|e| io::Error::other(format!("CUDA init failed: {e}")))?;
    println!("CUDA device {} initialized, kernels compiled", args.device);

    // Baseline val
    let val_max = if args.val_games > 0 { args.val_games } else { 0 };
    let baseline = evaluate_split(
        &mut trainer,
        &val_packs,
        val_max,
        args.games_per_step,
        args.reader_threads,
        args.prefetch_depth,
    )
    .map_err(|e| io::Error::other(format!("baseline eval failed: {e}")))?;
    let (b_top, b_share) = top_bucket_info(&baseline.bucket_hist);
    println!(
        "baseline val_loss {:.6} samples {} games {} top_bucket b{} ({:.2}%)",
        baseline.mean_loss,
        baseline.samples,
        baseline.games,
        b_top,
        b_share * 100.0
    );

    for epoch in 1..=args.epochs {
        let train = train_one_epoch(&mut trainer, &train_packs, &args, scaled_lr, epoch, args.epochs)
            .map_err(|e| io::Error::other(format!("train epoch failed: {e}")))?;
        let (t_top, t_share) = top_bucket_info(&train.bucket_hist);
        println!(
            "epoch {} train_loss {:.6} samples {} games {} steps {} top_bucket b{} ({:.2}%)",
            epoch,
            train.mean_loss,
            train.samples,
            train.games,
            train.steps,
            t_top,
            t_share * 100.0
        );

        let val = evaluate_split(
            &mut trainer,
            &val_packs,
            val_max,
            args.games_per_step,
            args.reader_threads,
            args.prefetch_depth,
        )
        .map_err(|e| io::Error::other(format!("val eval failed: {e}")))?;
        let (v_top, v_share) = top_bucket_info(&val.bucket_hist);
        println!(
            "epoch {} val_loss {:.6} samples {} games {} top_bucket b{} ({:.2}%)",
            epoch,
            val.mean_loss,
            val.samples,
            val.games,
            v_top,
            v_share * 100.0
        );

        if args.save_each_epoch {
            let weights = trainer.read_weights().map_err(|e| io::Error::other(format!("read_weights failed: {e}")))?;
            let ckpt = epoch_checkpoint_path(&args.output, epoch);
            write_progress_bin(&ckpt, &weights)?;
            println!("epoch {} checkpoint: {}", epoch, ckpt.display());
        }
    }

    let weights = trainer.read_weights().map_err(|e| io::Error::other(format!("final read_weights failed: {e}")))?;
    write_progress_bin(&args.output, &weights)?;
    let bytes = fs::metadata(&args.output)?.len();
    println!("Wrote {} weights to {} ({} bytes)", SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, args.output.display(), bytes);

    Ok(())
}
