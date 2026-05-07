mod direct;
mod montybinpack;
mod rng;
pub mod sfbinpack;
pub mod shogipack;
mod text;
pub mod viribinpack;

pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use shogipack::ShogiPackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::{ViriBinpackLoader, ViriFilter};

use bulletformat::BulletFormat;

use crate::game::{inputs::SparseInputType, outputs::OutputBuckets};

use super::Wgt;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, f: F);
}

pub(crate) type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;

#[derive(Clone)]
pub struct DefaultDataLoader<I: SparseInputType, O, D> {
    input_getter: I,
    output_getter: O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    scale: f32,
    loader: D,
}

impl<I: SparseInputType, O, D> DefaultDataLoader<I, O, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        scale: f32,
        loader: D,
    ) -> Self {
        Self { input_getter, output_getter, blend_getter, weight_getter, use_win_rate_model, wdl, scale, loader }
    }
}

impl<I, O, D> DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    pub fn load_and_map_batches<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        mut f: F,
    ) {
        let mut incomplete_buf = Vec::new();

        self.loader.map_chunks(start_batch * batch_size, |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let should_break = f(&incomplete_buf);
                    incomplete_buf.clear();

                    if should_break {
                        return true;
                    }
                } else {
                    incomplete_buf.extend_from_slice(chunk);
                }

                remainder
            } else {
                0
            };

            if chunk.len() >= remainder {
                let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
                incomplete_buf.extend_from_slice(chunks.remainder());

                for batch in chunks {
                    let should_break = f(batch);

                    if should_break {
                        return true;
                    }
                }
            }

            false
        });
    }

    pub fn prepare(&self, data: &[I::RequiredDataType], threads: usize, blend: f32) -> PreparedData<I, O> {
        PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
        )
    }
}

/// A batch of data, in the correct format for the GPU.
pub struct PreparedData<I: SparseInputType, O> {
    pub(crate) input_getter: I,
    pub(crate) output_getter: O,
    pub(crate) batch_size: usize,
    pub(crate) stm: Vec<i32>,
    pub(crate) nstm: Vec<i32>,
    pub(crate) buckets: Vec<i32>,
    pub(crate) targets: Vec<f32>,
    pub(crate) weights: Vec<f32>,
    /// HandCount dense auxiliary input。`I::hand_count_dims()` が `Some` のとき
    /// `Some(hand_count_dim * batch_size)` 長の flat Vec を保持する (列方向 = batch index)。
    /// 次元数 (`hand_count_dim`) はここでは持たず、consumer (model 定義側) が
    /// `input_getter.hand_count_dims()` から取得する前提。
    pub(crate) hand_count: Option<Vec<f32>>,
}

impl<I, O> PreparedData<I, O>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);
        let input_size = input_getter.num_inputs();
        let output_size = if wdl { 3 } else { 1 };
        let sparse_size = max_active * batch_size;
        let hand_count_dims = input_getter.hand_count_dims();
        let hand_count_dim = hand_count_dims.unwrap_or(0);

        let hand_count_init = hand_count_dims.map(|dims| vec![0.0; dims * batch_size]);

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: vec![0; sparse_size],
            nstm: vec![0; sparse_size],
            buckets: vec![0; batch_size],
            targets: vec![0.0; output_size * batch_size],
            weights: vec![0.0; batch_size],
            hand_count: hand_count_init,
        };

        let sparse_chunk_size = max_active * chunk_size;

        // HandCount 用の並列チャンクを事前に materialise。Option は並列ループ内で扱う。
        let hand_count_chunk_size = hand_count_dim * chunk_size;
        let num_chunks = batch_size.div_ceil(chunk_size);
        let hand_count_slices: Vec<Option<&mut [f32]>> = if let Some(hc) = prep.hand_count.as_mut() {
            hc.chunks_mut(hand_count_chunk_size).map(Some).collect()
        } else {
            (0..num_chunks).map(|_| None).collect()
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.chunks_mut(sparse_chunk_size))
                .zip(prep.nstm.chunks_mut(sparse_chunk_size))
                .zip(prep.buckets.chunks_mut(chunk_size))
                .zip(prep.targets.chunks_mut(output_size * chunk_size))
                .zip(prep.weights.chunks_mut(chunk_size))
                .zip(hand_count_slices)
                .for_each(
                    |(
                        (((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk),
                        hand_count_chunk,
                    )| {
                        let inp = &prep.input_getter;
                        let out = &prep.output_getter;
                        s.spawn(move || {
                            let chunk_len = data_chunk.len();
                            let mut hand_count_chunk = hand_count_chunk;

                            for i in 0..chunk_len {
                                let pos = &data_chunk[i];

                                if let Some(hc_slice) = hand_count_chunk.as_deref_mut() {
                                    let offset = hand_count_dim * i;
                                    let end = offset + hand_count_dim;
                                    // 事前に 0 で埋め済み。fill_hand_count は
                                    // 書き込みのみで読まないので再初期化は不要。
                                    inp.fill_hand_count(pos, &mut hc_slice[offset..end]);
                                }
                                // STM と NSTM は独立カウンタで管理: 非対称 feature
                                // (HandThreat defensive 等) で |STM_active| != |NSTM_active|
                                // を許可するため。symmetric な input type は
                                // map_features_split の default impl 経由で
                                // 両側同時に進むので従来挙動と一致する。
                                let mut j_stm: usize = 0;
                                let mut j_nstm: usize = 0;
                                let sparse_offset = max_active * i;

                                inp.map_features_split(pos, |our_opt, opp_opt| {
                                    if let Some(our) = our_opt {
                                        assert!(our < input_size, "STM feature index exceeded input size!");
                                        stm_chunk[sparse_offset + j_stm] = our as i32;
                                        j_stm += 1;
                                    }
                                    if let Some(opp) = opp_opt {
                                        assert!(opp < input_size, "NSTM feature index exceeded input size!");
                                        nstm_chunk[sparse_offset + j_nstm] = opp as i32;
                                        j_nstm += 1;
                                    }
                                });

                                // STM / NSTM の未使用スロットを -1 で埋める (独立)
                                for j in j_stm..max_active {
                                    stm_chunk[sparse_offset + j] = -1;
                                }
                                for j in j_nstm..max_active {
                                    nstm_chunk[sparse_offset + j] = -1;
                                }

                                assert!(
                                    j_stm <= max_active && j_nstm <= max_active,
                                    "More inputs provided than the specified maximum!"
                                );

                                buckets_chunk[i] = i32::from(out.bucket(pos));
                                weights_chunk[i] = weight_getter.map_or(1.0, |w| w(pos));

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = f32::from(pos.score());
                                    let score = if use_win_rate_model {
                                        let p = (score - 270.0) / 380.0;
                                        let pm = (-score - 270.0) / 380.0;
                                        0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
                                    } else {
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }
                        });
                    },
                );
        });

        prep
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
