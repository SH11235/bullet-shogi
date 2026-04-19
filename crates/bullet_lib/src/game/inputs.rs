mod adapter;
mod ataxx147;
mod chess768;
mod chess_buckets;
mod chess_buckets_mk;
mod factorised;
mod shogi_halfka;
mod shogi_halfka_hm_hand_count;
mod shogi_halfka_hm_hand_threat;
mod shogi_halfka_hm_hand_threat_defensive;
mod shogi_halfka_hm_threat;
mod shogi_halfkp;
pub mod shogi_threat_exclusion;

#[allow(deprecated)]
mod legacy;

pub use adapter::MarlinFormatAdapter;
pub use ataxx147::{Ataxx98, Ataxx147};
pub use chess_buckets::{ChessBuckets, ChessBucketsMirrored};
pub use chess768::Chess768;
pub use factorised::{Factorised, Factorises};
pub use shogi_halfka::{
    FEATURE_HASH_HM_V2, FEATURE_HASH_NONMIRROR, HALFKA_HM_DIMENSIONS, NUM_KING_BUCKETS, PIECE_INPUTS, ShogiHalfKA,
    ShogiHalfKA_hm,
};
pub use shogi_halfka_hm_hand_count::{HAND_COUNT_DIMS, ShogiHalfKaHmHandCount};
pub use shogi_halfka_hm_hand_threat::ShogiHalfKaHmHandThreat;
pub use shogi_halfka_hm_hand_threat_defensive::ShogiHalfKaHmHandThreatDefensive;
pub use shogi_halfka_hm_threat::ShogiHalfKaHmThreat;
pub use shogi_halfkp::{FEATURE_HASH, ShogiHalfKP};
pub use shogi_threat_exclusion::ThreatProfile;

#[allow(deprecated)]
pub use chess_buckets_mk::*;

#[allow(deprecated)]
pub use legacy::InputType;

#[deprecated(note = "See `examples/progression/3_input_buckets.rs` for a faster alternative to this.")]
pub type ChessBucketsFactorised = Factorised<ChessBuckets, Chess768>;

#[allow(deprecated)]
impl ChessBucketsFactorised {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self::from_parts(ChessBuckets::new(buckets), Chess768)
    }
}

#[deprecated(note = "See `examples/progression/3_input_buckets.rs` for a faster alternative to this.")]
pub type ChessBucketsMirroredFactorised = Factorised<ChessBucketsMirrored, Chess768>;

#[allow(deprecated)]
impl ChessBucketsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMirrored::new(buckets), Chess768)
    }
}

pub trait SparseInputType: Clone + Send + Sync + 'static {
    type RequiredDataType: Send + Sync;

    /// The total number of inputs
    fn num_inputs(&self) -> usize;

    /// The maximum number of active inputs
    fn max_active(&self) -> usize;

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F);

    /// Asymmetric feature emission.
    ///
    /// Each call to `f(stm_opt, nstm_opt)` activates at most one index on each
    /// perspective independently. `None` means "no feature on this side for this
    /// call", allowing asymmetric active sets (|STM| != |NSTM|).
    ///
    /// The default implementation delegates to the symmetric `map_features`,
    /// emitting `(Some(stm), Some(nstm))` for every call. Existing symmetric
    /// input types do not need to override this method — they remain fully
    /// backward compatible.
    ///
    /// Input types that need asymmetric features (e.g. HandThreat defensive)
    /// must override this method and leave `map_features` calling a single-
    /// purpose symmetric emission (or panic).
    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.map_features(pos, |stm, nstm| f(Some(stm), Some(nstm)));
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String;

    /// Description of the input type
    fn description(&self) -> String;

    /// Dense auxiliary input (`"hand_count"`) の次元数。
    ///
    /// `Some(dims)` を返すと、`PreparedData` は `hand_count` 用の dense 領域を
    /// 確保し、サンプルごとに `fill_hand_count` を呼び出してトレーナ graph の
    /// `"hand_count"` 入力ノードに渡す。
    ///
    /// 既定は `None`（HandCount dense 入力を使用しない）。
    fn hand_count_dims(&self) -> Option<usize> {
        None
    }

    /// Dense auxiliary input (`"hand_count"`) をサンプル分埋める。
    ///
    /// `out.len()` は `hand_count_dims().unwrap_or(0)` と一致する。
    /// `hand_count_dims() == None` の input type では呼び出されないため既定は空実装。
    fn fill_hand_count(&self, _pos: &Self::RequiredDataType, _out: &mut [f32]) {}

    fn is_factorised(&self) -> bool {
        false
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        assert!(self.is_factorised());
        unmerged
    }
}

pub const fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;

    while i < N {
        if arr[i] > max {
            max = arr[i];
        }

        i += 1;
    }
    max + 1
}
