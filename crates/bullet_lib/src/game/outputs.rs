use bulletformat::{ChessBoard, chess::MarlinFormat};

use crate::shogi::PackedSfenValue;

pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&self, pos: &T) -> u8;
}

#[deprecated(note = "You do not need to specify this anymore, it is the default!")]
#[derive(Clone, Copy, Default)]
pub struct Single;

#[allow(deprecated)]
impl<T: 'static> OutputBuckets<T> for Single {
    const BUCKETS: usize = 1;

    fn bucket(&self, _: &T) -> u8 {
        0
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

impl<const N: usize> OutputBuckets<MarlinFormat> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &MarlinFormat) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

/// 将棋 LayerStacks 用出力バケット
///
/// 両玉の相対段に基づいて N バケットに分類する。
/// rshogi の `compute_bucket_index` / `compute_king_ranks` と同一ロジック。
///
/// 標準では N=9 (3×3 マトリクス):
/// ```text
///        e_rank 0-2  e_rank 3-5  e_rank 6-8
/// f_rank 0-2:    0        1            2
/// f_rank 3-5:    3        4            5
/// f_rank 6-8:    6        7            8
/// ```
#[derive(Clone, Copy, Default)]
pub struct ShogiKingRankBucket<const N: usize>;

impl<const N: usize> OutputBuckets<PackedSfenValue> for ShogiKingRankBucket<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let board = pos.decode();

        let side_to_move = board.side_to_move;
        let f_king = board.king_square(side_to_move);
        let e_king = board.king_square(side_to_move.opponent());

        // 味方玉の段（味方から見た相対段）
        let f_rank = match side_to_move {
            crate::shogi::Color::Black => f_king.rank() as usize,
            crate::shogi::Color::White => 8 - f_king.rank() as usize,
        };

        // 相手玉の段（相手から見た相対段）
        let e_rank = match side_to_move {
            crate::shogi::Color::Black => 8 - e_king.rank() as usize,
            crate::shogi::Color::White => e_king.rank() as usize,
        };

        const F_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
        const E_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

        let bucket = F_TO_INDEX[f_rank.min(8)] + E_TO_INDEX[e_rank.min(8)];
        bucket.min(N - 1) as u8
    }
}

/// Default boundaries for shogi ply-based 9-bucket split.
///
/// bucket0: <=30, bucket1: <=44, ..., bucket7: <=138, bucket8: >=139
pub const SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS: [u16; 8] = [30, 44, 58, 72, 86, 100, 116, 138];

/// Number of features used by progress-based bucket model.
pub const SHOGI_PROGRESS8_NUM_FEATURES: usize = 6;

/// Feature order for progress-based bucket model (coeff_v1).
pub const SHOGI_PROGRESS8_FEATURE_ORDER: [&str; SHOGI_PROGRESS8_NUM_FEATURES] = [
    "x_board_non_king",
    "x_hand_total",
    "x_major_board",
    "x_promoted_board",
    "x_stm_king_rank_rel",
    "x_ntm_king_rank_rel",
];

/// Number of buckets for progress8.
pub const SHOGI_PROGRESS8_NUM_BUCKETS: usize = 8;

/// Progress-based 8 bucket assignment (logistic regression).
///
/// `p = sigmoid(bias + Σ(w_i * ((x_i - mean_i) / std_i)))`
/// `bucket = min(7, floor(p * 8.0))`
#[derive(Clone, Copy)]
pub struct ShogiProgressBucket8 {
    pub mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub std: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub bias: f32,
    pub z_clip: [f32; 2],
}

impl ShogiProgressBucket8 {
    pub const fn new(
        mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        std: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        bias: f32,
        z_clip: [f32; 2],
    ) -> Self {
        Self { mean, std, weights, bias, z_clip }
    }

    /// Extract raw progress-model features in coeff_v1 order.
    pub fn extract_features(pos: &PackedSfenValue) -> [f32; SHOGI_PROGRESS8_NUM_FEATURES] {
        let board = pos.decode();

        let board_non_king = board
            .board
            .iter()
            .filter(|p| p.piece_type != crate::shogi::PieceType::None && p.piece_type != crate::shogi::PieceType::King)
            .count() as f32;

        let hand_total =
            board.black_hand.counts.iter().chain(board.white_hand.counts.iter()).map(|&v| v as f32).sum::<f32>();

        let major_board = board
            .board
            .iter()
            .filter(|p| {
                matches!(
                    p.piece_type,
                    crate::shogi::PieceType::Bishop
                        | crate::shogi::PieceType::Rook
                        | crate::shogi::PieceType::Horse
                        | crate::shogi::PieceType::Dragon
                )
            })
            .count() as f32;

        let promoted_board = board.board.iter().filter(|p| p.piece_type.is_promoted()).count() as f32;

        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());

        let stm_king_rank_rel = match stm {
            crate::shogi::Color::Black => f_king.rank() as f32,
            crate::shogi::Color::White => (8 - f_king.rank()) as f32,
        };

        let ntm_king_rank_rel = match stm {
            crate::shogi::Color::Black => (8 - e_king.rank()) as f32,
            crate::shogi::Color::White => e_king.rank() as f32,
        };

        [board_non_king, hand_total, major_board, promoted_board, stm_king_rank_rel, ntm_king_rank_rel]
    }

    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let x = Self::extract_features(pos);

        let mut z = self.bias;
        for i in 0..SHOGI_PROGRESS8_NUM_FEATURES {
            let std = if self.std[i] > 0.0 { self.std[i] } else { 1.0 };
            let x_norm = (x[i] - self.mean[i]) / std;
            z += self.weights[i] * x_norm;
        }

        let z_min = self.z_clip[0].min(self.z_clip[1]);
        let z_max = self.z_clip[0].max(self.z_clip[1]);
        let z_clamped = z.clamp(z_min, z_max);
        let p = 1.0 / (1.0 + (-z_clamped).exp());
        p.clamp(0.0, 1.0)
    }
}

impl Default for ShogiProgressBucket8 {
    fn default() -> Self {
        // docs/progress-bucket-coeff-script-spec-v1.md の JSON 例を既定値として採用。
        Self {
            mean: [30.12, 8.45, 2.18, 1.63, 6.71, 6.24],
            std: [3.77, 4.02, 0.66, 1.40, 1.31, 1.27],
            weights: [-0.81, 0.56, -0.32, 0.48, 0.11, -0.09],
            bias: -0.15,
            z_clip: [-8.0, 8.0],
        }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressBucket8 {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as u8
    }
}

/// 将棋 LayerStacks 用の手数ベース 9 バケット。
///
/// `game_ply` を固定境界で 9 分割する:
/// - b0: <= bounds[0]
/// - ...
/// - b7: <= bounds[7]
/// - b8: > bounds[7]
#[derive(Clone, Copy)]
pub struct ShogiPlyBucket9 {
    pub bounds: [u16; 8],
}

impl Default for ShogiPlyBucket9 {
    fn default() -> Self {
        Self { bounds: SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiPlyBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let ply = pos.game_ply();
        for (i, &bound) in self.bounds.iter().enumerate() {
            if ply <= bound {
                return i as u8;
            }
        }
        8
    }
}

/// Runtime-selectable 9-bucket mode for shogi LayerStacks.
#[derive(Clone, Copy)]
pub enum ShogiLayerStackBucket9 {
    KingRank9,
    Ply9([u16; 8]),
    Progress8(ShogiProgressBucket8),
}

impl Default for ShogiLayerStackBucket9 {
    fn default() -> Self {
        Self::KingRank9
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiLayerStackBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        match self {
            Self::KingRank9 => ShogiKingRankBucket::<9>.bucket(pos),
            Self::Ply9(bounds) => {
                let ply = pos.game_ply();
                for (i, &bound) in bounds.iter().enumerate() {
                    if ply <= bound {
                        return i as u8;
                    }
                }
                8
            }
            // 9bucket互換モード:
            // progress8 は bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8(progress) => progress.bucket(pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn psv_with_ply(ply: u16) -> PackedSfenValue {
        let mut psv = PackedSfenValue::default();
        let bytes = psv.as_bytes_mut();
        let le = ply.to_le_bytes();
        bytes[36] = le[0];
        bytes[37] = le[1];
        psv
    }

    #[test]
    fn test_shogi_ply_bucket9_default_bounds() {
        let bucket = ShogiPlyBucket9::default();
        assert_eq!(bucket.bucket(&psv_with_ply(0)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(30)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(31)), 1);
        assert_eq!(bucket.bucket(&psv_with_ply(138)), 7);
        assert_eq!(bucket.bucket(&psv_with_ply(139)), 8);
        assert_eq!(bucket.bucket(&psv_with_ply(400)), 8);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_ply_mode() {
        let bucket = ShogiLayerStackBucket9::Ply9([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(bucket.bucket(&psv_with_ply(10)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(21)), 2);
        assert_eq!(bucket.bucket(&psv_with_ply(80)), 7);
        assert_eq!(bucket.bucket(&psv_with_ply(81)), 8);
    }

    #[test]
    fn test_shogi_progress_bucket8_range() {
        let bucket = ShogiProgressBucket8::default();
        for ply in [1u16, 30, 60, 100, 150, 220, 300] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8(ShogiProgressBucket8::default());
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-in-9 bucket must be in 0..=7, got {}", b);
        }
    }
}
