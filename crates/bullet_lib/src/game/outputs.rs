use bulletformat::{chess::MarlinFormat, ChessBoard};

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
}
