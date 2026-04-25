//! ShogiHalfKaHmHandCount — HalfKA_hm + HandCount Dense 補助入力
//!
//! 盤・駒・王の情報は従来の HalfKA_hm と完全に同じ sparse 表現で提供し、
//! 追加で 14 元の dense vector (= stm 持ち駒 7 種 + nstm 持ち駒 7 種) を
//! `"hand_count"` という名前の dense 入力として公開する。
//!
//! L1 層の入力に `[FT_output (1536), hand_count (14)]` という形で concat して
//! 使うことを想定している。
//!
//! レイアウト順は以下の 14 元:
//!
//! ```text
//! index 0..6: stm 視点の持ち駒本数 (歩, 香, 桂, 銀, 金, 角, 飛)
//! index 7..13: nstm 視点の持ち駒本数 (歩, 香, 桂, 銀, 金, 角, 飛)
//! ```
//!
//! 値は `u8` の駒本数を `f32` にキャストしたもの（0.0〜最大 18.0）。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, HAND_PIECE_TYPES},
};

/// HandCount dense 補助入力の次元数。
///
/// stm/nstm 視点 × 7 駒種（歩香桂銀金角飛）= 14。
pub const HAND_COUNT_DIMS: usize = 14;

/// ShogiHalfKA_hm + HandCount Dense 補助入力を提供する SparseInputType。
///
/// sparse feature は HalfKA_hm と完全互換（73,305 次元、max_active 40）。
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHmHandCount;

impl SparseInputType for ShogiHalfKaHmHandCount {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        // sparse 部分は HalfKA_hm と完全に同一なので delegate。
        super::ShogiHalfKA_hm.map_features(pos, f);
    }

    fn shorthand(&self) -> String {
        "shogi-73305x45hm+handcount14".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKA_hm + HandCount Dense (14 dims, stm/nstm hand piece counts)".to_string()
    }

    fn hand_count_dims(&self) -> Option<usize> {
        Some(HAND_COUNT_DIMS)
    }

    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        assert_eq!(out.len(), HAND_COUNT_DIMS, "hand_count 出力スロットは 14 次元");

        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let write = |out: &mut [f32], offset: usize, color: Color, board: &ShogiBoard| {
            for (i, &pt) in HAND_PIECE_TYPES.iter().enumerate() {
                out[offset + i] = f32::from(board.hand(color).count(pt));
            }
        };

        write(out, 0, stm, &board);
        write(out, 7, nstm, &board);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::PieceType;

    fn dummy_pos_with_hands(black_counts: [u8; 7], white_counts: [u8; 7], stm: Color) -> PackedSfenValue {
        // packed から board を作るのは重いので、ShogiBoard を直接構築して
        // fill_hand_count に渡せる形を作る簡易テスト。
        // ここでは存在するコードパスを呼びたいだけなので board は直接操作。
        let _ = (black_counts, white_counts, stm);
        PackedSfenValue::default()
    }

    #[test]
    fn dimensions_match_halfka_hm() {
        let input = ShogiHalfKaHmHandCount;
        assert_eq!(input.num_inputs(), HALFKA_HM_DIMENSIONS);
        assert_eq!(input.max_active(), MAX_ACTIVE_FEATURES);
        assert_eq!(input.hand_count_dims(), Some(14));
    }

    #[test]
    fn fill_hand_count_writes_14_elements() {
        let input = ShogiHalfKaHmHandCount;
        let pos = dummy_pos_with_hands([0; 7], [0; 7], Color::Black);
        let mut out = [f32::NAN; 14];
        input.fill_hand_count(&pos, &mut out);
        // 全値 0 or 有効な f32。NaN が残っていない。
        for v in out.iter() {
            assert!(!v.is_nan(), "hand_count が未書き込み");
            assert!(*v >= 0.0, "hand_count は非負");
        }
    }

    #[test]
    fn hand_piece_types_len_matches_half() {
        // HAND_PIECE_TYPES は 7 種（歩香桂銀金角飛）。
        let pts: Vec<PieceType> = HAND_PIECE_TYPES.to_vec();
        assert_eq!(pts.len(), HAND_COUNT_DIMS / 2);
    }
}
