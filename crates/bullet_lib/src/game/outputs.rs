use bulletformat::{ChessBoard, chess::MarlinFormat};

use crate::shogi::PackedSfenValue;
#[cfg(test)]
use crate::shogi::{Color, ShogiBoard};

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

// =============================================================================
// 将棋用出力バケット
// =============================================================================

/// 両玉の段に基づく9バケット (nnue-pytorch 互換)
///
/// LayerStack アーキテクチャで使用する出力バケット。
/// 両玉の段位置に基づいて 0-8 のバケットを選択する。
///
/// # バケット計算
///
/// - 味方玉の段: 0-2 → 0, 3-5 → 3, 6-8 → 6
/// - 相手玉の段: 0-2 → 0, 3-5 → 1, 6-8 → 2
/// - bucket = F_TO_INDEX[f_rank] + E_TO_INDEX[e_rank]
///
/// # 注意
///
/// 段 (rank) は両玉とも自分視点で計算:
/// - 味方玉: 1段目が陣内 (rank=0-2)、9段目が敵陣 (rank=6-8)
/// - 相手玉: 相手視点なので反転（相手の1段目 = 自分の9段目）
#[derive(Clone, Copy, Default)]
pub struct ShogiKingRankBucket;

impl OutputBuckets<PackedSfenValue> for ShogiKingRankBucket {
    const BUCKETS: usize = 9;

    #[inline]
    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        // 高速版: PackedSfen の最初の15ビットから直接計算
        // フルボードデコードを回避して学習時のオーバーヘッドを削減
        pos.compute_bucket_fast()
    }
}

/// 将棋盤面から玉位置バケットを計算（テスト・検証用）
///
/// 両玉の段に基づいて 0-8 のバケットインデックスを返す。
/// 本番では `PackedSfenValue::compute_bucket_fast()` を使用。
#[cfg(test)]
#[inline]
fn compute_king_rank_bucket(board: &ShogiBoard) -> u8 {
    let stm = board.side_to_move;

    let f_king_sq = board.king_square(stm);
    let e_king_sq = board.king_square(stm.opponent());

    // 片玉データの場合はバケット 0 を返す
    if !f_king_sq.is_valid() || !e_king_sq.is_valid() {
        return 0;
    }

    // 味方玉の段（味方視点: 先手なら1段目=0、後手なら9段目=0）
    let f_rank = if stm == Color::Black { f_king_sq.rank() as usize } else { 8 - f_king_sq.rank() as usize };

    // 相手玉の段（相手視点: 相手から見た段）
    let e_rank = if stm == Color::Black { 8 - e_king_sq.rank() as usize } else { e_king_sq.rank() as usize };

    // バケット計算テーブル
    // 段 0-2 → 0, 3-5 → 1or3, 6-8 → 2or6
    const F_TO_INDEX: [u8; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
    const E_TO_INDEX: [u8; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    F_TO_INDEX[f_rank] + E_TO_INDEX[e_rank]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::{Piece, PieceType, Square};

    #[test]
    fn test_king_rank_bucket_initial() {
        // 初期配置: 先手玉5九(rank=8)、後手玉5一(rank=0)
        // 先手視点:
        //   f_rank = 8 (自陣深部) → F_TO_INDEX[8] = 6
        //   e_rank = 8 - 0 = 8 (相手から見て敵陣) → E_TO_INDEX[8] = 2
        //   bucket = 6 + 2 = 8
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::new(4, 0); // 5一
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        assert_eq!(compute_king_rank_bucket(&board), 8);
    }

    #[test]
    fn test_king_rank_bucket_white_turn() {
        // 後手番: 先手玉5九(rank=8)、後手玉5一(rank=0)
        // 後手視点:
        //   f_king = 5一 → f_rank = 8 - 0 = 8 (後手視点で自陣深部)
        //   e_king = 5九 → e_rank = 8 (後手視点で見ると rank=8)
        //   bucket = 6 + 2 = 8
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::White;
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::new(4, 0); // 5一
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        assert_eq!(compute_king_rank_bucket(&board), 8);
    }

    #[test]
    fn test_king_rank_bucket_mid_game() {
        // 中盤想定: 先手玉5七(rank=6)、後手玉5三(rank=2)
        // 先手視点:
        //   f_rank = 6 → F_TO_INDEX[6] = 6
        //   e_rank = 8 - 2 = 6 → E_TO_INDEX[6] = 2
        //   bucket = 6 + 2 = 8
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;
        board.black_king_sq = Square::new(4, 6); // 5七
        board.white_king_sq = Square::new(4, 2); // 5三
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        assert_eq!(compute_king_rank_bucket(&board), 8);
    }

    #[test]
    fn test_king_rank_bucket_all_combinations() {
        // バケット値の全組み合わせをテスト
        // e_rank は stm 視点で計算: e_rank = 8 - e_king_sq.rank() (先手視点)
        //
        // bucket = F_TO_INDEX[f_rank] + E_TO_INDEX[e_rank]
        // F_TO_INDEX: [0,0,0,3,3,3,6,6,6]
        // E_TO_INDEX: [0,0,0,1,1,1,2,2,2]
        let test_cases = [
            // (f_rank, e_king_rank_orig, expected_bucket)
            // f_rank=0 → F[0]=0, e_rank=8-8=0 → E[0]=0, bucket=0
            (0, 8, 0),
            // f_rank=1 → F[1]=0, e_rank=8-5=3 → E[3]=1, bucket=1
            (1, 5, 1),
            // f_rank=2 → F[2]=0, e_rank=8-2=6 → E[6]=2, bucket=2
            (2, 2, 2),
            // f_rank=4 → F[4]=3, e_rank=8-7=1 → E[1]=0, bucket=3
            (4, 7, 3),
            // f_rank=7 → F[7]=6, e_rank=8-4=4 → E[4]=1, bucket=7
            (7, 4, 7),
        ];

        for (f_rank, e_king_rank_orig, expected_bucket) in test_cases {
            let mut board = ShogiBoard::default();
            board.side_to_move = Color::Black;
            board.black_king_sq = Square::new(4, f_rank);
            board.white_king_sq = Square::new(4, e_king_rank_orig);
            board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
            board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

            assert_eq!(
                compute_king_rank_bucket(&board),
                expected_bucket,
                "f_rank={}, e_king_rank_orig={}",
                f_rank,
                e_king_rank_orig
            );
        }
    }

    #[test]
    fn test_king_rank_bucket_invalid_king() {
        // 片玉データのテスト
        let mut board = ShogiBoard::default();
        board.side_to_move = Color::Black;
        board.black_king_sq = Square::new(4, 8);
        board.white_king_sq = Square::NONE; // 無効な玉位置
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);

        // 無効な場合はバケット 0 を返す
        assert_eq!(compute_king_rank_bucket(&board), 0);
    }
}
