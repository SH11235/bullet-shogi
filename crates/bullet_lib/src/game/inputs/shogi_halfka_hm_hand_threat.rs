//! HalfKA_hm + HandThreat 連結特徴量 (案 A: full drop-attack pair)
//!
//! HalfKA_hm (73,305 次元) と HandThreat (121,104 次元) を連結した
//! sparse input 型を提供する。
//!
//! ## 仕様
//!
//! - HalfKA_hm: 既存の `ShogiHalfKA_hm` と同一ロジック (shogi_halfka_hm_threat.rs と共通)
//! - HandThreat: rshogi `hand_threat_features.rs` と同一 index 計算
//! - 設計ノート: `docs/performance/hand_threat_design_20260413.md` (rshogi リポジトリ)
//! - Profile 当面なし (案 A full drop-attack pair のみ)
//!
//! ## 依存
//!
//! board Threat の attack pattern LUT (`FromOffsetTable`、`compute_attack_order` 等) は
//! `shogi_halfka_hm_threat.rs` の pub(super) helper を流用する。

use super::SparseInputType;
use super::shogi_halfka::{
    HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, halfka_index, is_hm_mirror, king_bonapiece, king_bucket, pack_bonapiece,
};
use super::shogi_halfka_hm_threat::{
    ATTACK_ORDER_TABLE, AttackOrderTable, FROM_OFFSET_TABLE, NUM_THREAT_CLASSES, Occupied, ThreatClass,
    attack_pattern_id, normalize_sq,
};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    bona_piece::BonaPiece,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, PieceType, Square},
};

// =============================================================================
// HandThreat 定数
// =============================================================================

/// Drop 可能な駒種の数 (Pawn/Lance/Knight/Silver/Gold/Bishop/Rook)
const HAND_NUM_CLASSES: usize = 7;

/// HandThreat active features の上限 (bullet-shogi 側の安全側上限)
const MAX_ACTIVE_HAND_THREAT_FEATURES: usize = 1024;

/// HalfKA_hm + HandThreat 合計の max active
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + MAX_ACTIVE_HAND_THREAT_FEATURES;

/// `HandThreatClass` → board `ThreatClass` マッピング
///
/// rshogi `hand_threat_features.rs` の `HAND_TO_BOARD_CLASS` と完全一致させること。
const HAND_TO_BOARD_CLASS: [ThreatClass; HAND_NUM_CLASSES] = [
    ThreatClass::Pawn,     // 0: Pawn
    ThreatClass::Lance,    // 1: Lance
    ThreatClass::Knight,   // 2: Knight
    ThreatClass::Silver,   // 3: Silver
    ThreatClass::GoldLike, // 4: Gold (board 側は GoldLike)
    ThreatClass::Bishop,   // 5: Bishop
    ThreatClass::Rook,     // 6: Rook
];

/// 各 HandThreatClass の drop → 利き数 (color ごと)
///
/// = [72, 324, 112, 328, 416, 816, 1296]
/// 合計 = 3,364
const HAND_ATTACKS_PER_COLOR: [usize; HAND_NUM_CLASSES] = [
    72,   // Pawn
    324,  // Lance
    112,  // Knight
    328,  // Silver
    416,  // GoldLike (Gold)
    816,  // Bishop
    1296, // Rook
];

// =============================================================================
// HandThreatClass
// =============================================================================

/// 持ち駒で drop 可能な駒種
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum HandThreatClass {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    Gold = 4,
    Bishop = 5,
    Rook = 6,
}

const ALL_HAND_THREAT_CLASSES: [HandThreatClass; HAND_NUM_CLASSES] = [
    HandThreatClass::Pawn,
    HandThreatClass::Lance,
    HandThreatClass::Knight,
    HandThreatClass::Silver,
    HandThreatClass::Gold,
    HandThreatClass::Bishop,
    HandThreatClass::Rook,
];

impl HandThreatClass {
    #[inline]
    fn as_board_class(self) -> ThreatClass {
        HAND_TO_BOARD_CLASS[self as usize]
    }

    #[inline]
    fn as_piece_type(self) -> PieceType {
        match self {
            Self::Pawn => PieceType::Pawn,
            Self::Lance => PieceType::Lance,
            Self::Knight => PieceType::Knight,
            Self::Silver => PieceType::Silver,
            Self::Gold => PieceType::Gold,
            Self::Bishop => PieceType::Bishop,
            Self::Rook => PieceType::Rook,
        }
    }
}

// =============================================================================
// hand_pair_base テーブル
// =============================================================================

/// hand_pair_base の pair 数
/// = 2 (drop_owner) × 7 (hand_class) × 2 (attacked_side) × 9 (attacked_class)
const HAND_NUM_PAIRS: usize = 2 * HAND_NUM_CLASSES * 2 * NUM_THREAT_CLASSES; // 252

/// hand_pair_base テーブルと HAND_THREAT_DIMENSIONS を構築
///
/// Layout (flat): `drop_owner * 126 + hc * 18 + attacked_side * 9 + ac`
/// 126 = 7 * 18, 18 = 2 * 9
const fn build_hand_pair_base() -> ([usize; HAND_NUM_PAIRS], usize) {
    let mut table = [0usize; HAND_NUM_PAIRS];
    let mut cumulative = 0usize;
    let mut drop_owner = 0usize;
    while drop_owner < 2 {
        let mut hc = 0usize;
        while hc < HAND_NUM_CLASSES {
            let mut attacked_side = 0usize;
            while attacked_side < 2 {
                let mut ac = 0usize;
                while ac < NUM_THREAT_CLASSES {
                    let idx = drop_owner * 126 + hc * 18 + attacked_side * 9 + ac;
                    table[idx] = cumulative;
                    cumulative += HAND_ATTACKS_PER_COLOR[hc];
                    ac += 1;
                }
                attacked_side += 1;
            }
            hc += 1;
        }
        drop_owner += 1;
    }
    (table, cumulative)
}

const HAND_PAIR_DATA: ([usize; HAND_NUM_PAIRS], usize) = build_hand_pair_base();

static HAND_PAIR_BASE: [usize; HAND_NUM_PAIRS] = HAND_PAIR_DATA.0;

/// HandThreat の総特徴量次元数
///
/// 案 A: 2 × 7 × 2 × 9 × (72+324+112+328+416+816+1296) = 36 × 3,364 = **121,104**
pub const HAND_THREAT_DIMENSIONS: usize = HAND_PAIR_DATA.1;

const _HAND_THREAT_DIMENSIONS_CHECK: () = {
    assert!(HAND_THREAT_DIMENSIONS == 121_104, "HAND_THREAT_DIMENSIONS must be 121,104");
};

#[inline]
fn hand_pair_base(drop_owner: usize, hc: HandThreatClass, attacked_side: usize, ac: ThreatClass) -> usize {
    let idx = drop_owner * 126 + (hc as usize) * 18 + attacked_side * 9 + ac as usize;
    HAND_PAIR_BASE[idx]
}

// =============================================================================
// Legal drop check
// =============================================================================

/// `hand_class` の駒が `color` で `sq` に打てるか (行きどころ無し判定のみ)
#[inline]
fn is_legal_drop_rank(hand_class: HandThreatClass, color: Color, sq: Square) -> bool {
    let rank = sq.rank() as usize;
    match hand_class {
        HandThreatClass::Pawn | HandThreatClass::Lance => {
            if color == Color::Black {
                rank != 0
            } else {
                rank != 8
            }
        }
        HandThreatClass::Knight => {
            if color == Color::Black {
                rank >= 2
            } else {
                rank <= 6
            }
        }
        _ => true,
    }
}

/// `color` が `sq.file()` の列に Pawn (非成り) を持っているか (二歩判定)
fn has_pawn_on_file(board: &ShogiBoard, color: Color, sq: Square) -> bool {
    let file = sq.file();
    for rank in 0..9u8 {
        let s = Square::new(file, rank);
        let pc = board.piece_on(s);
        if !pc.is_none() && pc.color == color && pc.piece_type == PieceType::Pawn {
            return true;
        }
    }
    false
}

// =============================================================================
// drop 駒の attack 列挙
// =============================================================================

/// drop 後の駒の実盤面攻撃先を列挙する
///
/// 駒種 (HandThreatClass) + color + drop_sq + occupied から計算。
/// 二歩・行きどころ無しチェックは呼び出し側で先に行う前提。
fn for_each_drop_attack<F: FnMut(Square)>(
    hand_class: HandThreatClass,
    color: Color,
    from: Square,
    occ: &Occupied,
    callback: F,
) {
    // drop 駒は PieceType::{Pawn,Lance,...,Rook} で、成り前の駒。
    // shogi_halfka_hm_threat の for_each_attack はスライダー occupied を考慮するが、
    // drop 駒の場合は同じロジックを手動展開する。
    //
    // 実際には shogi_halfka_hm_threat::for_each_attack を流用すればよい:
    let pt = hand_class.as_piece_type();
    super::shogi_halfka_hm_threat::for_each_attack(pt, color, from, occ, callback);
}

// =============================================================================
// hand_threat_index
// =============================================================================

/// HandThreat index を計算
///
/// `attack_order` は O(1) LUT lookup (`ATTACK_ORDER_TABLE`) を使用する。
#[inline]
fn hand_threat_index(
    drop_owner: usize,
    hand_class: HandThreatClass,
    oriented_color: Color,
    attacked_side: usize,
    attacked_class: ThreatClass,
    drop_sq_n: Square,
    attack_to_sq_n: Square,
) -> usize {
    let base = hand_pair_base(drop_owner, hand_class, attacked_side, attacked_class);

    let pattern = attack_pattern_id(hand_class.as_board_class(), oriented_color);
    let from_offset_table = &*FROM_OFFSET_TABLE;
    let from_off = from_offset_table.get(pattern, drop_sq_n);
    let attack_ord = ATTACK_ORDER_TABLE.get(pattern, drop_sq_n, attack_to_sq_n);
    debug_assert_ne!(
        attack_ord,
        AttackOrderTable::INVALID,
        "hand_threat attack_order: to_sq {} not attacked by pattern {pattern} at {}",
        attack_to_sq_n.0,
        drop_sq_n.0
    );
    base + from_off + attack_ord as usize
}

// =============================================================================
// ShogiHalfKaHmHandThreat
// =============================================================================

/// HalfKA_hm + HandThreat 連結特徴量
///
/// SparseInputType を実装し、HalfKA_hm (73,305) + HandThreat (121,104) を
/// 連結した sparse input として提供する。
#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Default)]
pub struct ShogiHalfKaHmHandThreat;

impl ShogiHalfKaHmHandThreat {
    pub fn new() -> Self {
        Self
    }

    pub fn hand_threat_dimensions(&self) -> usize {
        HAND_THREAT_DIMENSIONS
    }
}

impl SparseInputType for ShogiHalfKaHmHandThreat {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS + HAND_THREAT_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_TOTAL
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        self.map_hand_threat_features(&board, f);
    }

    fn shorthand(&self) -> String {
        format!("shogi-{}x45hm+handthreat", HALFKA_HM_DIMENSIONS + HAND_THREAT_DIMENSIONS)
    }

    fn description(&self) -> String {
        format!("Shogi HalfKA_hm ({}) + HandThreat ({}) concatenated", HALFKA_HM_DIMENSIONS, HAND_THREAT_DIMENSIONS)
    }
}

// =============================================================================
// 特徴量列挙
// =============================================================================

impl ShogiHalfKaHmHandThreat {
    fn map_hand_threat_features<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king_sq = board.king_square(stm);
        let nstm_king_sq = board.king_square(nstm);
        if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
            return;
        }

        let stm_kb = king_bucket(stm_king_sq, stm);
        let stm_hm = is_hm_mirror(stm_king_sq, stm);
        let nstm_kb = king_bucket(nstm_king_sq, nstm);
        let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);

        // -------------------------------------------------------
        // Part 1: HalfKA_hm 特徴量 (既存 ShogiHalfKaHmThreat と同一)
        // -------------------------------------------------------

        // 盤上の駒（王以外）
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);
                    let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                    let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                    let stm_idx = halfka_index(stm_kb, stm_packed);

                    let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                    let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                    let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                    f(stm_idx, nstm_idx);
                }
            }
        }

        // 両方の玉の特徴量
        {
            let stm_king_sq_idx = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
            let stm_friend_king_bp = king_bonapiece(stm_king_sq_idx, true);
            let stm_friend_packed = pack_bonapiece(stm_friend_king_bp, stm_hm);
            let stm_friend_idx = halfka_index(stm_kb, stm_friend_packed);

            let nstm_king_sq_for_stm =
                if stm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
            let stm_enemy_king_bp = king_bonapiece(nstm_king_sq_for_stm, false);
            let stm_enemy_packed = pack_bonapiece(stm_enemy_king_bp, stm_hm);
            let stm_enemy_idx = halfka_index(stm_kb, stm_enemy_packed);

            let nstm_king_sq_idx =
                if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
            let nstm_friend_king_bp = king_bonapiece(nstm_king_sq_idx, true);
            let nstm_friend_packed = pack_bonapiece(nstm_friend_king_bp, nstm_hm);
            let nstm_friend_idx = halfka_index(nstm_kb, nstm_friend_packed);

            let stm_king_sq_for_nstm =
                if nstm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
            let nstm_enemy_king_bp = king_bonapiece(stm_king_sq_for_nstm, false);
            let nstm_enemy_packed = pack_bonapiece(nstm_enemy_king_bp, nstm_hm);
            let nstm_enemy_idx = halfka_index(nstm_kb, nstm_enemy_packed);

            f(stm_friend_idx, nstm_friend_idx);
            f(stm_enemy_idx, nstm_enemy_idx);
        }

        // 手駒の特徴量
        for owner in [Color::Black, Color::White] {
            for &pt in &HAND_PIECE_TYPES {
                let count = board.hand(owner).count(pt);
                if count == 0 {
                    continue;
                }
                for i in 1..=count {
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                    if stm_bp != BonaPiece::ZERO {
                        let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                        let stm_idx = halfka_index(stm_kb, stm_packed);

                        let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                        let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                        let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                        f(stm_idx, nstm_idx);
                    }
                }
            }
        }

        // -------------------------------------------------------
        // Part 2: HandThreat 特徴量
        // -------------------------------------------------------

        let occ = Occupied::from_board(board);

        // 両 drop_owner (friend/enemy from STM) をループ
        for &drop_color in &[stm, nstm] {
            // drop_owner flag: STM perspective と NSTM perspective で異なる
            // STM perspective: drop_color == stm → 0 (friend), else 1 (enemy)
            // NSTM perspective: drop_color == nstm → 0 (friend), else 1 (enemy)

            // 各 hand class を処理
            for &hand_class in &ALL_HAND_THREAT_CLASSES {
                if board.hand(drop_color).count(hand_class.as_piece_type()) == 0 {
                    continue;
                }

                for drop_raw in 0..81u8 {
                    let drop_sq = Square(drop_raw);

                    // (1) occupied
                    if occ.is_occupied(drop_raw) {
                        continue;
                    }
                    // (2) 行きどころ無し
                    if !is_legal_drop_rank(hand_class, drop_color, drop_sq) {
                        continue;
                    }
                    // (3) 二歩
                    if hand_class == HandThreatClass::Pawn && has_pawn_on_file(board, drop_color, drop_sq) {
                        continue;
                    }

                    // drop 後の attack 列挙
                    for_each_drop_attack(hand_class, drop_color, drop_sq, &occ, |to_sq| {
                        let target_pc = board.piece_on(to_sq);
                        if target_pc.is_none() {
                            return;
                        }
                        let target_pt = target_pc.piece_type;
                        let target_color = target_pc.color;

                        if target_pt == PieceType::King {
                            return;
                        }

                        let attacked_class = match ThreatClass::from_piece_type(target_pt) {
                            Some(c) => c,
                            None => return,
                        };

                        // --- STM perspective ---
                        let stm_drop_owner = if drop_color == stm { 0 } else { 1 };
                        let stm_attacked_side = if target_color == stm { 0 } else { 1 };
                        let stm_drop_n = normalize_sq(drop_sq, stm, stm_hm);
                        let stm_to_n = normalize_sq(to_sq, stm, stm_hm);
                        let stm_oriented = if stm == Color::Black { drop_color } else { drop_color.opponent() };
                        let stm_hand_idx = hand_threat_index(
                            stm_drop_owner,
                            hand_class,
                            stm_oriented,
                            stm_attacked_side,
                            attacked_class,
                            stm_drop_n,
                            stm_to_n,
                        );
                        debug_assert!(stm_hand_idx < HAND_THREAT_DIMENSIONS);
                        let stm_idx = HALFKA_HM_DIMENSIONS + stm_hand_idx;

                        // --- NSTM perspective ---
                        let nstm_drop_owner = if drop_color == nstm { 0 } else { 1 };
                        let nstm_attacked_side = if target_color == nstm { 0 } else { 1 };
                        let nstm_drop_n = normalize_sq(drop_sq, nstm, nstm_hm);
                        let nstm_to_n = normalize_sq(to_sq, nstm, nstm_hm);
                        let nstm_oriented = if nstm == Color::Black { drop_color } else { drop_color.opponent() };
                        let nstm_hand_idx = hand_threat_index(
                            nstm_drop_owner,
                            hand_class,
                            nstm_oriented,
                            nstm_attacked_side,
                            attacked_class,
                            nstm_drop_n,
                            nstm_to_n,
                        );
                        debug_assert!(nstm_hand_idx < HAND_THREAT_DIMENSIONS);
                        let nstm_idx = HALFKA_HM_DIMENSIONS + nstm_hand_idx;

                        f(stm_idx, nstm_idx);
                    });
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_threat_dimensions_value() {
        assert_eq!(HAND_THREAT_DIMENSIONS, 121_104);
    }

    #[test]
    fn test_hand_attacks_per_color_totals() {
        assert_eq!(HAND_ATTACKS_PER_COLOR, [72, 324, 112, 328, 416, 816, 1296]);
        assert_eq!(HAND_ATTACKS_PER_COLOR.iter().sum::<usize>(), 3_364);
    }

    #[test]
    fn test_hand_pair_base_monotone() {
        let mut prev: Option<usize> = None;
        for drop_owner in 0..2 {
            for hc in 0..HAND_NUM_CLASSES {
                for attacked_side in 0..2 {
                    for ac in 0..NUM_THREAT_CLASSES {
                        let idx = drop_owner * 126 + hc * 18 + attacked_side * 9 + ac;
                        let base = HAND_PAIR_BASE[idx];
                        if let Some(p) = prev {
                            assert!(base > p, "must be strictly increasing");
                        }
                        prev = Some(base);
                    }
                }
            }
        }
    }

    #[test]
    fn test_num_inputs() {
        let input = ShogiHalfKaHmHandThreat::new();
        assert_eq!(input.num_inputs(), HALFKA_HM_DIMENSIONS + 121_104);
    }

    #[test]
    fn test_drop_legality_black_pawn() {
        let p = HandThreatClass::Pawn;
        assert!(!is_legal_drop_rank(p, Color::Black, Square::new(0, 0)));
        assert!(is_legal_drop_rank(p, Color::Black, Square::new(0, 1)));
    }

    #[test]
    fn test_drop_legality_black_knight() {
        let n = HandThreatClass::Knight;
        assert!(!is_legal_drop_rank(n, Color::Black, Square::new(0, 0)));
        assert!(!is_legal_drop_rank(n, Color::Black, Square::new(0, 1)));
        assert!(is_legal_drop_rank(n, Color::Black, Square::new(0, 2)));
    }

    #[test]
    fn test_drop_legality_white_pawn() {
        let p = HandThreatClass::Pawn;
        assert!(!is_legal_drop_rank(p, Color::White, Square::new(0, 8)));
        assert!(is_legal_drop_rank(p, Color::White, Square::new(0, 7)));
    }

    #[test]
    fn test_hand_to_board_class_mapping() {
        assert_eq!(HandThreatClass::Pawn.as_board_class() as usize, ThreatClass::Pawn as usize);
        assert_eq!(HandThreatClass::Gold.as_board_class() as usize, ThreatClass::GoldLike as usize);
    }

    /// Cross-validation 用: 特定局面の HandThreat indices を sorted Vec で返す
    ///
    /// rshogi 側と bullet-shogi 側で同じ順序の (stm_idx, nstm_idx) ペアを生成し、
    /// 一致することを検証する。HalfKA_hm 側の indices は除外し、HandThreat 部分
    /// (offset >= HALFKA_HM_DIMENSIONS) のみを返す。
    fn collect_hand_threat_only(board: &ShogiBoard) -> Vec<(usize, usize)> {
        let input = ShogiHalfKaHmHandThreat::new();
        let mut pairs = Vec::new();
        input.map_hand_threat_features(board, |stm_idx, nstm_idx| {
            if stm_idx >= HALFKA_HM_DIMENSIONS {
                pairs.push((stm_idx - HALFKA_HM_DIMENSIONS, nstm_idx - HALFKA_HM_DIMENSIONS));
            }
        });
        pairs.sort();
        pairs
    }

    /// 最小テストケース: 先手玉=5九、後手玉=5一、先手飛車=5五、先手持ち駒=Pawn×1
    ///
    /// HandThreat feature は 1 件のみ active:
    /// - 先手の持ち駒歩を 5六 (raw 41) に打つと、5五 (raw 40) の先手飛車を攻撃
    /// - attacked_side: (先手視点) friend (Black Rook is friend from Black)
    /// - attacked_class: Rook
    fn build_minimal_pawn_drop_position() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        board.board[Square::new(4, 4).index()] = Piece::new(Color::Black, PieceType::Rook); // 5五飛
        // 先手の持ち駒に歩を 1 枚追加
        board.black_hand.set(PieceType::Pawn, 1);
        board
    }

    /// Cross-validation Golden vector 書き出し
    ///
    /// このテストは `cargo test test_write_hand_threat_golden -- --ignored` で
    /// 手動実行する。rshogi 側の cross-validation test が読み込む。
    ///
    /// 出力先: /tmp/hand_threat_golden_minimal.txt
    /// 形式: 各行に `stm_idx nstm_idx` (スペース区切り)、sorted
    #[test]
    #[ignore]
    fn test_write_hand_threat_golden() {
        use std::io::Write;
        let board = build_minimal_pawn_drop_position();
        let pairs = collect_hand_threat_only(&board);

        let mut file = std::fs::File::create("/tmp/hand_threat_golden_minimal.txt").expect("create golden file");
        for (stm, nstm) in &pairs {
            writeln!(file, "{} {}", stm, nstm).expect("write line");
        }
        eprintln!("Wrote {} pairs to /tmp/hand_threat_golden_minimal.txt", pairs.len());
        for (stm, nstm) in &pairs {
            eprintln!("  stm={} nstm={}", stm, nstm);
        }
    }

    /// 最小局面で HandThreat feature がちょうど 1 件 active であることを確認
    #[test]
    fn test_minimal_position_has_one_hand_threat() {
        let board = build_minimal_pawn_drop_position();
        let pairs = collect_hand_threat_only(&board);
        assert_eq!(pairs.len(), 1, "minimal position should produce exactly 1 hand threat feature");
    }

    /// Snapshot regression test for optimization verification
    ///
    /// DLSuisho15b_deduped_shuffled.bin から 1000 局面を読み、各局面の
    /// HandThreat feature (sorted) を snapshot ファイルに dump する。
    ///
    /// ## Usage
    ///
    /// ```bash
    /// # 最適化前に baseline snapshot を取る
    /// cd /mnt/nvme1/development/bullet-shogi
    /// cargo test -p bullet_lib --release test_snapshot_hand_threat_corpus -- --ignored --nocapture
    /// cp /tmp/hand_threat_snapshot.txt /tmp/hand_threat_snapshot_before.txt
    ///
    /// # 最適化実装
    ///
    /// # 最適化後に snapshot を再生成して diff
    /// cargo test -p bullet_lib --release test_snapshot_hand_threat_corpus -- --ignored --nocapture
    /// diff /tmp/hand_threat_snapshot_before.txt /tmp/hand_threat_snapshot.txt
    /// # 差分が 0 行なら bit-exact 同一
    /// ```
    ///
    /// 出力形式 (per position):
    /// ```
    /// pos N: K features
    ///   stm_offset nstm_offset  (HALFKA_HM_DIMENSIONS を差し引いた HandThreat 内部 index)
    ///   ...
    /// ```
    #[test]
    #[ignore]
    fn test_snapshot_hand_threat_corpus() {
        use crate::shogi::PackedSfenValue;
        use std::fs::File;
        use std::io::{Read, Write};

        const PACK_PATH: &str = "/mnt/nvme1/development/bullet-shogi/data/DLSuisho15b_deduped_shuffled.bin";
        const NUM_POSITIONS: usize = 1000;
        const SNAPSHOT_PATH: &str = "/tmp/hand_threat_snapshot.txt";

        let mut file = File::open(PACK_PATH).unwrap_or_else(|e| panic!("failed to open {PACK_PATH}: {e}"));

        let input = ShogiHalfKaHmHandThreat::new();
        let mut dump = String::new();
        let mut total_features = 0usize;
        let mut positions_with_features = 0usize;

        for pos_idx in 0..NUM_POSITIONS {
            let mut buf = [0u8; 40];
            if file.read_exact(&mut buf).is_err() {
                eprintln!("EOF at position {pos_idx}");
                break;
            }
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(&buf);

            // HandThreat features のみ抽出して sort
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            input.map_features(&psv, |stm_idx, nstm_idx| {
                if stm_idx >= HALFKA_HM_DIMENSIONS {
                    pairs.push((stm_idx - HALFKA_HM_DIMENSIONS, nstm_idx - HALFKA_HM_DIMENSIONS));
                }
            });
            pairs.sort();

            dump.push_str(&format!("pos {}: {} features\n", pos_idx, pairs.len()));
            for (stm_off, nstm_off) in &pairs {
                dump.push_str(&format!("  {} {}\n", stm_off, nstm_off));
            }

            total_features += pairs.len();
            if !pairs.is_empty() {
                positions_with_features += 1;
            }
        }

        let mut out = File::create(SNAPSHOT_PATH).unwrap_or_else(|e| panic!("failed to create {SNAPSHOT_PATH}: {e}"));
        out.write_all(dump.as_bytes()).expect("write snapshot");

        eprintln!(
            "Snapshot written: {} positions, {} with features, {} total features",
            NUM_POSITIONS, positions_with_features, total_features
        );
        eprintln!("File: {SNAPSHOT_PATH}");
    }

    #[test]
    fn test_startpos_no_hand_threats() {
        // 持ち駒 0 の startpos → HandThreat 特徴量は 0 件 (HalfKA 側のみ active)
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        let input = ShogiHalfKaHmHandThreat::new();
        let mut hand_indices_count = 0usize;
        input.map_hand_threat_features(&board, |stm_idx, _nstm_idx| {
            if stm_idx >= HALFKA_HM_DIMENSIONS {
                hand_indices_count += 1;
            }
        });
        assert_eq!(hand_indices_count, 0, "startpos should have no hand threats");
    }
}
