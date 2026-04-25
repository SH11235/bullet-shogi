//! HalfKA_hm + HandThreat (defensive 削減版) 連結特徴量
//!
//! HandThreat の次元を full pair (121,104) から `drop_owner=enemy` かつ
//! `attacked_side=friend` のみに絞った defensive 版 (30,276 次元)。
//!
//! ## 設計背景
//!
//! full pair 版は active 数が大きく、学習・推論ともコストが重い。defensive は
//! 「相手が drop して自分の駒を攻撃する脅威」という防御的意味論だけを符号化し、
//! active / 次元を 1/4 に削減する。詳細は rshogi リポジトリの
//! `docs/hand_threat_reduction_design.md` を参照。
//!
//! ## 非対称 emission
//!
//! defensive の active set は perspective 依存で非対称:
//! - STM 視点: "NSTM drop → STM target" のみ (nstm-drop 事象の数)
//! - NSTM 視点: "STM drop → NSTM target" のみ (stm-drop 事象の数)
//!
//! 同一局面でも `|STM_active| != |NSTM_active|` が通常。このため
//! `SparseInputType::map_features_split` を用いて `Option<usize>` でどちらか
//! 片側の feature だけを emit する。loader は両 chunk を独立に扱う。
//!
//! ## rshogi 実装との整合
//!
//! rshogi `crates/rshogi-core/src/nnue/hand_threat_features.rs` の
//! `hand-threat-defensive` feature と index formula を厳密一致させること。
//! 不一致は verify-nnue cross-validation で検出される。

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

/// HandThreat active features の上限 (defensive は full pair の 1/4 程度だが、
/// safety マージンを取って 512 とする)
const MAX_ACTIVE_HAND_THREAT_DEFENSIVE_FEATURES: usize = 512;

/// HalfKA_hm + HandThreat defensive 合計の max active
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + MAX_ACTIVE_HAND_THREAT_DEFENSIVE_FEATURES;

/// `HandThreatClass` → board `ThreatClass` マッピング
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
const HAND_ATTACKS_PER_COLOR: [usize; HAND_NUM_CLASSES] = [72, 324, 112, 328, 416, 816, 1296];

// =============================================================================
// HandThreatClass
// =============================================================================

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
// hand_pair_base テーブル (defensive)
// =============================================================================

/// defensive の pair 数
/// = 1 (drop_owner=enemy) × 7 (hand_class) × 1 (attacked_side=friend) × 9 (attacked_class)
const HAND_NUM_PAIRS: usize = HAND_NUM_CLASSES * NUM_THREAT_CLASSES; // 63

/// defensive: layout `hc * 9 + ac`
/// rshogi 側 `hand_pair_base` と厳密一致させる必要がある。
const fn build_hand_pair_base() -> ([usize; HAND_NUM_PAIRS], usize) {
    let mut table = [0usize; HAND_NUM_PAIRS];
    let mut cumulative = 0usize;
    let mut hc = 0usize;
    while hc < HAND_NUM_CLASSES {
        let mut ac = 0usize;
        while ac < NUM_THREAT_CLASSES {
            let idx = hc * NUM_THREAT_CLASSES + ac;
            table[idx] = cumulative;
            cumulative += HAND_ATTACKS_PER_COLOR[hc];
            ac += 1;
        }
        hc += 1;
    }
    (table, cumulative)
}

const HAND_PAIR_DATA: ([usize; HAND_NUM_PAIRS], usize) = build_hand_pair_base();

static HAND_PAIR_BASE: [usize; HAND_NUM_PAIRS] = HAND_PAIR_DATA.0;

/// HandThreat 総特徴量次元数 (defensive)
///
/// `1 × 7 × 1 × 9 × (HAND_ATTACKS_PER_COLOR の合計)` = 9 × 3,364 = **30,276**
pub const HAND_THREAT_DIMENSIONS: usize = HAND_PAIR_DATA.1;

const _HAND_THREAT_DIMENSIONS_CHECK: () = {
    assert!(HAND_THREAT_DIMENSIONS == 30_276, "HAND_THREAT_DIMENSIONS (defensive) must be 30,276");
};

/// defensive: (hc, ac) → base offset
#[inline]
fn hand_pair_base(hc: HandThreatClass, ac: ThreatClass) -> usize {
    HAND_PAIR_BASE[(hc as usize) * NUM_THREAT_CLASSES + ac as usize]
}

// =============================================================================
// Legal drop check
// =============================================================================

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

fn for_each_drop_attack<F: FnMut(Square)>(
    hand_class: HandThreatClass,
    color: Color,
    from: Square,
    occ: &Occupied,
    callback: F,
) {
    let pt = hand_class.as_piece_type();
    super::shogi_halfka_hm_threat::for_each_attack(pt, color, from, occ, callback);
}

// =============================================================================
// hand_threat_index (defensive)
// =============================================================================

/// defensive: drop_owner と attacked_side は fixed なので引数から外す。
///
/// `oriented_color` は現行 full pair と同じく perspective 依存で「drop 駒の
/// 方向性」を encode する。defensive では drop_color は常に perspective の
/// 敵なので、oriented_color も perspective の敵色で固定される。
#[inline]
fn hand_threat_index(
    hand_class: HandThreatClass,
    oriented_color: Color,
    attacked_class: ThreatClass,
    drop_sq_n: Square,
    attack_to_sq_n: Square,
) -> usize {
    let base = hand_pair_base(hand_class, attacked_class);
    let pattern = attack_pattern_id(hand_class.as_board_class(), oriented_color);
    let from_offset_table = &*FROM_OFFSET_TABLE;
    let from_off = from_offset_table.get(pattern, drop_sq_n);
    let attack_ord = ATTACK_ORDER_TABLE.get(pattern, drop_sq_n, attack_to_sq_n);
    debug_assert_ne!(
        attack_ord,
        AttackOrderTable::INVALID,
        "defensive hand_threat attack_order: to_sq {} not attacked by pattern {pattern} at {}",
        attack_to_sq_n.0,
        drop_sq_n.0
    );
    base + from_off + attack_ord as usize
}

// =============================================================================
// ShogiHalfKaHmHandThreatDefensive
// =============================================================================

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Default)]
pub struct ShogiHalfKaHmHandThreatDefensive;

impl ShogiHalfKaHmHandThreatDefensive {
    pub fn new() -> Self {
        Self
    }

    pub fn hand_threat_dimensions(&self) -> usize {
        HAND_THREAT_DIMENSIONS
    }
}

impl SparseInputType for ShogiHalfKaHmHandThreatDefensive {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS + HAND_THREAT_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_TOTAL
    }

    /// symmetric API (後方互換用 fallback)。通常は map_features_split を使う。
    /// このメソッドは「STM / NSTM の両側に同時 emit 可能な features のみ」を
    /// 出力する。defensive では HalfKA_hm 部分のみが対称で、HandThreat 部分は
    /// 非対称なのでこの方法ではエラー (unreachable) にする。
    ///
    /// bullet-shogi の loader は map_features_split を優先的に呼ぶため、
    /// このメソッドは通常の訓練経路からは呼ばれない。ただし snapshot test 等の
    /// 外部ツールが map_features を直接呼ぶ場合のために panic ではなく
    /// HalfKA 部分のみを emit する "best-effort" 挙動にする。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        self.map_halfka_hm_features(&board, f);
        // HandThreat は defensive では非対称なので symmetric f() では正確に
        // emit できない。呼び出し側は map_features_split を使うべき。
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        // HalfKA_hm 部分は対称なので両側 Some で emit
        self.map_halfka_hm_features(&board, |stm, nstm| f(Some(stm), Some(nstm)));
        // HandThreat 部分は非対称なので片側のみ emit
        self.map_hand_threat_features_defensive(&board, f);
    }

    fn shorthand(&self) -> String {
        format!("shogi-{}x45hm+handthreatdef", HALFKA_HM_DIMENSIONS + HAND_THREAT_DIMENSIONS)
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({}) + HandThreat defensive ({}) concatenated (asymmetric)",
            HALFKA_HM_DIMENSIONS, HAND_THREAT_DIMENSIONS
        )
    }
}

impl ShogiHalfKaHmHandThreatDefensive {
    /// HalfKA_hm 部分の feature 列挙 (既存 ShogiHalfKaHmHandThreat と同一ロジック)
    fn map_halfka_hm_features<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
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

        // 盤上の駒 (王以外)
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

        // 王の特徴量
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
    }

    /// HandThreat defensive 部分の非対称 feature 列挙
    ///
    /// 各 drop 事象について:
    /// - drop_color が nstm (= STM の敵) の場合: STM 視点の defensive index を
    ///   `Some(stm_hand_idx)` で emit。nstm 側は None。
    /// - drop_color が stm (= NSTM の敵) の場合: NSTM 視点の defensive index を
    ///   `Some(nstm_hand_idx)` で emit。stm 側は None。
    ///
    /// target は friend side のみ。
    fn map_hand_threat_features_defensive<F: FnMut(Option<usize>, Option<usize>)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king_sq = board.king_square(stm);
        let nstm_king_sq = board.king_square(nstm);
        if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
            return;
        }

        let stm_hm = is_hm_mirror(stm_king_sq, stm);
        let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);

        let occ = Occupied::from_board(board);

        // drop_color を両方処理するが、それぞれ異なる perspective 向けの
        // feature を emit する:
        //   drop_color = stm  → NSTM 視点の defensive (stm は nstm の敵)
        //   drop_color = nstm → STM 視点の defensive (nstm は stm の敵)
        for &drop_color in &[stm, nstm] {
            // この drop が各 perspective にとって defensive feature として
            // 有効かを判定する:
            //   drop_color == stm  → NSTM 視点で drop_owner=1 (stm = nstm の enemy)
            //                         STM 視点で drop_owner=0 (stm = stm の friend) → 無効
            //   drop_color == nstm → STM 視点で drop_owner=1 → 有効
            //                         NSTM 視点で drop_owner=0 → 無効
            //
            // つまり drop_color != perspective の時だけ defensive の対象になる。
            let emit_to_stm = drop_color != stm; // drop_color == nstm
            let emit_to_nstm = drop_color != nstm; // drop_color == stm
            debug_assert_ne!(
                emit_to_stm, emit_to_nstm,
                "defensive: a drop_color must belong to exactly one perspective"
            );

            for &hand_class in &ALL_HAND_THREAT_CLASSES {
                if board.hand(drop_color).count(hand_class.as_piece_type()) == 0 {
                    continue;
                }

                for drop_raw in 0..81u8 {
                    let drop_sq = Square(drop_raw);

                    if occ.is_occupied(drop_raw) {
                        continue;
                    }
                    if !is_legal_drop_rank(hand_class, drop_color, drop_sq) {
                        continue;
                    }
                    if hand_class == HandThreatClass::Pawn && has_pawn_on_file(board, drop_color, drop_sq) {
                        continue;
                    }

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

                        // defensive 条件: target_color = friend side of the perspective
                        // emit_to_stm (drop_color==nstm) → target must be stm
                        // emit_to_nstm (drop_color==stm) → target must be nstm
                        if emit_to_stm {
                            if target_color != stm {
                                return;
                            }
                            let drop_n = normalize_sq(drop_sq, stm, stm_hm);
                            let to_n = normalize_sq(to_sq, stm, stm_hm);
                            let oriented = if stm == Color::Black { drop_color } else { drop_color.opponent() };
                            let hand_idx = hand_threat_index(hand_class, oriented, attacked_class, drop_n, to_n);
                            debug_assert!(hand_idx < HAND_THREAT_DIMENSIONS);
                            f(Some(HALFKA_HM_DIMENSIONS + hand_idx), None);
                        } else {
                            // emit_to_nstm
                            if target_color != nstm {
                                return;
                            }
                            let drop_n = normalize_sq(drop_sq, nstm, nstm_hm);
                            let to_n = normalize_sq(to_sq, nstm, nstm_hm);
                            let oriented = if nstm == Color::Black { drop_color } else { drop_color.opponent() };
                            let hand_idx = hand_threat_index(hand_class, oriented, attacked_class, drop_n, to_n);
                            debug_assert!(hand_idx < HAND_THREAT_DIMENSIONS);
                            f(None, Some(HALFKA_HM_DIMENSIONS + hand_idx));
                        }
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
        assert_eq!(HAND_THREAT_DIMENSIONS, 30_276);
    }

    #[test]
    fn test_hand_pair_base_monotone() {
        let mut prev: Option<usize> = None;
        for hc in 0..HAND_NUM_CLASSES {
            for ac in 0..NUM_THREAT_CLASSES {
                let idx = hc * NUM_THREAT_CLASSES + ac;
                let base = HAND_PAIR_BASE[idx];
                if let Some(p) = prev {
                    assert!(base > p, "must be strictly increasing");
                }
                prev = Some(base);
            }
        }
    }

    #[test]
    fn test_num_inputs() {
        let input = ShogiHalfKaHmHandThreatDefensive::new();
        assert_eq!(input.num_inputs(), HALFKA_HM_DIMENSIONS + 30_276);
    }
}
