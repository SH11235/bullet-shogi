//! HalfKA_hm + Threat 連結特徴量
//!
//! HalfKA_hm (73,305 次元) と Threat (216,720 次元) を連結した
//! sparse input 型を提供する。
//!
//! ## 仕様
//!
//! - HalfKA_hm: 既存の `ShogiHalfKA_hm` と同一ロジック
//! - Threat: rshogi `threat_features.rs` と同一 index 計算
//! - 仕様メモ: `docs/threat_spec.md` (rshogi リポジトリ)

use std::sync::LazyLock;

use super::SparseInputType;
use super::shogi_halfka::{
    HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, halfka_index, is_hm_mirror, king_bonapiece, king_bucket, pack_bonapiece,
};
use super::shogi_threat_exclusion::ThreatProfile;
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    bona_piece::BonaPiece,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, PieceType, Square},
};

// =============================================================================
// Threat 定数
// =============================================================================

/// ThreatClass の数（King 除外）
pub(super) const NUM_THREAT_CLASSES: usize = 9;

/// active threat features の最大数（安全側の上限）
const MAX_ACTIVE_THREAT_FEATURES: usize = 320;

/// 最大アクティブ特徴数の合計（HalfKA_hm 40 + Threat 320）
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + MAX_ACTIVE_THREAT_FEATURES;

// =============================================================================
// ThreatClass
// =============================================================================

/// Threat 駒種分類（King 除外、9 family）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum ThreatClass {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    GoldLike = 4,
    Bishop = 5,
    Rook = 6,
    Horse = 7,
    Dragon = 8,
}

impl ThreatClass {
    /// PieceType から ThreatClass への変換。King は None。
    #[inline]
    pub(super) fn from_piece_type(pt: PieceType) -> Option<Self> {
        match pt {
            PieceType::Pawn => Some(Self::Pawn),
            PieceType::Lance => Some(Self::Lance),
            PieceType::Knight => Some(Self::Knight),
            PieceType::Silver => Some(Self::Silver),
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => Some(Self::GoldLike),
            PieceType::Bishop => Some(Self::Bishop),
            PieceType::Rook => Some(Self::Rook),
            PieceType::Horse => Some(Self::Horse),
            PieceType::Dragon => Some(Self::Dragon),
            PieceType::King | PieceType::None => None,
        }
    }
}

// =============================================================================
// 各クラスの空盤面利き数 (per color)
// =============================================================================

const ATTACKS_PER_COLOR: [usize; NUM_THREAT_CLASSES] = [
    72,   // Pawn
    324,  // Lance
    112,  // Knight
    328,  // Silver
    416,  // GoldLike
    816,  // Bishop
    1296, // Rook
    1104, // Horse
    1552, // Dragon
];

// =============================================================================
// pair_base テーブル
// =============================================================================

const NUM_PAIRS: usize = 2 * NUM_THREAT_CLASSES * 2 * NUM_THREAT_CLASSES; // 324

/// 除外された pair の sentinel 値
const EXCLUDED_PAIR_BASE: usize = usize::MAX;

/// 指定された profile で pair_base テーブルと THREAT_DIMENSIONS を構築 (runtime)
fn build_pair_base(profile: ThreatProfile) -> ([usize; NUM_PAIRS], usize) {
    let mut table = [0usize; NUM_PAIRS];
    let mut cumulative = 0usize;
    for attacker_side in 0..2 {
        for (ac, &attacks) in ATTACKS_PER_COLOR.iter().enumerate().take(NUM_THREAT_CLASSES) {
            for ds in 0..2 {
                for dc in 0..NUM_THREAT_CLASSES {
                    let idx = attacker_side * 162 + ac * 18 + ds * 9 + dc;
                    if profile.is_excluded(attacker_side, ac, ds, dc) {
                        table[idx] = EXCLUDED_PAIR_BASE;
                    } else {
                        table[idx] = cumulative;
                        cumulative += attacks;
                    }
                }
            }
        }
    }
    (table, cumulative)
}

/// pair_base テーブルから値を取得。除外された pair は None を返す。
#[inline]
fn lookup_pair_base(
    pair_base: &[usize; NUM_PAIRS],
    attacker_side: usize,
    ac: ThreatClass,
    attacked_side: usize,
    dc: ThreatClass,
) -> Option<usize> {
    let idx = attacker_side * 162 + (ac as usize) * 18 + attacked_side * 9 + dc as usize;
    let base = pair_base[idx];
    if base == EXCLUDED_PAIR_BASE { None } else { Some(base) }
}

// =============================================================================
// Attack pattern / from_offset / attack_order（色別 LUT）
// =============================================================================

/// Attack pattern の総数: 9(Black) + 5(White の方向性駒)
pub(super) const NUM_ATTACK_PATTERNS: usize = 14;

/// 方向性駒かどうか
#[inline]
pub(super) fn is_directional(class: ThreatClass) -> bool {
    matches!(
        class,
        ThreatClass::Pawn | ThreatClass::Lance | ThreatClass::Knight | ThreatClass::Silver | ThreatClass::GoldLike
    )
}

/// attack_pattern_id: 方向性駒は色別、非方向性駒は色不問
#[inline]
pub(super) fn attack_pattern_id(class: ThreatClass, oriented_color: Color) -> usize {
    if oriented_color == Color::White && is_directional(class) {
        NUM_THREAT_CLASSES + class as usize // 9..13
    } else {
        class as usize // 0..8
    }
}

// =============================================================================
// 空盤面利き計算（Bitboard を使わず、座標ベースで列挙）
// =============================================================================

/// 空盤面上の攻撃先マスを raw 値昇順で返す。
/// 返り値: (マスの配列, マス数)
pub(super) fn attacks_empty_board(class: ThreatClass, color: Color, from: Square) -> ([u8; 36], usize) {
    let mut targets = [0u8; 36];
    let mut count = 0;
    let file = from.file() as i8;
    let rank = from.rank() as i8;

    match class {
        ThreatClass::Pawn => {
            // 先手: 1マス前 (rank-1)、後手: 1マス後 (rank+1)
            let dr: i8 = if color == Color::Black { -1 } else { 1 };
            let r = rank + dr;
            if (0..9).contains(&r) {
                targets[count] = Square::new(file as u8, r as u8).0;
                count += 1;
            }
        }
        ThreatClass::Lance => {
            // 先手: rank-1, rank-2, ... 0 まで。後手: rank+1, ... 8 まで。
            // 空盤面なので遮蔽なし。
            if color == Color::Black {
                for r in (0..rank).rev() {
                    targets[count] = Square::new(file as u8, r as u8).0;
                    count += 1;
                }
            } else {
                for r in (rank + 1)..9 {
                    targets[count] = Square::new(file as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Knight => {
            // 先手: (file-1, rank-2), (file+1, rank-2)
            // 後手: (file-1, rank+2), (file+1, rank+2)
            let dr: i8 = if color == Color::Black { -2 } else { 2 };
            for df in [-1i8, 1] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Silver => {
            // 先手: (-1,-1),(0,-1),(1,-1),(-1,1),(1,1)
            // 後手: (-1,1),(0,1),(1,1),(-1,-1),(1,-1)
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 5] = [(-1, forward), (0, forward), (1, forward), (-1, -forward), (1, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::GoldLike => {
            // 先手: (-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(0,1)
            // 後手: (-1,1),(0,1),(1,1),(-1,0),(1,0),(0,-1)
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 6] = [(-1, forward), (0, forward), (1, forward), (-1, 0), (1, 0), (0, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Bishop => {
            // 4方向斜め、空盤面なので盤端まで
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
        }
        ThreatClass::Rook => {
            // 4方向直線、空盤面なので盤端まで
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
        }
        ThreatClass::Horse => {
            // 角 + 上下左右1マス (King moves)
            // 斜め（空盤面）
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
            // 上下左右1マス
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Dragon => {
            // 飛 + 斜め1マス
            // 直線（空盤面）
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
            // 斜め1マス
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
    }

    // raw 値昇順でソート（挿入ソート: count は最大 ~20 程度）
    for i in 1..count {
        let key = targets[i];
        let mut j = i;
        while j > 0 && targets[j - 1] > key {
            targets[j] = targets[j - 1];
            j -= 1;
        }
        targets[j] = key;
    }

    (targets, count)
}

// =============================================================================
// from_offset テーブル
// =============================================================================

/// 全 attack pattern の from_offset テーブル
pub(super) struct FromOffsetTable {
    data: [[usize; 81]; NUM_ATTACK_PATTERNS],
}

impl FromOffsetTable {
    pub(super) fn new() -> Self {
        let all_classes: [ThreatClass; NUM_THREAT_CLASSES] = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];

        let mut data = [[0usize; 81]; NUM_ATTACK_PATTERNS];

        for &class in &all_classes {
            // Black (先手) の from_offset
            {
                let pattern = class as usize;
                let mut cumulative = 0usize;
                for sq_raw in 0..81u8 {
                    data[pattern][sq_raw as usize] = cumulative;
                    let (_, cnt) = attacks_empty_board(class, Color::Black, Square(sq_raw));
                    cumulative += cnt;
                }
            }
            // White (後手) の方向性駒は別エントリ
            if is_directional(class) {
                let pattern = NUM_THREAT_CLASSES + class as usize;
                let mut cumulative = 0usize;
                for sq_raw in 0..81u8 {
                    data[pattern][sq_raw as usize] = cumulative;
                    let (_, cnt) = attacks_empty_board(class, Color::White, Square(sq_raw));
                    cumulative += cnt;
                }
            }
        }

        Self { data }
    }

    #[inline]
    pub(super) fn get(&self, pattern: usize, sq_n: Square) -> usize {
        self.data[pattern][sq_n.index()]
    }
}

/// 遅延初期化された `FromOffsetTable` シングルトン
pub(super) static FROM_OFFSET_TABLE: LazyLock<FromOffsetTable> = LazyLock::new(FromOffsetTable::new);

// =============================================================================
// attack_order LUT (O(1) lookup)
// =============================================================================

/// attack_order の事前計算 LUT (rshogi の ATTACK_ORDER_TABLE と同設計)
///
/// `data[attack_pattern][from_sq][to_sq]` = 空盤面上で from_sq の駒が to_sq を攻撃する際の
/// 0-indexed 順位 (raw 昇順)。攻撃しない (from, to) ペアは `INVALID` (u8::MAX)。
///
/// サイズ: 14 × 81 × 81 = 91,854 bytes ≈ 90 KiB
///
/// LazyLock で初回アクセス時に 1 度だけ構築。以降は `get()` が O(1) lookup。
pub(super) struct AttackOrderTable {
    data: [[[u8; 81]; 81]; NUM_ATTACK_PATTERNS],
}

impl AttackOrderTable {
    pub(super) const INVALID: u8 = u8::MAX;

    pub(super) fn new() -> Self {
        let all_classes: [ThreatClass; NUM_THREAT_CLASSES] = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];

        let mut data = [[[Self::INVALID; 81]; 81]; NUM_ATTACK_PATTERNS];
        for &class in &all_classes {
            // Black (先手) pattern
            {
                let pattern = class as usize;
                Self::fill_pattern(&mut data[pattern], class, Color::Black);
            }
            // White (後手) 方向性駒は別エントリ
            if is_directional(class) {
                let pattern = NUM_THREAT_CLASSES + class as usize;
                Self::fill_pattern(&mut data[pattern], class, Color::White);
            }
        }
        Self { data }
    }

    fn fill_pattern(table: &mut [[u8; 81]; 81], class: ThreatClass, color: Color) {
        for from_raw in 0..81u8 {
            let (targets, count) = attacks_empty_board(class, color, Square(from_raw));
            for (order, &to_raw) in targets.iter().take(count).enumerate() {
                table[from_raw as usize][to_raw as usize] = order as u8;
            }
        }
    }

    /// O(1) で attack_order を取得
    #[inline]
    pub(super) fn get(&self, pattern: usize, from_sq: Square, to_sq: Square) -> u8 {
        self.data[pattern][from_sq.0 as usize][to_sq.0 as usize]
    }
}

/// 遅延初期化された `AttackOrderTable` シングルトン
pub(super) static ATTACK_ORDER_TABLE: LazyLock<AttackOrderTable> = LazyLock::new(AttackOrderTable::new);

// =============================================================================
// attack_order 計算 (legacy、テスト互換用)
// =============================================================================

/// 空盤面上で from_sq の駒が to_sq を攻撃するときの、raw 昇順での順位
///
/// **注意**: ホットパスでは `ATTACK_ORDER_TABLE.get(...)` を直接使うこと。
/// この関数は毎回 `attacks_empty_board` を再計算するため遅い。
/// 既存のテストコードと `threat_index` 内の legacy path との互換維持のためだけに残している。
#[allow(dead_code)]
pub(super) fn compute_attack_order(class: ThreatClass, color: Color, from_sq: Square, to_sq: Square) -> usize {
    let (targets, count) = attacks_empty_board(class, color, from_sq);
    let to_raw = to_sq.0;
    for (i, &target) in targets.iter().enumerate().take(count) {
        if target == to_raw {
            return i;
        }
    }
    panic!("attack_order: to_sq {} is not attacked by {:?} ({:?}) at {}", to_sq.0, class, color, from_sq.0);
}

// =============================================================================
// 実盤面上の利き計算
// =============================================================================

/// occupied bitset: 81 マス分のビットマップ
pub(super) struct Occupied {
    bits: [u64; 2], // bits[0]: sq 0..63, bits[1]: sq 64..80
}

impl Occupied {
    pub(super) fn from_board(board: &ShogiBoard) -> Self {
        let mut bits = [0u64; 2];
        for sq in 0..81u8 {
            if board.board[sq as usize].is_some() {
                if sq < 64 {
                    bits[0] |= 1u64 << sq;
                } else {
                    bits[1] |= 1u64 << (sq - 64);
                }
            }
        }
        Self { bits }
    }

    #[inline]
    pub(super) fn is_occupied(&self, sq: u8) -> bool {
        if sq < 64 { (self.bits[0] >> sq) & 1 != 0 } else { (self.bits[1] >> (sq - 64)) & 1 != 0 }
    }
}

/// 実盤面上の攻撃先マスを列挙し、コールバックを呼ぶ
pub(super) fn for_each_attack<F: FnMut(Square)>(
    pt: PieceType,
    color: Color,
    from: Square,
    occ: &Occupied,
    mut callback: F,
) {
    let file = from.file() as i8;
    let rank = from.rank() as i8;

    match pt {
        PieceType::Pawn => {
            let dr: i8 = if color == Color::Black { -1 } else { 1 };
            let r = rank + dr;
            if (0..9).contains(&r) {
                callback(Square::new(file as u8, r as u8));
            }
        }
        PieceType::Lance => {
            if color == Color::Black {
                for r in (0..rank).rev() {
                    let sq = Square::new(file as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                }
            } else {
                for r in (rank + 1)..9 {
                    let sq = Square::new(file as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                }
            }
        }
        PieceType::Knight => {
            let dr: i8 = if color == Color::Black { -2 } else { 2 };
            for df in [-1i8, 1] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Silver => {
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 5] = [(-1, forward), (0, forward), (1, forward), (-1, -forward), (1, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Gold | PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 6] = [(-1, forward), (0, forward), (1, forward), (-1, 0), (1, 0), (0, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Bishop => {
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
        }
        PieceType::Rook => {
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
        }
        PieceType::Horse => {
            // 角行き（スライダー）
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
            // 上下左右1マス
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Dragon => {
            // 飛行き（スライダー）
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
            // 斜め1マス
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        // King と None は threat に含まない
        PieceType::King | PieceType::None => {}
    }
}

// =============================================================================
// マス正規化
// =============================================================================

/// マスを perspective 基準 + HM mirror で正規化
#[inline]
pub(super) fn normalize_sq(sq: Square, perspective: Color, hm_mirror: bool) -> Square {
    let sq_n = if perspective == Color::Black { sq } else { sq.inverse() };
    if hm_mirror { sq_n.mirror_file() } else { sq_n }
}

// =============================================================================
// Threat index 計算
// =============================================================================

/// Threat index 計算用のパラメータ
struct ThreatParams {
    attacker_side: usize,
    attacker_class: ThreatClass,
    oriented_color: Color,
    attacked_side: usize,
    attacked_class: ThreatClass,
    from_sq_n: Square,
    to_sq_n: Square,
}

/// Threat index を計算する。除外された pair は None を返す。
///
/// `attack_order` は O(1) LUT lookup (`ATTACK_ORDER_TABLE`) を使用する。
#[inline]
fn threat_index(
    params: &ThreatParams,
    pair_base_table: &[usize; NUM_PAIRS],
    from_offset_table: &FromOffsetTable,
) -> Option<usize> {
    let base = lookup_pair_base(
        pair_base_table,
        params.attacker_side,
        params.attacker_class,
        params.attacked_side,
        params.attacked_class,
    )?;
    let pattern = attack_pattern_id(params.attacker_class, params.oriented_color);
    let from_off = from_offset_table.get(pattern, params.from_sq_n);
    let attack_ord = ATTACK_ORDER_TABLE.get(pattern, params.from_sq_n, params.to_sq_n);
    debug_assert_ne!(
        attack_ord,
        AttackOrderTable::INVALID,
        "attack_order: to_sq {} not attacked by pattern {pattern} at {}",
        params.to_sq_n.0,
        params.from_sq_n.0
    );
    Some(base + from_off + attack_ord as usize)
}

// =============================================================================
// ShogiHalfKaHmThreat
// =============================================================================

/// HalfKA_hm + Threat 連結特徴量
///
/// `SparseInputType` を実装し、HalfKA_hm 特徴量と Threat 特徴量を
/// 連結した sparse input として提供する。
///
/// `ThreatProfile` で除外 pair を runtime 選択する。
#[allow(non_camel_case_types)]
#[derive(Clone, Debug)]
pub struct ShogiHalfKaHmThreat {
    profile: ThreatProfile,
    pair_base: Box<[usize; NUM_PAIRS]>,
    threat_dims: usize,
}

impl ShogiHalfKaHmThreat {
    /// 指定 profile で構築
    pub fn new(profile: ThreatProfile) -> Self {
        let (pair_base, threat_dims) = build_pair_base(profile);
        Self { profile, pair_base: Box::new(pair_base), threat_dims }
    }

    /// Threat 特徴量の次元数
    pub fn threat_dimensions(&self) -> usize {
        self.threat_dims
    }

    /// ThreatProfile を取得
    pub fn profile(&self) -> ThreatProfile {
        self.profile
    }
}

impl SparseInputType for ShogiHalfKaHmThreat {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS + self.threat_dims
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_TOTAL
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        self.map_threat_features(&board, f);
    }

    fn shorthand(&self) -> String {
        format!("shogi-{}x45hm+threat", HALFKA_HM_DIMENSIONS + self.threat_dims)
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({}) + Threat ({}, profile={}) concatenated",
            HALFKA_HM_DIMENSIONS, self.threat_dims, self.profile
        )
    }
}

// =============================================================================
// 特徴量列挙
// =============================================================================

impl ShogiHalfKaHmThreat {
    /// HalfKA_hm + Threat の特徴量を列挙する
    fn map_threat_features<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king_sq = board.king_square(stm);
        let nstm_king_sq = board.king_square(nstm);
        if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
            return;
        }

        // -------------------------------------------------------
        // Part 1: HalfKA_hm 特徴量（既存ロジック複製）
        // -------------------------------------------------------

        let stm_kb = king_bucket(stm_king_sq, stm);
        let stm_hm = is_hm_mirror(stm_king_sq, stm);
        let nstm_kb = king_bucket(nstm_king_sq, nstm);
        let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);

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
        // Part 2: Threat 特徴量
        // -------------------------------------------------------

        let from_offset_table = &*FROM_OFFSET_TABLE;
        let occ = Occupied::from_board(board);

        // STM perspective
        let stm_friend = stm;

        // NSTM perspective
        let nstm_friend = nstm;

        // 全盤上駒を列挙して threat pair を生成
        for sq_raw in 0..81u8 {
            let from_sq = Square(sq_raw);
            let pc = board.piece_on(from_sq);
            if pc.is_none() {
                continue;
            }
            let pt = pc.piece_type;
            let attacker_color = pc.color;

            // King は除外
            if pt == PieceType::King {
                continue;
            }

            let attacker_class = match ThreatClass::from_piece_type(pt) {
                Some(c) => c,
                None => continue,
            };

            // 実盤面上の攻撃先を列挙
            for_each_attack(pt, attacker_color, from_sq, &occ, |to_sq| {
                let target_pc = board.piece_on(to_sq);
                if target_pc.is_none() {
                    return;
                }
                let target_pt = target_pc.piece_type;
                let target_color = target_pc.color;

                // King は除外
                if target_pt == PieceType::King {
                    return;
                }

                let attacked_class = match ThreatClass::from_piece_type(target_pt) {
                    Some(c) => c,
                    None => return,
                };

                // --- STM perspective ---
                let stm_attacker_side = if attacker_color == stm_friend { 0 } else { 1 };
                let stm_attacked_side = if target_color == stm_friend { 0 } else { 1 };
                let stm_from_n = normalize_sq(from_sq, stm, stm_hm);
                let stm_to_n = normalize_sq(to_sq, stm, stm_hm);
                let stm_oriented_color = if stm == Color::Black { attacker_color } else { attacker_color.opponent() };
                let stm_threat_idx = threat_index(
                    &ThreatParams {
                        attacker_side: stm_attacker_side,
                        attacker_class,
                        oriented_color: stm_oriented_color,
                        attacked_side: stm_attacked_side,
                        attacked_class,
                        from_sq_n: stm_from_n,
                        to_sq_n: stm_to_n,
                    },
                    &self.pair_base,
                    from_offset_table,
                );
                let Some(stm_threat_idx) = stm_threat_idx else {
                    return; // excluded pair
                };
                debug_assert!(stm_threat_idx < self.threat_dims);
                let stm_idx = HALFKA_HM_DIMENSIONS + stm_threat_idx;

                // --- NSTM perspective ---
                let nstm_attacker_side = if attacker_color == nstm_friend { 0 } else { 1 };
                let nstm_attacked_side = if target_color == nstm_friend { 0 } else { 1 };
                let nstm_from_n = normalize_sq(from_sq, nstm, nstm_hm);
                let nstm_to_n = normalize_sq(to_sq, nstm, nstm_hm);
                let nstm_oriented_color = if nstm == Color::Black { attacker_color } else { attacker_color.opponent() };
                let nstm_threat_idx = threat_index(
                    &ThreatParams {
                        attacker_side: nstm_attacker_side,
                        attacker_class,
                        oriented_color: nstm_oriented_color,
                        attacked_side: nstm_attacked_side,
                        attacked_class,
                        from_sq_n: nstm_from_n,
                        to_sq_n: nstm_to_n,
                    },
                    &self.pair_base,
                    from_offset_table,
                );
                let Some(nstm_threat_idx) = nstm_threat_idx else {
                    return; // excluded pair
                };
                debug_assert!(nstm_threat_idx < self.threat_dims);
                let nstm_idx = HALFKA_HM_DIMENSIONS + nstm_threat_idx;

                f(stm_idx, nstm_idx);
            });
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::PieceType;

    fn full_input() -> ShogiHalfKaHmThreat {
        ShogiHalfKaHmThreat::new(ThreatProfile::Full)
    }

    #[test]
    fn test_total_dimensions() {
        let input = full_input();
        assert_eq!(input.num_inputs(), HALFKA_HM_DIMENSIONS + 216_720);
    }

    #[test]
    fn test_cross_side_dimensions() {
        let input = ShogiHalfKaHmThreat::new(ThreatProfile::CrossSide);
        assert_eq!(input.threat_dimensions(), 96_320);
    }

    #[test]
    fn test_max_active() {
        let input = full_input();
        assert_eq!(input.max_active(), 40 + 320);
    }

    #[test]
    fn test_pair_base_dimensions() {
        let input = full_input();
        // 最後の non-excluded pair の base + attacks_per_color == threat_dims
        let mut last_base = 0usize;
        let mut last_ac = 0usize;
        for (i, &base) in input.pair_base.iter().enumerate() {
            if base != EXCLUDED_PAIR_BASE && base >= last_base {
                last_base = base;
                last_ac = (i % 162) / 18;
            }
        }
        assert_eq!(last_base + ATTACKS_PER_COLOR[last_ac], input.threat_dimensions());
    }

    #[test]
    fn test_attacks_per_color_totals() {
        let all_classes = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];
        for (i, &class) in all_classes.iter().enumerate() {
            let total: usize = (0..81u8)
                .map(|sq| {
                    let (_, cnt) = attacks_empty_board(class, Color::Black, Square(sq));
                    cnt
                })
                .sum();
            assert_eq!(total, ATTACKS_PER_COLOR[i], "{:?}: expected {}, got {}", class, ATTACKS_PER_COLOR[i], total);
        }
    }

    #[test]
    fn test_from_offset_pawn() {
        let table = FromOffsetTable::new();
        let pattern = attack_pattern_id(ThreatClass::Pawn, Color::Black);
        // sq=0 (file=0, rank=0): offset=0
        assert_eq!(table.get(pattern, Square(0)), 0);
        // sq=1 (file=0, rank=1): offset=0 (sq=0 has 0 attacks for Black pawn at rank=0)
        assert_eq!(table.get(pattern, Square(1)), 0);
        // sq=2 (file=0, rank=2): offset=1 (sq=1 has 1 attack)
        assert_eq!(table.get(pattern, Square(2)), 1);
    }

    #[test]
    fn test_from_offset_rook() {
        let table = FromOffsetTable::new();
        let pattern = attack_pattern_id(ThreatClass::Rook, Color::Black);
        // Rook: 全マスで attacks=16
        for sq_raw in 0..81u8 {
            assert_eq!(table.get(pattern, Square(sq_raw)), 16 * sq_raw as usize);
        }
    }

    #[test]
    fn test_attack_order_rook_center() {
        // Rook at sq=40 (5五): 16 攻撃先
        let (targets, count) = attacks_empty_board(ThreatClass::Rook, Color::Black, Square(40));
        assert_eq!(count, 16);

        // 最初の攻撃先は order=0
        let first_sq = Square(targets[0]);
        assert_eq!(compute_attack_order(ThreatClass::Rook, Color::Black, Square(40), first_sq), 0);
    }

    #[test]
    fn test_threat_index_range() {
        let input = full_input();
        let from_offset_table = FromOffsetTable::new();
        let all_classes = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];

        for &class in &all_classes {
            for &oriented_color in &[Color::Black, Color::White] {
                for sq_raw in 0..81u8 {
                    let sq = Square(sq_raw);
                    let (targets, count) = attacks_empty_board(class, oriented_color, sq);
                    for &target_raw in targets.iter().take(count) {
                        let to = Square(target_raw);
                        for as_ in 0..2usize {
                            for ds in 0..2usize {
                                for &dc_class in &all_classes {
                                    let idx = threat_index(
                                        &ThreatParams {
                                            attacker_side: as_,
                                            attacker_class: class,
                                            oriented_color,
                                            attacked_side: ds,
                                            attacked_class: dc_class,
                                            from_sq_n: sq,
                                            to_sq_n: to,
                                        },
                                        &input.pair_base,
                                        &from_offset_table,
                                    );
                                    if let Some(idx) = idx {
                                        assert!(
                                            idx < input.threat_dimensions(),
                                            "index {} out of range for class={:?} color={:?} sq={} to={}",
                                            idx,
                                            class,
                                            oriented_color,
                                            sq.0,
                                            to.0
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_map_features_startpos() {
        // 初期局面を手動で構築
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };

        // 玉
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 先手の歩 (7段)
        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
        }
        // 後手の歩 (3段)
        for file in 0..9u8 {
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        // 角・飛
        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop); // 8八角
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook); // 2八飛
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop); // 2二角
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook); // 8二飛

        // 香
        board.board[Square::new(0, 8).index()] = Piece::new(Color::Black, PieceType::Lance); // 1九香
        board.board[Square::new(8, 8).index()] = Piece::new(Color::Black, PieceType::Lance); // 9九香
        board.board[Square::new(0, 0).index()] = Piece::new(Color::White, PieceType::Lance); // 1一香
        board.board[Square::new(8, 0).index()] = Piece::new(Color::White, PieceType::Lance); // 9一香

        // 桂
        board.board[Square::new(1, 8).index()] = Piece::new(Color::Black, PieceType::Knight); // 2九桂
        board.board[Square::new(7, 8).index()] = Piece::new(Color::Black, PieceType::Knight); // 8九桂
        board.board[Square::new(1, 0).index()] = Piece::new(Color::White, PieceType::Knight); // 2一桂
        board.board[Square::new(7, 0).index()] = Piece::new(Color::White, PieceType::Knight); // 8一桂

        // 銀
        board.board[Square::new(2, 8).index()] = Piece::new(Color::Black, PieceType::Silver); // 3九銀
        board.board[Square::new(6, 8).index()] = Piece::new(Color::Black, PieceType::Silver); // 7九銀
        board.board[Square::new(2, 0).index()] = Piece::new(Color::White, PieceType::Silver); // 3一銀
        board.board[Square::new(6, 0).index()] = Piece::new(Color::White, PieceType::Silver); // 7一銀

        // 金
        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold); // 4九金
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold); // 6九金
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold); // 4一金
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold); // 6一金

        let mut halfka_count = 0usize;
        let mut threat_count = 0usize;

        let input = full_input();
        let total_dims = input.num_inputs();
        input.map_threat_features(&board, |stm_idx, nstm_idx| {
            if stm_idx < HALFKA_HM_DIMENSIONS {
                halfka_count += 1;
            } else {
                threat_count += 1;
                assert!(stm_idx < total_dims, "stm threat index out of range: {stm_idx}");
            }
            if nstm_idx < HALFKA_HM_DIMENSIONS {
                // HalfKA 部分
            } else {
                assert!(nstm_idx < total_dims, "nstm threat index out of range: {nstm_idx}");
            }
        });

        // HalfKA: 38駒(王以外) + 2(両玉) = 40
        assert_eq!(halfka_count, 40);

        // Threat: 初期局面では threat pair が存在する（歩 vs 歩の対面等）
        assert!(threat_count > 0, "threat features should be non-empty in startpos");
    }

    #[test]
    fn test_map_features_sq_nb_guard() {
        // 片玉データはスキップされる
        let board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::NONE,
            ..Default::default()
        };

        let mut count = 0;
        let input = full_input();
        input.map_threat_features(&board, |_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_shorthand() {
        let input = full_input();
        let expected = format!("shogi-{}x45hm+threat", input.num_inputs());
        assert_eq!(input.shorthand(), expected);
    }

    #[test]
    fn test_normalize_sq() {
        // Black perspective, no mirror: そのまま
        let sq = Square::new(4, 4); // 5五
        assert_eq!(normalize_sq(sq, Color::Black, false), sq);

        // Black perspective, mirror: file 反転
        assert_eq!(normalize_sq(sq, Color::Black, true), sq.mirror_file());

        // White perspective, no mirror: inverse
        assert_eq!(normalize_sq(sq, Color::White, false), sq.inverse());

        // White perspective, mirror: inverse + mirror
        assert_eq!(normalize_sq(sq, Color::White, true), sq.inverse().mirror_file());
    }

    /// Canonical test vector: rshogi の threat_features.rs と同一の初期局面 threat index を検証
    /// この値が変わったら rshogi 側も同時に更新すること。
    /// Profile 0 (full) のみ有効（他の profile では index が変わる）。
    #[test]
    #[cfg(not(any(feature = "threat-profile-same-class", feature = "threat-profile-same-class-major-pawn",)))]
    fn test_canonical_startpos_threat_indices() {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };

        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop);
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop);
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook);

        board.board[Square::new(0, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(8, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(0, 0).index()] = Piece::new(Color::White, PieceType::Lance);
        board.board[Square::new(8, 0).index()] = Piece::new(Color::White, PieceType::Lance);

        board.board[Square::new(1, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(7, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(1, 0).index()] = Piece::new(Color::White, PieceType::Knight);
        board.board[Square::new(7, 0).index()] = Piece::new(Color::White, PieceType::Knight);

        board.board[Square::new(2, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(6, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(2, 0).index()] = Piece::new(Color::White, PieceType::Silver);
        board.board[Square::new(6, 0).index()] = Piece::new(Color::White, PieceType::Silver);

        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold);

        let mut stm_threat_indices = Vec::new();
        let input = full_input();
        input.map_threat_features(&board, |stm_idx, _nstm_idx| {
            if stm_idx >= HALFKA_HM_DIMENSIONS {
                stm_threat_indices.push(stm_idx - HALFKA_HM_DIMENSIONS);
            }
        });
        stm_threat_indices.sort();

        #[rustfmt::skip]
        let expected: Vec<usize> = vec![
            1330, 1618, 7147, 7148, 7231, 7232, 11047, 11213,
            16475, 16578, 23268, 23270, 24087, 25717, 37487, 40080,
            43974, 112573, 112861, 116503, 116504, 116587, 116588,
            122160, 122650, 128533, 128636, 138321, 138323, 139136,
            140770, 158280, 160871, 164753,
        ];

        assert_eq!(
            stm_threat_indices,
            expected,
            "Canonical mismatch with rshogi! count: bullet={} vs expected={}",
            stm_threat_indices.len(),
            expected.len()
        );
    }
}
