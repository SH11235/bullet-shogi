//! .pack ファイルローダー
//!
//! GenSfen が出力する .pack 形式（可変長対局棋譜）を読み取り、
//! `PackedSfenValue` レコードとしてバッチ供給するデータローダー。
//!
//! ## .pack フォーマット
//!
//! 各対局:
//! ```text
//! [start_flag: u8] — 1=平手, 0=任意局面
//! if 0: [HuffmanCodedPos: 32byte][game_ply: u16]
//! 繰り返し: [move16: u16][eval: i16]
//! [終局マーカー: u16 (from==to)] [reason: u8]
//! ```
//!
//! ## パイプライン (4段)
//!
//! 1. Reader:  .pack ファイルを読み、対局バイト列を切り出す
//! 2. Expander: 対局 → 個別局面 (`PackedSfenValue`) に展開、フィルタ適用
//! 3. Shuffle: シャッフルバッファに蓄積し Fisher-Yates シャッフル
//! 4. Batch:   batch_size に分割してコールバックへ

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::sync::mpsc;

use crate::shogi::packed_sfen::PackedSfenValue;
use crate::shogi::types::{Color, Hand, Piece, PieceType};

use super::DataLoader;
use super::rng::SimpleRand;

// =============================================================================
// Huffman テーブル (YaneuraOu PSfen 形式 / Apery HCP 形式)
// =============================================================================

/// Huffman 符号エントリ
#[derive(Clone, Copy)]
struct HuffmanCode {
    code: u8,
    bits: u8,
}

/// YaneuraOu PackedSfen 形式のハフマンテーブル
///
/// インデックス: 0=空, 1=歩, 2=香, 3=桂, 4=銀, 5=角, 6=飛, 7=金
const YO_HUFFMAN_TABLE: [HuffmanCode; 8] = [
    HuffmanCode { code: 0x00, bits: 1 }, // NO_PIECE
    HuffmanCode { code: 0x01, bits: 2 }, // PAWN
    HuffmanCode { code: 0x03, bits: 4 }, // LANCE
    HuffmanCode { code: 0x0b, bits: 4 }, // KNIGHT
    HuffmanCode { code: 0x07, bits: 4 }, // SILVER
    HuffmanCode { code: 0x1f, bits: 6 }, // BISHOP
    HuffmanCode { code: 0x3f, bits: 6 }, // ROOK
    HuffmanCode { code: 0x0f, bits: 5 }, // GOLD
];

/// Apery/cshogi HCP 形式のハフマンテーブル
///
/// YO とは KNIGHT(0x07) と SILVER(0x0b) のコードが入れ替わっている。
const HCP_HUFFMAN_TABLE: [HuffmanCode; 8] = [
    HuffmanCode { code: 0x00, bits: 1 }, // NO_PIECE
    HuffmanCode { code: 0x01, bits: 2 }, // PAWN
    HuffmanCode { code: 0x03, bits: 4 }, // LANCE
    HuffmanCode { code: 0x07, bits: 4 }, // KNIGHT ← YO=0x0b
    HuffmanCode { code: 0x0b, bits: 4 }, // SILVER ← YO=0x07
    HuffmanCode { code: 0x1f, bits: 6 }, // BISHOP
    HuffmanCode { code: 0x3f, bits: 6 }, // ROOK
    HuffmanCode { code: 0x0f, bits: 5 }, // GOLD
];

/// HCP 手駒/駒箱の prefix-free コードテーブル (cshogi 準拠)
#[derive(Clone, Copy)]
struct HcpHandCode {
    code: u8,
    bits: u8,
    piece_idx: usize,
    color: Option<Color>,
    is_piecebox: bool,
}

const HCP_HAND_TABLE: [HcpHandCode; 21] = [
    // 先手手駒
    HcpHandCode { code: 0x00, bits: 3, piece_idx: 1, color: Some(Color::Black), is_piecebox: false }, // 歩
    HcpHandCode { code: 0x01, bits: 5, piece_idx: 2, color: Some(Color::Black), is_piecebox: false }, // 香
    HcpHandCode { code: 0x03, bits: 5, piece_idx: 3, color: Some(Color::Black), is_piecebox: false }, // 桂
    HcpHandCode { code: 0x05, bits: 5, piece_idx: 4, color: Some(Color::Black), is_piecebox: false }, // 銀
    HcpHandCode { code: 0x07, bits: 5, piece_idx: 7, color: Some(Color::Black), is_piecebox: false }, // 金
    HcpHandCode { code: 0x1f, bits: 7, piece_idx: 5, color: Some(Color::Black), is_piecebox: false }, // 角
    HcpHandCode { code: 0x3f, bits: 7, piece_idx: 6, color: Some(Color::Black), is_piecebox: false }, // 飛
    // 後手手駒
    HcpHandCode { code: 0x04, bits: 3, piece_idx: 1, color: Some(Color::White), is_piecebox: false }, // 歩
    HcpHandCode { code: 0x11, bits: 5, piece_idx: 2, color: Some(Color::White), is_piecebox: false }, // 香
    HcpHandCode { code: 0x13, bits: 5, piece_idx: 3, color: Some(Color::White), is_piecebox: false }, // 桂
    HcpHandCode { code: 0x15, bits: 5, piece_idx: 4, color: Some(Color::White), is_piecebox: false }, // 銀
    HcpHandCode { code: 0x17, bits: 5, piece_idx: 7, color: Some(Color::White), is_piecebox: false }, // 金
    HcpHandCode { code: 0x5f, bits: 7, piece_idx: 5, color: Some(Color::White), is_piecebox: false }, // 角
    HcpHandCode { code: 0x7f, bits: 7, piece_idx: 6, color: Some(Color::White), is_piecebox: false }, // 飛
    // 駒箱
    HcpHandCode { code: 0x02, bits: 3, piece_idx: 1, color: None, is_piecebox: true }, // 歩
    HcpHandCode { code: 0x09, bits: 5, piece_idx: 2, color: None, is_piecebox: true }, // 香
    HcpHandCode { code: 0x0b, bits: 5, piece_idx: 3, color: None, is_piecebox: true }, // 桂
    HcpHandCode { code: 0x0d, bits: 5, piece_idx: 4, color: None, is_piecebox: true }, // 銀
    HcpHandCode { code: 0x1d, bits: 5, piece_idx: 7, color: None, is_piecebox: true }, // 金
    HcpHandCode { code: 0x0f, bits: 7, piece_idx: 5, color: None, is_piecebox: true }, // 角
    HcpHandCode { code: 0x2f, bits: 7, piece_idx: 6, color: None, is_piecebox: true }, // 飛
];

// =============================================================================
// ビットストリーム (読み取り / 書き込み)
// =============================================================================

/// LSB-first ビットストリーム (読み取り)
struct BitReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    #[inline]
    fn cursor(&self) -> usize {
        self.cursor
    }

    #[inline]
    fn read_bit(&mut self) -> u8 {
        let byte_idx = self.cursor / 8;
        if byte_idx >= self.data.len() {
            return 0;
        }
        let bit_idx = self.cursor & 7;
        self.cursor += 1;
        (self.data[byte_idx] >> bit_idx) & 1
    }

    #[inline]
    fn read_bits(&mut self, n: usize) -> u32 {
        let mut result = 0u32;
        for i in 0..n {
            result |= (self.read_bit() as u32) << i;
        }
        result
    }
}

/// LSB-first ビットストリーム (書き込み)
struct BitWriter {
    data: [u8; 32],
    cursor: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self { data: [0u8; 32], cursor: 0 }
    }

    #[inline]
    fn write_bit(&mut self, b: bool) {
        if self.cursor < 256 {
            let byte_idx = self.cursor / 8;
            let bit_idx = self.cursor & 7;
            if b {
                self.data[byte_idx] |= 1 << bit_idx;
            }
            self.cursor += 1;
        }
    }

    #[inline]
    fn write_bits(&mut self, val: u32, n: usize) {
        for i in 0..n {
            self.write_bit((val >> i) & 1 != 0);
        }
    }

    #[inline]
    fn bit_position(&self) -> usize {
        self.cursor
    }

    fn finish(self) -> [u8; 32] {
        self.data
    }
}

// =============================================================================
// Huffman デコード / エンコード ヘルパー
// =============================================================================

/// 駒種インデックス → PieceType
fn piece_type_from_index(idx: usize) -> PieceType {
    match idx {
        1 => PieceType::Pawn,
        2 => PieceType::Lance,
        3 => PieceType::Knight,
        4 => PieceType::Silver,
        5 => PieceType::Bishop,
        6 => PieceType::Rook,
        7 => PieceType::Gold,
        _ => PieceType::None,
    }
}

/// PieceType → Huffman テーブルインデックス
fn piece_type_to_index(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn | PieceType::ProPawn => 1,
        PieceType::Lance | PieceType::ProLance => 2,
        PieceType::Knight | PieceType::ProKnight => 3,
        PieceType::Silver | PieceType::ProSilver => 4,
        PieceType::Bishop | PieceType::Horse => 5,
        PieceType::Rook | PieceType::Dragon => 6,
        PieceType::Gold => 7,
        _ => 0,
    }
}

/// 指定テーブルでハフマン復号 → 駒種インデックス (0=空)
fn decode_huffman(stream: &mut BitReader, table: &[HuffmanCode; 8]) -> Option<usize> {
    let mut code = 0u8;
    let mut bits = 0u8;

    loop {
        code |= stream.read_bit() << bits;
        bits += 1;
        if bits > 6 {
            return None;
        }
        for (i, h) in table.iter().enumerate() {
            if h.code == code && h.bits == bits {
                return Some(i);
            }
        }
    }
}

/// HCP 手駒/駒箱の prefix-free コードを1エントリ読み取る
fn decode_hcp_hand_entry(stream: &mut BitReader) -> Option<(usize, Option<Color>, bool)> {
    let mut code = 0u8;
    let mut bits = 0u8;

    loop {
        code |= stream.read_bit() << bits;
        bits += 1;
        if bits > 7 {
            return None;
        }
        for entry in &HCP_HAND_TABLE {
            if entry.code == code && entry.bits == bits {
                return Some((entry.piece_idx, entry.color, entry.is_piecebox));
            }
        }
    }
}

// =============================================================================
// MiniPosition — 指し手適用とパック機能を持つ最小局面表現
// =============================================================================

/// .pack ローダー内部で使用する最小限の局面表現
///
/// HCP/平手から初期化し、cshogi move16 を適用しながら
/// PSfen 形式にパックする機能を提供する。
struct MiniPosition {
    board: [Piece; 81],
    hands: [Hand; 2],
    side_to_move: Color,
    game_ply: u16,
}

impl MiniPosition {
    /// 平手初期局面
    fn hirate() -> Self {
        let mut pos =
            Self { board: [Piece::NONE; 81], hands: [Hand::EMPTY; 2], side_to_move: Color::Black, game_ply: 1 };

        // 盤面配置 (YaneuraOu マス番号: file*9+rank, file=0→1筋, rank=0→1段)
        // --- 後手陣 ---
        // 1段目 (rank=0): 香桂銀金玉金銀桂香 (9筋→1筋 = file 8→0)
        let rank0_pieces = [
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::King,
            PieceType::Gold,
            PieceType::Silver,
            PieceType::Knight,
            PieceType::Lance,
        ];
        for (file_rev, &pt) in rank0_pieces.iter().enumerate() {
            let file = 8 - file_rev;
            let sq = file * 9; // rank=0
            pos.board[sq] = Piece::new(Color::White, pt);
        }
        // 2段目 (rank=1): 角(8二=file7*9+1=64), 飛(2二=file1*9+1=10)
        pos.board[7 * 9 + 1] = Piece::new(Color::White, PieceType::Bishop);
        pos.board[9 + 1] = Piece::new(Color::White, PieceType::Rook);
        // 3段目 (rank=2): 歩9枚
        for file in 0..9 {
            pos.board[file * 9 + 2] = Piece::new(Color::White, PieceType::Pawn);
        }

        // --- 先手陣 ---
        // 9段目 (rank=8): 香桂銀金玉金銀桂香 (9筋→1筋 = file 8→0)
        for (file_rev, &pt) in rank0_pieces.iter().enumerate() {
            let file = 8 - file_rev;
            let sq = file * 9 + 8;
            pos.board[sq] = Piece::new(Color::Black, pt);
        }
        // 8段目 (rank=7): 飛(8八=file7*9+7=70), 角(2八=file1*9+7=16)
        pos.board[7 * 9 + 7] = Piece::new(Color::Black, PieceType::Rook);
        pos.board[9 + 7] = Piece::new(Color::Black, PieceType::Bishop);
        // 7段目 (rank=6): 歩9枚
        for file in 0..9 {
            pos.board[file * 9 + 6] = Piece::new(Color::Black, PieceType::Pawn);
        }

        pos
    }

    /// HuffmanCodedPos (HCP) 32バイトからデコード
    fn from_hcp(hcp: &[u8; 32], game_ply: u16) -> Option<Self> {
        let mut pos = Self { board: [Piece::NONE; 81], hands: [Hand::EMPTY; 2], side_to_move: Color::Black, game_ply };

        let mut stream = BitReader::new(hcp);

        // 手番 (1 bit)
        pos.side_to_move = if stream.read_bit() == 0 { Color::Black } else { Color::White };

        // 先手玉 (7 bit)
        let bk = stream.read_bits(7) as u8;
        if bk >= 81 {
            return None;
        }
        pos.board[bk as usize] = Piece::new(Color::Black, PieceType::King);

        // 後手玉 (7 bit)
        let wk = stream.read_bits(7) as u8;
        if wk >= 81 {
            return None;
        }
        pos.board[wk as usize] = Piece::new(Color::White, PieceType::King);

        // 盤上の駒 (HCP テーブル, color → promotion 順)
        for sq in 0..81usize {
            if sq == bk as usize || sq == wk as usize {
                continue;
            }
            let idx = decode_huffman(&mut stream, &HCP_HUFFMAN_TABLE)?;
            if idx == 0 {
                continue; // 空マス
            }
            let base_pt = piece_type_from_index(idx);

            // HCP: 先後フラグ → 成りフラグ (PSfen とは逆順)
            let color = if stream.read_bit() == 0 { Color::Black } else { Color::White };
            let promoted = if base_pt != PieceType::Gold { stream.read_bit() != 0 } else { false };

            let pt = if promoted { base_pt.promote() } else { base_pt };
            pos.board[sq] = Piece::new(color, pt);
        }

        // 手駒/駒箱 (HCP 独自 prefix-free コード)
        while stream.cursor() < 256 {
            let (piece_idx, color, is_piecebox) = decode_hcp_hand_entry(&mut stream)?;
            if is_piecebox {
                continue;
            }
            let pt = piece_type_from_index(piece_idx);
            let color = color?;
            let hand_idx = color as usize;
            pos.hands[hand_idx].add(pt, 1);
        }

        Some(pos)
    }

    /// 玉の位置を検索
    fn king_square(&self, color: Color) -> u8 {
        for (i, &p) in self.board.iter().enumerate() {
            if p.piece_type == PieceType::King && p.color == color {
                return i as u8;
            }
        }
        81 // NONE
    }

    /// cshogi move16 を適用
    ///
    /// ## cshogi move16 形式
    /// - bits 0-6:  移動先 (to)
    /// - bits 7-13: 移動元 (from) または打ち駒種 (from >= 81)
    /// - bit 14:    成りフラグ
    fn do_move(&mut self, move16: u16) {
        let to = (move16 & 0x7F) as usize;
        let from_or_pt = ((move16 >> 7) & 0x7F) as usize;
        let promote = (move16 & 0x4000) != 0;
        let stm = self.side_to_move;

        if from_or_pt >= 81 {
            // 駒打ち (cshogi: 0=歩,1=香,2=桂,3=銀,4=角,5=飛,6=金)
            let pt = match from_or_pt - 81 {
                0 => PieceType::Pawn,
                1 => PieceType::Lance,
                2 => PieceType::Knight,
                3 => PieceType::Silver,
                4 => PieceType::Bishop,
                5 => PieceType::Rook,
                6 => PieceType::Gold,
                _ => return,
            };
            self.subtract_hand(stm, pt);
            self.board[to] = Piece::new(stm, pt);
        } else {
            // 通常移動
            let moving = self.board[from_or_pt];
            let captured = self.board[to];

            // 取った駒を持ち駒に追加
            if captured.is_some() && captured.piece_type != PieceType::King {
                let cap_pt = captured.piece_type.unpromote();
                self.hands[stm as usize].add(cap_pt, 1);
            }

            // 移動元を空にする
            self.board[from_or_pt] = Piece::NONE;

            // 成り処理
            let pt = if promote { moving.piece_type.promote() } else { moving.piece_type };
            self.board[to] = Piece::new(stm, pt);
        }

        self.side_to_move = stm.opponent();
        self.game_ply += 1;
    }

    /// 持ち駒を1枚減らす
    fn subtract_hand(&mut self, color: Color, pt: PieceType) {
        let hand = &mut self.hands[color as usize];
        let c = hand.count(pt);
        if c > 0 {
            hand.set(pt, c - 1);
        }
    }

    /// 現在の局面を YaneuraOu PSfen (32バイト) にエンコード
    fn pack_to_psfen(&self) -> [u8; 32] {
        let mut w = BitWriter::new();

        // 1. 手番 (1 bit)
        w.write_bit(self.side_to_move == Color::White);

        // 2. 先手玉位置 (7 bit)
        w.write_bits(self.king_square(Color::Black) as u32, 7);

        // 3. 後手玉位置 (7 bit)
        w.write_bits(self.king_square(Color::White) as u32, 7);

        // 4. 盤上の駒 (81マス, 玉スキップ)
        let bk = self.king_square(Color::Black) as usize;
        let wk = self.king_square(Color::White) as usize;
        for sq in 0..81usize {
            if sq == bk || sq == wk {
                continue;
            }
            let piece = self.board[sq];
            if piece.is_none() {
                // 空マス: 0 (1bit)
                w.write_bit(false);
            } else {
                let idx = piece_type_to_index(piece.piece_type);
                let h = &YO_HUFFMAN_TABLE[idx];
                w.write_bits(h.code as u32, h.bits as usize);

                // 金以外は成りフラグ
                let raw_pt = piece.piece_type.unpromote();
                if raw_pt != PieceType::Gold {
                    w.write_bit(piece.piece_type.is_promoted());
                }

                // 先後フラグ
                w.write_bit(piece.color == Color::White);
            }
        }

        // 5. 手駒 (YO: シフトハフマン + 成りフラグ(0) + 先後フラグ)
        let hand_pts = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
        ];
        for &color in &[Color::Black, Color::White] {
            for &pt in &hand_pts {
                let count = self.hands[color as usize].count(pt);
                for _ in 0..count {
                    let idx = piece_type_to_index(pt);
                    let h = &YO_HUFFMAN_TABLE[idx];
                    // 手駒用: code >> 1, bits - 1
                    w.write_bits((h.code >> 1) as u32, (h.bits - 1) as usize);
                    // 金以外は成りフラグ (手駒は成っていないので 0)
                    if pt != PieceType::Gold {
                        w.write_bit(false);
                    }
                    // 先後フラグ
                    w.write_bit(color == Color::White);
                }
            }
        }

        // 残りビットは 0 埋め（BitWriter は初期値 0 なのでそのまま）
        // 駒落ち等で 256bit に満たない場合は駒箱パディングで埋める
        while w.bit_position() < 256 {
            // 駒箱の歩: ハフマン(0, 1bit) + 成りフラグ(1) + 先後フラグ(0) = 3bit
            w.write_bit(false);
            w.write_bit(true);
            w.write_bit(false);
        }

        w.finish()
    }

    /// 現在の局面から PackedSfenValue を生成
    fn to_packed_sfen_value(&self, score: i16, move16: u16, game_result: i8) -> PackedSfenValue {
        let psfen = self.pack_to_psfen();
        let mut data = [0u8; 40];
        data[0..32].copy_from_slice(&psfen);
        data[32..34].copy_from_slice(&score.to_le_bytes());
        data[34..36].copy_from_slice(&move16.to_le_bytes());
        data[36..38].copy_from_slice(&self.game_ply.to_le_bytes());
        data[38] = game_result as u8;
        // data[39] = 0 (padding)

        // Safety: PackedSfenValue は [u8; 40] と同レイアウト
        PackedSfenValue::from_raw(data)
    }
}

// =============================================================================
// PackedSfenValue::from_raw ヘルパー
// =============================================================================

impl PackedSfenValue {
    /// 生のバイト配列から PackedSfenValue を構築
    fn from_raw(data: [u8; 40]) -> Self {
        // Safety: PackedSfenValue は #[repr(C)] の [u8; 40] ラッパー
        // data フィールドへの直接アクセス手段が as_bytes_mut のみのため、
        // デフォルト値を生成してからコピーする
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&data);
        psv
    }
}

// =============================================================================
// .pack パーサー
// =============================================================================

/// .pack ファイルのバイト列カーソル
/// .pack ファイルを逐次的に読むカーソル。
/// 多 GB 規模の corpus でも全量メモリにロードしないようストリーム読み出しする。
struct PackCursor {
    reader: BufReader<File>,
    eof: bool,
}

impl PackCursor {
    fn new(reader: BufReader<File>) -> Self {
        Self { reader, eof: false }
    }

    /// 次の 1 byte を peek してファイル終端かを判定する。
    /// `&mut self` なのは BufReader の内部バッファを fill する必要があるため。
    fn eof(&mut self) -> bool {
        if self.eof {
            return true;
        }
        match self.reader.fill_buf() {
            Ok([]) | Err(_) => {
                self.eof = true;
                true
            }
            Ok(_) => false,
        }
    }

    fn read_u8(&mut self) -> Option<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).ok().map(|_| buf[0])
    }

    fn read_u16(&mut self) -> Option<u16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).ok().map(|_| u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Option<i16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).ok().map(|_| i16::from_le_bytes(buf))
    }

    fn read_bytes_32(&mut self) -> Option<[u8; 32]> {
        let mut buf = [0u8; 32];
        self.reader.read_exact(&mut buf).ok().map(|_| buf)
    }
}

/// 終局マーカー判定: from == to
fn is_end_marker(move16: u16) -> bool {
    let to = move16 & 0x7F;
    let from = (move16 >> 7) & 0x7F;
    to == from
}

/// .pack の game_result (0=draw, 1=black_win, 2=white_win) を
/// PSV の per-STM game_result (1=win, -1=loss, 0=draw) に変換
fn convert_game_result(pack_result: u8, stm: Color) -> i8 {
    match pack_result {
        1 => {
            if stm == Color::Black {
                1
            } else {
                -1
            }
        }
        2 => {
            if stm == Color::White {
                1
            } else {
                -1
            }
        }
        _ => 0, // draw or unknown
    }
}

/// 1対局分のデータ
struct RawGameData {
    /// 開始局面 (MiniPosition)
    start_pos: MiniPosition,
    /// (cshogi move16, eval) のリスト
    moves: Vec<(u16, i16)>,
    /// .pack の game_result (0=draw, 1=black_win, 2=white_win)
    pack_game_result: u8,
}

/// .pack カーソルから1対局を読み取る
fn read_one_game(cursor: &mut PackCursor) -> Option<RawGameData> {
    let start_flag = cursor.read_u8()?;

    let start_pos = match start_flag {
        1 => MiniPosition::hirate(),
        0 => {
            let hcp = cursor.read_bytes_32()?;
            let game_ply = cursor.read_u16()?;
            MiniPosition::from_hcp(&hcp, game_ply)?
        }
        _ => return None,
    };

    let mut moves = Vec::new();
    let pack_game_result;

    loop {
        let move16 = cursor.read_u16()?;
        if is_end_marker(move16) {
            pack_game_result = (move16 & 0x7F) as u8;
            let _reason = cursor.read_u8()?;
            break;
        }
        let eval = cursor.read_i16()?;
        moves.push((move16, eval));
    }

    Some(RawGameData { start_pos, moves, pack_game_result })
}

/// 1対局を PackedSfenValue のリストに展開
fn expand_game(game: RawGameData) -> Vec<PackedSfenValue> {
    let mut pos = game.start_pos;
    let mut result = Vec::with_capacity(game.moves.len());

    for &(move16, eval) in &game.moves {
        let stm = pos.side_to_move;
        let game_result = convert_game_result(game.pack_game_result, stm);

        let psv = pos.to_packed_sfen_value(eval, move16, game_result);
        result.push(psv);

        pos.do_move(move16);
    }

    result
}

// =============================================================================
// ShogiPackLoader
// =============================================================================

/// .pack ファイルを読み取り PackedSfenValue を供給するデータローダー
///
/// GenSfen の .pack 形式（可変長対局棋譜）を読み込み、各局面を
/// PackedSfenValue に展開してバッファシャッフル付きで供給する。
///
/// ## 使用例
///
/// ```rust,ignore
/// use bullet_lib::value::loader::shogipack::ShogiPackLoader;
///
/// let loader = ShogiPackLoader::new(
///     "data/train.pack",
///     256,  // buffer_size_mb
///     4,    // threads
///     |_psv| true,  // フィルタ (全通し)
/// );
/// ```
#[derive(Clone)]
pub struct ShogiPackLoader<T: Fn(&PackedSfenValue) -> bool> {
    file_paths: Vec<String>,
    buffer_size: usize,
    filter: T,
}

impl<T: Fn(&PackedSfenValue) -> bool> ShogiPackLoader<T> {
    /// 単一ファイルから作成
    pub fn new(path: &str, buffer_size_mb: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, filter)
    }

    /// 複数ファイルを連結して作成
    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, filter: T) -> Self {
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<PackedSfenValue>() / 2,
            filter,
        }
    }
}

impl<T> DataLoader<PackedSfenValue> for ShogiPackLoader<T>
where
    T: Fn(&PackedSfenValue) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[PackedSfenValue]) -> bool>(&self, start_batch: usize, batch_size: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let filter = self.filter.clone();

        // ===== Resume support (consume-and-drop, best-effort) =====
        //
        // .pack は (1) 可変長レコード (2) caller 提供 filter (3) shuffle buffer の
        // 3 要素により bit-exact な seek が原理的に不可能。本実装は expander 段で
        // start_batch * batch_size 個の filter 通過 position を input 順序で
        // 読み飛ばす "best-effort consume-and-drop" を採用している。
        //
        // **既知の限界**: shuffle buffer (= buffer_size 個の position) 単位で見ると、
        // fresh run は buffer 内の random subset を emit しているのに対し、
        // resume run は buffer の先頭 N 個を input 順序で drop する。このため
        // 境界 1 shuffle window 分の position について、
        //   - fresh で emit 済み (= 学習済み) のうち一部が resume でも emit される (重複学習)
        //   - fresh で未 emit のうち一部が resume の skip 対象に入って drop される (永久 skip)
        // という現象が起きる。影響は最大 1 shuffle buffer 分 (~256k〜数 M position) に
        // 限定され、データセット全体の 0.01〜0.1% スケールのため学習結果への影響は
        // 軽微 (NN にとってはノイズ未満) と判断して受容している。
        //
        // **完全な bit-exact resume が必要な場合**: .pack を事前に .psv に展開して
        // DirectSequentialDataLoader (固定長レコード前提なので byte 単位で seek 可) で
        // 読むこと。これが現在の主要な学習パスでもある。
        //
        // **本 loader を実学習で多用するなら**: shuffle 段の RNG seed を引数化して
        // post-shuffle skip に切り替える、もしくは shuffle window 境界で checkpoint
        // を保存する設計に refactor する必要がある。詳細議論は PR #12 review thread 参照。
        let positions_to_skip = start_batch.saturating_mul(batch_size);
        if positions_to_skip > 0 {
            eprintln!(
                "[ShogiPackLoader] WARNING: start_batch={start_batch} (skip {positions_to_skip} positions). \n\
                .pack resume is BEST-EFFORT, NOT bit-exact: ~1 shuffle buffer worth of positions \n\
                near the resume boundary will be partially replayed and partially never seen. \n\
                For bit-exact resume, preprocess .pack to .psv and use DirectSequentialDataLoader.",
            );
        }

        // ----- Stage 1: Reader (ファイル → RawGameData バッチ) -----
        // 空の Vec は "1 sweep 完了 (= 全ファイル一周)" のマーカーとして downstream に流れ、
        // shuffle 段の tail flush をトリガーする。
        let reader_buffer_size = 256;
        let (reader_tx, reader_rx) = mpsc::sync_channel::<Vec<RawGameData>>(8);
        let (reader_stop_tx, reader_stop_rx) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut buffer = Vec::with_capacity(reader_buffer_size);

            'dataloading: loop {
                for file_path in &file_paths {
                    // データセット指定ミス (パス typo / 権限なし) は fail-fast。
                    // silent な continue だと reader loop が無限化し、学習スレッドが
                    // batch を永遠に待ってハングするため。
                    let file = File::open(file_path).unwrap_or_else(|e| {
                        panic!("Failed to open .pack file {file_path:?}: {e}");
                    });
                    // 多 GB のコーパスでも OOM しないよう BufReader 経由で逐次読みする。
                    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
                    let mut cursor = PackCursor::new(reader);

                    while !cursor.eof() {
                        let game = match read_one_game(&mut cursor) {
                            Some(g) => g,
                            None => break, // 可変長のため位置復帰不可
                        };
                        buffer.push(game);

                        if buffer.len() >= reader_buffer_size {
                            if reader_stop_rx.try_recv().unwrap_or(false) || reader_tx.send(buffer).is_err() {
                                break 'dataloading;
                            }
                            buffer = Vec::with_capacity(reader_buffer_size);
                        }
                    }
                }

                // 1 sweep 完了。残りバッファを送信。
                if !buffer.is_empty() {
                    if reader_stop_rx.try_recv().unwrap_or(false) || reader_tx.send(buffer).is_err() {
                        break;
                    }
                    buffer = Vec::with_capacity(reader_buffer_size);
                }
                // sweep 終了マーカー (空 Vec)。小規模 corpus でも shuffle buffer が
                // flush されるよう downstream に通知する。
                if reader_stop_rx.try_recv().unwrap_or(false) || reader_tx.send(Vec::new()).is_err() {
                    break;
                }
            }
        });

        // ----- Stage 2: Expander (RawGameData → PackedSfenValue, フィルタ適用) -----
        // 空 Vec の sweep 終了マーカーは expand せずそのまま downstream へ転送する。
        // resume の skip カウンタもここで管理する。
        let (expand_tx, expand_rx) = mpsc::sync_channel::<Vec<PackedSfenValue>>(16);
        let (expand_stop_tx, expand_stop_rx) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut skipped: usize = 0;
            // filter 全弾き等で「filter を通過する position が 0 の sweep」が連続したら
            // 設定ミス or 空データセットと判断し panic で fail-fast。silent な hang は
            // debug 不能なため。"filter 通過数" で判定するので resume の skip 中
            // (positions_to_skip 消化中) でも誤発火しない。
            let mut filter_accepted_in_sweep: usize = 0;
            let mut consecutive_empty_sweeps: usize = 0;
            const MAX_EMPTY_SWEEPS: usize = 2;

            'dataloading: while let Ok(games) = reader_rx.recv() {
                if expand_stop_rx.try_recv().unwrap_or(false) {
                    reader_stop_tx.send(true).ok();
                    break 'dataloading;
                }

                let is_sweep_end = games.is_empty();
                let mut positions = Vec::new();
                for game in games {
                    let expanded = expand_game(game);
                    for psv in expanded {
                        if !filter(&psv) {
                            continue;
                        }
                        filter_accepted_in_sweep += 1;
                        if skipped < positions_to_skip {
                            skipped += 1;
                            continue;
                        }
                        positions.push(psv);
                    }
                }

                // sweep 終了マーカーは空でも必ず流す (shuffle 段の tail flush 用)。
                let send_needed = is_sweep_end || !positions.is_empty();
                if send_needed && expand_tx.send(positions).is_err() {
                    reader_stop_tx.send(true).ok();
                    break 'dataloading;
                }

                if is_sweep_end {
                    if filter_accepted_in_sweep == 0 {
                        consecutive_empty_sweeps += 1;
                        if consecutive_empty_sweeps >= MAX_EMPTY_SWEEPS {
                            panic!(
                                "ShogiPackLoader: filter accepted 0 positions in {MAX_EMPTY_SWEEPS} consecutive \
                                sweeps. Filter is too restrictive or the .pack corpus contains no usable data.",
                            );
                        }
                    } else {
                        consecutive_empty_sweeps = 0;
                    }
                    filter_accepted_in_sweep = 0;
                }
            }
        });

        // ----- Stage 3: Shuffle (バッファ蓄積 → Fisher-Yates シャッフル) -----
        // 通常はバッファが buffer_size に達した時点で flush するが、
        // 1 sweep 全体で buffer_size に満たない小規模 corpus の場合に学習が
        // ハングしないよう、sweep 終了マーカー (空 Vec) を受けたら残バッファを flush する。
        let (shuffle_tx, shuffle_rx) = mpsc::sync_channel::<Vec<PackedSfenValue>>(0);
        let (shuffle_stop_tx, shuffle_stop_rx) = mpsc::sync_channel::<bool>(1);

        // Note: filter 全弾きの fail-fast 検知は expander 段で行う (filter 通過数で判定するため)。
        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);

            'dataloading: while let Ok(positions) = expand_rx.recv() {
                let is_sweep_end = positions.is_empty();
                for entry in positions {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() >= buffer_size {
                        shuffle(&mut shuffle_buffer);

                        if shuffle_stop_rx.try_recv().unwrap_or(false) || shuffle_tx.send(shuffle_buffer).is_err() {
                            expand_stop_tx.send(true).ok();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }

                // tail flush: 1 sweep 通しても buffer_size に満たないケース対応。
                if is_sweep_end && !shuffle_buffer.is_empty() {
                    shuffle(&mut shuffle_buffer);

                    if shuffle_stop_rx.try_recv().unwrap_or(false) || shuffle_tx.send(shuffle_buffer).is_err() {
                        expand_stop_tx.send(true).ok();
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::with_capacity(buffer_size);
                }
            }
        });

        // ----- Stage 4: Batch (チャンク分割 → コールバック) -----
        let (batch_tx, batch_rx) = mpsc::sync_channel::<Vec<PackedSfenValue>>(16);
        let (batch_stop_tx, batch_stop_rx) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(shuffle_buffer) = shuffle_rx.recv() {
                for batch in shuffle_buffer.chunks(batch_size) {
                    if batch_stop_rx.try_recv().unwrap_or(false) || batch_tx.send(batch.to_vec()).is_err() {
                        shuffle_stop_tx.send(true).ok();
                        break 'dataloading;
                    }
                }
            }
        });

        'dataloading: while let Ok(inputs) = batch_rx.recv() {
            for batch in inputs.chunks(batch_size) {
                let should_break = f(batch);
                if should_break {
                    batch_stop_tx.send(true).ok();
                    break 'dataloading;
                }
            }
        }

        drop(batch_rx);
    }
}

/// Fisher-Yates シャッフル
fn shuffle(data: &mut [PackedSfenValue]) {
    let mut rng = SimpleRand::with_seed();
    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::Square;

    #[test]
    fn test_hirate_pack_roundtrip() {
        // 平手初期局面をパックし、既存の PackedSfenValue デコーダで復元して検証
        let pos = MiniPosition::hirate();
        let psfen = pos.pack_to_psfen();

        // PackedSfenValue にラップしてデコード
        let mut data = [0u8; 40];
        data[0..32].copy_from_slice(&psfen);
        let psv = PackedSfenValue::from_raw(data);
        let board = psv.decode();

        // 手番
        assert_eq!(board.side_to_move, Color::Black);

        // 先手玉 (5九 = file4*9+rank8 = 44)
        assert_eq!(board.black_king_sq, Square(44));

        // 後手玉 (5一 = file4*9+rank0 = 36)
        assert_eq!(board.white_king_sq, Square(36));

        // 先手歩 (7七 = file6*9+6 = 60)
        let p = board.board[60];
        assert_eq!(p.piece_type, PieceType::Pawn);
        assert_eq!(p.color, Color::Black);

        // 後手飛 (2二 = file1*9+1 = 10)
        let p = board.board[10];
        assert_eq!(p.piece_type, PieceType::Rook);
        assert_eq!(p.color, Color::White);

        // 先手角 (8八 = file7*9+7 = 70 ではなく 2八 = file1*9+7 = 16)
        let p = board.board[16];
        assert_eq!(p.piece_type, PieceType::Bishop);
        assert_eq!(p.color, Color::Black);
    }

    #[test]
    fn test_do_move_normal() {
        let mut pos = MiniPosition::hirate();
        // 7六歩: from=60 (7七=file6*9+6), to=51 (7六=file5*9+6)
        // ↑ 訂正: 7七 = file(9-7)*9+rank(7-1) = file6*9+6=60 は正しいが
        //         7六 = file6*9+5=59
        // Wait: YO座標系では file=0 が1筋, rank=0 が1段
        // 7七 = 7筋7段 → file=6, rank=6 → sq=6*9+6=60
        // 7六 = 7筋6段 → file=6, rank=5 → sq=6*9+5=59
        let move16 = 59 | (60 << 7); // to=59, from=60
        pos.do_move(move16);

        assert_eq!(pos.board[60], Piece::NONE); // 元の位置は空
        assert_eq!(pos.board[59].piece_type, PieceType::Pawn);
        assert_eq!(pos.board[59].color, Color::Black);
        assert_eq!(pos.side_to_move, Color::White);
    }

    #[test]
    fn test_do_move_capture() {
        let mut pos = MiniPosition::hirate();
        // 簡易テスト: 歩を相手の駒の位置に移動（実戦ではありえないが機能テスト）
        // 先手歩 at sq=60 を後手歩 at sq=56 (file6*9+2=56) に移動
        // これは合法手ではないがロジックのテスト
        let move16 = 56u16 | (60 << 7);
        pos.do_move(move16);

        assert_eq!(pos.board[60], Piece::NONE);
        assert_eq!(pos.board[56].piece_type, PieceType::Pawn);
        assert_eq!(pos.board[56].color, Color::Black);
        // 取った歩が先手の持ち駒に
        assert_eq!(pos.hands[Color::Black as usize].count(PieceType::Pawn), 1);
    }

    #[test]
    fn test_do_move_drop() {
        let mut pos = MiniPosition::hirate();
        // まず持ち駒に歩を追加
        pos.hands[Color::Black as usize].add(PieceType::Pawn, 1);

        // 歩打ち: to=40 (5五=file4*9+4), drop=歩(0)
        // from_or_pt = 81 + 0 = 81
        let move16 = 40u16 | (81 << 7);
        pos.do_move(move16);

        assert_eq!(pos.board[40].piece_type, PieceType::Pawn);
        assert_eq!(pos.board[40].color, Color::Black);
        assert_eq!(pos.hands[Color::Black as usize].count(PieceType::Pawn), 0);
    }

    #[test]
    fn test_do_move_promote() {
        let mut pos = MiniPosition::hirate();
        // 先手歩を成らせるテスト
        // 歩を3段目(rank=2)に置いて、2段目(rank=1)に成りで進む
        pos.board[6 * 9 + 2] = Piece::new(Color::Black, PieceType::Pawn); // 7三に配置
        let from_sq = 6 * 9 + 2; // 7三
        let to_sq = 6 * 9 + 1; // 7二

        // 元の3段目の歩を消す（3段目には後手歩がある）
        let move16 = to_sq as u16 | ((from_sq as u16) << 7) | 0x4000; // promote bit
        pos.do_move(move16);

        assert_eq!(pos.board[to_sq].piece_type, PieceType::ProPawn);
        assert_eq!(pos.board[to_sq].color, Color::Black);
    }

    #[test]
    fn test_end_marker() {
        // draw: to=from=0 → move16 = 0 | (0 << 7) = 0, game_result = 0
        assert!(is_end_marker(0));

        // black_win: to=from=1 → move16 = 1 | (1 << 7) = 129
        assert!(is_end_marker(129));

        // 通常の手 (to != from)
        assert!(!is_end_marker(59 | (60 << 7)));
    }

    #[test]
    fn test_convert_game_result() {
        assert_eq!(convert_game_result(0, Color::Black), 0); // draw
        assert_eq!(convert_game_result(1, Color::Black), 1); // black wins, stm=black → win
        assert_eq!(convert_game_result(1, Color::White), -1); // black wins, stm=white → loss
        assert_eq!(convert_game_result(2, Color::White), 1); // white wins, stm=white → win
        assert_eq!(convert_game_result(2, Color::Black), -1); // white wins, stm=black → loss
    }
}
