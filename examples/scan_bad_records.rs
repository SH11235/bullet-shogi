/// DLSuisho15b_unique の不正レコード調査ツール
///
/// PackedSfenValue を1件ずつデコードし、特徴量数が40を超えるレコードを検出する。
///
/// Usage:
///   cargo run --release --example scan_bad_records -- <file_path> [--limit N] [--dump-first]
use std::env;
use std::fs::File;
use std::io::Read;

use bullet_lib::shogi::{
    ShogiBoard,
    packed_sfen::PackedSfenValue,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, PieceType},
};

const RECORD_SIZE: usize = 40;

fn count_features(board: &ShogiBoard) -> usize {
    let mut count = 0usize;

    // 盤上の駒（王以外）
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            count += board.pieces(color, pt).count();
        }
    }

    // 両玉 = 2（玉が有効な場合）
    if board.king_square(Color::Black).is_valid() {
        count += 1;
    }
    if board.king_square(Color::White).is_valid() {
        count += 1;
    }

    // 手駒
    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            let c = board.hand(owner).count(pt);
            count += c as usize;
        }
    }

    count
}

fn count_pieces_by_type(board: &ShogiBoard) -> Vec<(String, usize)> {
    let mut result = Vec::new();

    // 盤上駒
    for &pt in &BOARD_PIECE_TYPES {
        for color in [Color::Black, Color::White] {
            let c = board.pieces(color, pt).count();
            if c > 0 {
                let color_str = if color == Color::Black { "B" } else { "W" };
                result.push((format!("{color_str}_{pt:?}"), c));
            }
        }
    }

    // 玉
    if board.king_square(Color::Black).is_valid() {
        result.push(("B_King".to_string(), 1));
    }
    if board.king_square(Color::White).is_valid() {
        result.push(("W_King".to_string(), 1));
    }

    // 手駒
    for owner in [Color::Black, Color::White] {
        for &pt in &HAND_PIECE_TYPES {
            let c = board.hand(owner).count(pt) as usize;
            if c > 0 {
                let color_str = if owner == Color::Black { "B" } else { "W" };
                result.push((format!("{color_str}_Hand_{pt:?}"), c));
            }
        }
    }

    result
}

fn dump_record(idx: usize, offset: u64, psv: &PackedSfenValue, board: &ShogiBoard, features: usize) {
    println!("=== BAD RECORD #{idx} at offset {offset} (byte {offset:#x}) ===");
    println!("  Features: {features} (max=40)");
    println!(
        "  STM: {:?}, Score: {}, Result: {}, Ply: {}",
        board.side_to_move,
        psv.score(),
        psv.game_result(),
        psv.game_ply(),
    );
    println!("  Black King: sq={}, White King: sq={}", board.black_king_sq.0, board.white_king_sq.0,);

    // 駒の内訳
    let pieces = count_pieces_by_type(board);
    println!("  Piece breakdown:");
    for (name, count) in &pieces {
        println!("    {name}: {count}");
    }

    // 駒種別合計
    let board_pieces: usize = BOARD_PIECE_TYPES
        .iter()
        .flat_map(|&pt| [Color::Black, Color::White].map(|c| board.pieces(c, pt).count()))
        .sum();
    let kings = [Color::Black, Color::White].iter().filter(|&&c| board.king_square(c).is_valid()).count();
    let hand_pieces: usize = [Color::Black, Color::White]
        .iter()
        .flat_map(|&c| HAND_PIECE_TYPES.map(|pt| board.hand(c).count(pt) as usize))
        .sum();
    println!("  Board pieces (excl king): {board_pieces}");
    println!("  Kings: {kings}");
    println!("  Hand pieces: {hand_pieces}");
    println!("  Total: {}", board_pieces + kings + hand_pieces);

    // 駒数チェック（歩18枚、香4枚 etc）
    check_piece_counts(board);

    // raw bytes
    let bytes = psv.as_bytes();
    print!("  Raw bytes: ");
    for b in bytes.iter() {
        print!("{b:02x} ");
    }
    println!();
    println!();
}

fn check_piece_counts(board: &ShogiBoard) {
    // 各駒種の最大枚数 (両方の手駒+盤上合計)
    let limits: &[(PieceType, &[PieceType], u8, &str)] = &[
        (PieceType::Pawn, &[PieceType::Pawn, PieceType::ProPawn], 18, "歩+と"),
        (PieceType::Lance, &[PieceType::Lance, PieceType::ProLance], 4, "香+成香"),
        (PieceType::Knight, &[PieceType::Knight, PieceType::ProKnight], 4, "桂+成桂"),
        (PieceType::Silver, &[PieceType::Silver, PieceType::ProSilver], 4, "銀+成銀"),
        (PieceType::Gold, &[PieceType::Gold], 4, "金"),
        (PieceType::Bishop, &[PieceType::Bishop, PieceType::Horse], 2, "角+馬"),
        (PieceType::Rook, &[PieceType::Rook, PieceType::Dragon], 2, "飛+龍"),
    ];

    for &(hand_pt, board_pts, max, name) in limits {
        let mut total = 0usize;
        // 盤上
        for &pt in board_pts {
            for color in [Color::Black, Color::White] {
                total += board.pieces(color, pt).count();
            }
        }
        // 手駒
        for color in [Color::Black, Color::White] {
            total += board.hand(color).count(hand_pt) as usize;
        }
        if total > max as usize {
            println!("  *** OVER LIMIT: {name} = {total} (max {max}) ***");
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file_path> [--limit N] [--dump-first] [--quiet]", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    let mut limit: Option<usize> = None;
    let mut dump_first = false;
    let mut quiet = false;
    let mut max_bad: usize = 100; // 最大表示数

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                i += 1;
                limit = Some(args[i].parse().expect("--limit requires a number"));
            }
            "--dump-first" => {
                dump_first = true;
            }
            "--quiet" => {
                quiet = true;
            }
            "--max-bad" => {
                i += 1;
                max_bad = args[i].parse().expect("--max-bad requires a number");
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut file = File::open(file_path).expect("Failed to open file");
    let file_size = file.metadata().expect("Failed to get metadata").len();
    let total_records = file_size / RECORD_SIZE as u64;
    let records_to_scan = limit.map_or(total_records, |l| l.min(total_records as usize) as u64);

    println!("File: {file_path}");
    println!("File size: {file_size} bytes ({} records)", total_records);
    if !file_size.is_multiple_of(RECORD_SIZE as u64) {
        println!(
            "WARNING: File size is NOT aligned to {RECORD_SIZE} bytes! Remainder: {} bytes",
            file_size % RECORD_SIZE as u64
        );
    }
    println!("Scanning {records_to_scan} records...");
    println!();

    let mut buf = [0u8; RECORD_SIZE];
    let mut bad_count = 0usize;
    let mut scanned = 0u64;
    let mut feature_histogram: [u64; 60] = [0; 60]; // features 0-59

    let start = std::time::Instant::now();

    for record_idx in 0..records_to_scan {
        if file.read_exact(&mut buf).is_err() {
            eprintln!("Read error at record {record_idx}");
            break;
        }

        let psv: PackedSfenValue = unsafe { std::mem::transmute(buf) };
        let board = ShogiBoard::from_packed_sfen(&psv);
        let features = count_features(&board);

        if features < 60 {
            feature_histogram[features] += 1;
        }

        if dump_first && record_idx == 0 {
            println!("--- First record ---");
            dump_record(0, 0, &psv, &board, features);
        }

        if features > 40 {
            bad_count += 1;
            if bad_count <= max_bad {
                let offset = record_idx * RECORD_SIZE as u64;
                dump_record(bad_count, offset, &psv, &board, features);
            }
        }

        scanned += 1;

        if !quiet && scanned.is_multiple_of(50_000_000) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = scanned as f64 / elapsed;
            eprintln!(
                "  Progress: {scanned}/{records_to_scan} ({:.1}%) - {:.0} records/sec - bad: {bad_count}",
                scanned as f64 / records_to_scan as f64 * 100.0,
                rate,
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("=== Summary ===");
    println!("Scanned: {scanned} records in {elapsed:.1}s ({:.0} records/sec)", scanned as f64 / elapsed);
    println!("Bad records (features > 40): {bad_count}");

    // ヒストグラム
    println!("\nFeature count distribution:");
    for (n, &count) in feature_histogram.iter().enumerate() {
        if count > 0 {
            println!("  {n:3} features: {count:>12} records");
        }
    }
}
