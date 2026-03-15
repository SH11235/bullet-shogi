# shogi_progress_kpabs_train

LayerStack の bucket 選択に使う `progress.bin` を学習するツール。

生成した `progress.bin` は `--bucket-mode progress8kpabs --progress-coeff progress.bin` で LayerStack 学習に使用する。

## モデル

KP-absolute 特徴量（玉位置 × 駒配置）に対する線形ロジスティック回帰。

```
z = Σ weights[kp_abs_index]
p = sigmoid(z)
bucket = min(7, floor(p * 8))
```

- 重みの数: 81 × 1548 = 125,388
- 出力形式: `progress.bin` (f64 little-endian, 1,003,104 bytes)
- YaneuraOu 互換フォーマット

## データの流れ

```
棋譜生成 → raw.psv（対局順保持、シャッフル前）
              ├→ progress.bin 学習 ← ここで raw.psv を使う
              └→ リスコア → シャッフル → train.psv → NNUE学習
```

- 進行度学習にはスコア（評価値）は使わない。局面の駒配置（KP 特徴量）のみ使用
- qsearch leaf 置換前のデータを使う（駒配置が変わりうるため）

## 2つのモード

### 厳密モード (`--game-relative`、推奨)

```bash
cargo run --release --example shogi_progress_kpabs_train -- \
  --data raw.psv \
  --output progress.bin \
  --game-relative \
  --max-positions 9000000 \
  --val-positions 500000 \
  --epochs 10 \
  --lr 0.001 \
  --batch-size 4096
```

- 教師値: `y = game_ply / total_ply`（各対局の実際の総手数を使用）
- 対局境界は `game_ply` が前レコード以下になったら新対局として検出
- **対局順保持データ（シャッフル前）が必須**
- bucket 分布がより均一になることが期待される

### 近似モード（デフォルト）

```bash
cargo run --release --example shogi_progress_kpabs_train -- \
  --data data/DLSuisho15b \
  --output progress.bin \
  --max-positions 50000000 \
  --val-positions 2000000 \
  --ply-max 256
```

- 教師値: `y = clamp((game_ply - 1) / (ply_max - 1), 0, 1)`（固定の `ply_max` で正規化）
- シャッフル済みデータでも使える
- bucket 分布が偏る問題あり（v75 実験では bucket 0-4 に 96.7% 集中）

## パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--data` | (必須) | カンマ区切りのファイルまたはディレクトリ。ディレクトリ指定時は直下の `*.bin` / `*.pack` を読む |
| `--output` | (必須) | 出力 `progress.bin` のパス |
| `--game-relative` | false | 厳密モード。対局順保持データが必要 |
| `--max-positions` | 50,000,000 | 1 epoch あたりの学習サンプル数 |
| `--val-positions` | 2,000,000 | 検証用サンプル数（先頭から取得、学習には使わない） |
| `--batch-size` | 4,096 | ミニバッチサイズ |
| `--lr` | 0.0002 | Adam の学習率 |
| `--epochs` | 1 | 学習パス数 |
| `--ply-max` | 256 | 近似モードの正規化上限（`--game-relative` 時は無視） |
| `--log-interval` | 100 | バッチごとのログ出力間隔 |

## データ分割

```
データストリーム:
├── val_positions (先頭 N 件) → validation loss の計算用
└── max_positions (その後ろ M 件) → 学習用（勾配更新に使う）
```

実際に読む合計局面数は `val_positions + max_positions`。

## 処理の流れ

1. `--data` で指定したファイル/ディレクトリから `.bin` / `.pack` を列挙
2. `hao_*` / `shuffled_*` を group-interleave して round-robin 読み出し
3. `--game-relative` の場合: 2-pass で全レコードの `game_ply` を読み、対局境界を検出して教師値を事前計算
4. 先頭 `val_positions` 件で baseline validation loss を計算
5. その後ろ `max_positions` 件を `epochs` 回学習（Adam optimizer, MSE loss）
6. 重みを f64 に変換して `progress.bin` に書き出し

## NNUE 学習での使用

```bash
cargo run --release --example shogi_layerstack -- \
  --data train.psv \
  --bucket-mode progress8kpabs \
  --progress-coeff progress.bin \
  ...
```
