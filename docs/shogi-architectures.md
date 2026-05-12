# bullet-shogi がサポートする将棋 NNUE アーキテクチャ

学習を回す前に「どの入力特徴量・どのネットワーク・どの出力バケットを選ぶか」を
整理するための一覧。各組合せに対応する CLI 例も載せている。

設定の選択肢は大きく 3 軸:

1. **入力特徴量** (`bullet_lib::game::inputs` の `SparseInputType` 実装)
2. **ネットワーク構成**: 従来ネットワーク (`shogi_simple`) / LayerStack (`shogi_layerstack`)
3. **出力バケット** (LayerStack のみ。従来ネットは単一バケット固定)

---

## 1. 入力特徴量

将棋固有の sparse input 型は `crates/bullet_lib/src/game/inputs/shogi_*.rs` に
実装されている。次元・最大アクティブ数・rshogi 推論側との互換性を以下にまとめる。

### 1.1 HalfKX 系（盤・駒・王のみ）

| 型 | 入力次元 | キングバケット | max_active | shorthand | 主な用途 |
|---|---:|---:|---:|---|---|
| `ShogiHalfKP` | 125,388 | 81 (全マス) | 38 | `shogi-125388x81` | nnue-pytorch / YaneuraOu HalfKP 互換、軽量学習 |
| `ShogiHalfKA` | 138,510 | 81 (非ミラー) | 40 | `shogi-138510x81` | HalfKA non-mirror。比較・互換用 |
| `ShogiHalfKA_hm` | 73,305 | 45 (Half-Mirror, 9段×5筋) | 40 | `shogi-73305x45hm` | **既定の主力**。HalfKA_hm_v2 (`FEATURE_HASH_HM_V2 = 0x7f134cb8`) |

- `HalfKP` は王を特徴に含めない（駒入力 `FE_OLD_END = 1548`）。
- `HalfKA*` は王も含める（駒入力 1629 / 1710）。
- `HalfKA_hm` は左右ミラーで筋を 1〜5 筋に正規化し次元を半分強に圧縮した版で、
  nnue-pytorch / 既存 HalfKA_hm_v2 と互換。
- いずれも `FEATURE_HASH` が `nnue-pytorch` 互換なのでネットワーク定義変換時に
  hash で版ずれを検知できる。

### 1.2 補助入力つき HalfKA_hm 系（LayerStack 専用）

すべて HalfKA_hm (73,305 次元 / max_active 40) を sparse 部分に持ち、追加の
情報を concat する。LayerStack 学習器 (`shogi_layerstack`) で利用する。
従来ネットワーク (`shogi_simple`) からは現状使えない。

| 型 | 追加次元 | 追加 max_active | 追加内容 |
|---|---:|---:|---|
| `ShogiHalfKaHmHandCount` | +14 (dense) | +0 | stm/nstm の持ち駒本数 7 種ずつ。L1 入力に dense vector として concat |
| `ShogiHalfKaHmThreat` | +最大 216,720 | +320 | 盤面 Threat (駒種 9 family × pair)。`ThreatProfile` で次元縮約可能 |
| `ShogiHalfKaHmHandThreat` | +121,104 | +1024 | 持ち駒 drop による脅威 (full drop-attack pair, 案 A) |
| `ShogiHalfKaHmHandThreatDefensive` | +30,276 | +α (非対称) | HandThreat の `drop_owner=enemy` かつ `attacked_side=friend` のみ符号化した防御版 |

**HandCount Dense (14 元)** のレイアウトは `index 0..6` が stm 持ち駒
(歩,香,桂,銀,金,角,飛)、`index 7..13` が nstm。値は本数を `f32` キャスト。

**Threat の `ThreatProfile`**（`shogi_threat_exclusion.rs`）:

| ID | CLI 値 | 内容 | Threat 部の次元 |
|---:|---|---|---:|
| 0 | `full` | 全 pair (Baseline) | 216,720 |
| 1 | `same-class` | 同種ペア全除外 | < full |
| 2 | `same-class-major-pawn` | 同種 + 大駒→歩除外 | < `same-class` |
| 10 | `cross-side` | cross-side 異種ペアのみ | 96,320 |

> Profile を明示しなかった場合は `full`。`quantised.bin` には `ThreatProfile=<id>`
> として書き出され、rshogi 側 loader が同じ profile で読み戻す。

**HandThreat defensive の非対称 emission**: STM 視点では「相手 drop → 自分の駒
攻撃」のみ、NSTM 視点では「自分 drop → 相手駒攻撃」のみを emit するため、
同一局面でも `|STM_active| != |NSTM_active|` になる。loader は両 chunk を
独立に扱うので、利用者側で意識する必要は通常ない。

---

## 2. ネットワーク構成

### 2.1 従来ネットワーク: `examples/shogi_simple.rs`

YaneuraOu / nnue-pytorch スタイルの **単一バケット 4 層 NNUE**
(`InputDim → L1×2 → L2 → L3 → 1`)。HalfKX 系入力 (HalfKP / HalfKA / HalfKA_hm) と
組み合わせる。出力フォーマットは bullet 形式 / nnue-pytorch 形式の両方に対応。

#### アーキテクチャプリセット (`--arch`)

| プリセット | L1 | L2 | L3 | 用途の目安 |
|---|---:|---:|---:|---|
| `256x2-32-32` (default) | 256 | 32 | 32 | nnue-pytorch HalfKP 互換 baseline |
| `512x2-8-96` | 512 | 8 | 96 | やねうら王 NNUE 寄りの形 |
| `512x2-32-32` | 512 | 32 | 32 | 中間サイズ |
| `1024x2-8-32` | 1024 | 8 | 32 | 大きめ FT・薄い L2 |
| `1024x2-16-64` | 1024 | 16 | 64 | より大きい構成 |

`--l1 / --l2 / --l3` でプリセットを上書きできる。

#### 出力フォーマット (`--output-format`)

| 値 | 説明 |
|---|---|
| `bullet` | bullet 標準。全パラメータ i16。デバッグ・ロード簡単 |
| `nnue-pytorch` (default) | NNUE ヘッダ + L0 i16 + L1〜Out バイアス i32 + 重み i8。nnue-pytorch / YaneuraOu と互換 |

#### 主な追加オプション

- `--scale <N>` / `--win-rate-model` / `--wrm-in-scaling`
- `--start-wdl` / `--end-wdl` (線形 WDL)
- `--lr-gamma` / `--lr-step` (LR 減衰)
- `--start-superbatch` / `--resume`（中断再開）

### 2.2 LayerStack: `examples/shogi_layerstack.rs`

Stockfish/yaneuraou で言うところの **SFNNwoPSQT-1536** に相当する LayerStack
ネットワーク。出力側に **9 つのバケット (LayerStack)** を持ち、各バケットが
独立した L1〜Out を備える。rshogi 推論側 (`NetworkLayerStacks::read()`) で
読み込める `quantised.bin` を出力する。

#### 形

```
HalfKA_hm (+補助入力)
        │
       FT  (= L0, 既定 1536, QA=127)
        │
   stm/nstm 連結 (=2*L0)
        │
   ┌────────────────────┐
   │  bucket selector   │  ← 出力バケット (9 種類)
   └────────────────────┘
        │（バケットごと）
        L1 (既定 16)
        L2 (既定 32)
        Output (1, scaled by FV_SCALE)
```

サイズ調整は `--l0 <FT> --l1 <SIZE> --l2 <SIZE>`（既定 1536 / 16 / 32）。
`--psqt` で PSQT shortcut layer を有効化できる。

#### 入力切替

排他的に 1 つだけ指定する（複数指定はエラー）:

| フラグ | 入力型 |
|---|---|
| (none) | `ShogiHalfKA_hm` |
| `--threat` | `ShogiHalfKaHmThreat` (`--threat-profile <full\|same-class\|same-class-major-pawn\|cross-side>`) |
| `--hand-threat` | `ShogiHalfKaHmHandThreat` |
| `--hand-threat-defensive` | `ShogiHalfKaHmHandThreatDefensive` |
| `--hand-count-dense` | `ShogiHalfKaHmHandCount` (L1 入力に 14 元 dense を concat) |

#### 出力バケット (`--bucket-mode`)

| 値 | 実装 | 説明 |
|---|---|---|
| `kingrank9` (default) | `ShogiKingRankBucket<9>` | 自玉の段で 0..=8 にバケット |
| `ply9` | `ShogiPlyBucket9` | 手数で 9 バケット。`--ply-bounds "b0,b1,...,b7"` で境界指定可 |
| `progress8kpabs` | `ShogiProgressKPAbs` | KP-Abs 進行度バケット。`--progress-coeff <coeff.bin>` 必須 |

#### Optimizer / その他

- `--optimizer <adamw|radam|ranger>`（既定 `ranger`）
- `--quantise-only --resume <ckpt>` で再量子化のみ実行可
- `--interleave-file-batches` / `--epoch-file-shuffle` / `--file-shuffle-seed`
  で複数ファイル混合とシャッフル制御

---

## 3. 入力 × ネットワーク 早見表

試作中のものもあり、削除されることもあります。

| 目的 | 入力 | ネットワーク | 例 |
|---|---|---|---|
| nnue-pytorch HalfKP 互換ネット | `HalfKP` | shogi_simple `256x2-32-32` | (A) |
| YaneuraOu 寄りの大きめ HalfKA_hm | `HalfKA_hm` | shogi_simple `1024x2-16-64` | (B) |
| LayerStack baseline | `HalfKA_hm` | shogi_layerstack 1536/16/32, `kingrank9` | (C) |
| 持ち駒 dense を加えて L1 強化 | `HandCount` | shogi_layerstack `--hand-count-dense` | (D) |
| Threat 入力で評価精度 push | `Threat` | shogi_layerstack `--threat --threat-profile cross-side` | (E) |
| 防御的 Drop 脅威の追加 | `HandThreat defensive` | shogi_layerstack `--hand-threat-defensive` | (F) |

### CLI 例

(A) HalfKP 256x2-32-32:

```bash
cargo run --release --example shogi_simple -- \
    --features halfkp --arch 256x2-32-32 \
    --data data/train.bin --net-id shogi-halfkp-256
```

(B) HalfKA_hm 1024x2-16-64:

```bash
cargo run --release --example shogi_simple -- \
    --features halfka-hm --arch 1024x2-16-64 \
    --data data/train.bin --net-id shogi-halfka-hm-1024
```

(C) LayerStack baseline:

```bash
cargo run --release --example shogi_layerstack -- \
    --data data/train.bin \
    --l0 1536 --l1 16 --l2 32 \
    --bucket-mode kingrank9 \
    --net-id shogi-ls-1536
```

(D) LayerStack + HandCount Dense:

```bash
cargo run --release --example shogi_layerstack -- \
    --data data/train.bin --hand-count-dense \
    --bucket-mode kingrank9 \
    --net-id shogi-ls-1536-hcd14
```

(E) LayerStack + Threat (cross-side profile):

```bash
cargo run --release --example shogi_layerstack -- \
    --data data/train.bin \
    --threat --threat-profile cross-side \
    --bucket-mode kingrank9 \
    --net-id shogi-ls-1536-threat-xs
```

(F) LayerStack + HandThreat defensive:

```bash
cargo run --release --example shogi_layerstack -- \
    --data data/train.bin --hand-threat-defensive \
    --bucket-mode kingrank9 \
    --net-id shogi-ls-1536-htdef
```
