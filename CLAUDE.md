# bullet-shogi リポ運用規約 (Claude Code 向け)

bullet (https://github.com/jw1912/bullet) の将棋対応フォーク。Claude Code
セッションが本リポで作業するときに必ず従う運用規約。人間 collaborator も読む
想定だが、現状 user は SH11235 単独。

## プロジェクト概要

- **目的**: 将棋 NNUE 学習のための高速トレーナー
- **ベース**: bullet (チェス用 NNUE 学習ライブラリ) のフォーク
- **言語**: Rust 2024 edition + CUDA
- **MSRV**: `Cargo.toml` の `workspace.package.rust-version` で固定

## ビルド

```bash
# CUDA backend (主用途、学習例の必須構成)
cargo build --release --features cuda --example shogi_layerstack

# mock backend (CUDA 無し環境、学習は走らない)
cargo build --release
```

`bullet_lib` は default features を持たず、CUDA 学習用 example は
`--features cuda` を必須とする。指定なしビルドは mock runtime にリンクされ、
trainer 実行時に panic する。

## CI 規約 (push 前必須)

PR / `git push` の前に以下 3 step を必ず走らせ、全 pass を確認する:

```bash
cargo fmt --check
cargo clippy --release --features cuda --all-targets -- -D warnings
cargo test  --release --features cuda
```

`.github/workflows/checks.yaml` は GitHub-hosted runner に CUDA が無いため
GPU 依存 path を skip / exclude するが、**本機 (CUDA install 済) では
`--features cuda` 込み check を必須**とする。CI green でも local check skip は
規約違反 (CI が見えない領域に未検出 lint / test fail が溜まる)。

## rust-version (MSRV) 規約

`Cargo.toml` の `workspace.package.rust-version` は **保守的に下げない**、
実際に開発で使う toolchain と揃える。本リポは個人プロジェクトで外部 consumer
ゼロ、crates.io 公開も無い。低い MSRV は clippy (`clippy::incompatible_msrv`)
で `let-else` / let-chain / `div_ceil` / `is_multiple_of` 等の便利 syntax /
method 利用を誤ってエラー扱いさせる害がある。

## upstream 同期規約

本リポは jw1912/bullet のフォークで、定期的に upstream/main を取り込む。

- upstream 取り込みは feature branch (例: `upstream-rewrite-integration`)
  経由、main 相当 (`shogi-support`) へは PR 経由でのみ merge
- upstream 由来のバグ修正は `upstream-fix-<topic>` branch を **upstream/main
  派生** で別途立て、本家への cross-repo PR 候補として保持する
- 将棋固有改修 (`shogi_layerstack` example, shogi 用 input type, packed_sfen
  等) は upstream PR に混ぜない (取り込み拒否されるため)

## commit / PR 規約

- commit message は日本語可、`{scope}: {summary}` 形式
  (例: `value/builder: entry_weights を loss に再代入してマスクを実効化`)
- main 相当 (`shogi-support`) への直接 push 禁止、必ず PR 経由
- `git push --force` は main 相当 / merge 済 branch に絶対しない
- `--no-verify` 禁止 (CI を skip して push しない)
- perf 改善 commit は計測値 (pos/sec mean、loss 軌跡、棋力 SPRT 等) を
  message に含める
- 否定結果も commit を残す (revert commit + 経緯 doc の 2 操作)

## コードコメント規約

コード内コメント (`.rs` / `Cargo.toml` / `.yaml` / `.sh` 等) は **初見の Rust
開発者がそのファイル単独で読んで意味が通る** ものに限る。以下は禁止:

- **作業ログ語彙**: 「削除済」「追加した」「今回」「以前は」「N → M に変更」
- **PM シーケンスラベル**: 「Stage N」「Phase N」「Step N」「Round N」
  「Iteration N」「Sprint N」「M1 / M2 / マイルストーン N」が
  プロジェクト/作業の順序を指すとき。
  ただし **algorithm の pass / step を指す場合は許容** (例:
  `Phase 1 of inverse-index sparse_ft_backward: ...` 等は OK)
- **Issue / PR 番号参照**: 「Issue #N」「PR #N で」「#NN review で」
- **Migration history**: 「ここから昇格」「以前は…にあった」「旧 path は」、
  「旧 X 互換」型の history 言及
- **実験 ID 参照**: 「v100 系は…」「v82 で…」等、`docs/experiments/`
  配下のローカル実験番号への言及。これらは本リポの実験管理上の連番で、
  外部 reader には解読不能

これらの情報は git log / PR description / `docs/` 配下に置く。コード内に書いて
よいのは:
- 非自明な不変条件 (例:「caller が `n_pos * MAX_ACTIVE` を保証する」)
- 言語仕様で表せない constraint の理由 (例:「`EliminateUnusedOperations`
  が出力未登録 op を削除するため乗算結果を変数に保持する」)
- コードを読んでも分からない外部参照 (例:「YaneuraOu PackedSfen 形式に追従」、
  論文 / upstream ライブラリの algorithm 出典)

## ドキュメント規約

`docs/` 直下 / `README.md` の `.md` も上記コードコメント規約と同じ「初見 OSS
reader 視点」を採る。加えて以下:

- **doc 冒頭の dated header 禁止**: `作成: YYYY-MM-DD`、`改訂: YYYY-MM-DD` 等。
  履歴は git log で見る。ADR のように Status / Date field が doc の意味の一部
  となる場合は OK
- **directory tree / 構成図は現状を反映**。「将来こうする」予定や削除済
  directory を残さない
- **dated 検証ブロック禁止**: 「2026-05-11 に X 環境で確認」型の log は
  `docs/experiments/` 専用、reference doc に混ぜない
- **略語は README / 専用 glossary 章で一回だけ定義**、コード内では glossary 登録
  済の略語を素のまま使ってよい

`docs/experiments/` は計測ログ / 仮説検証経緯の置き場であり、上記制約の対象外
(実験 ID / 日付 / PR 番号を含んでよい history doc)。

## 関連リポジトリ

```
~/development/
├── rshogi/               # 推論エンジン本体 (USI engine)
│   └── crates/engine-core/src/nnue/   # NNUE 推論実装
├── nnue-pytorch/         # オリジナル (chess)
├── nnue-pytorch-myfork/  # 自前 fork (将棋)
├── nnue-pytorch-nodchip/ # 将棋向け先行 fork
├── bullet-shogi/         # 本リポ (学習側)
├── bullet/               # bullet 原典 (参照用)
└── Stockfish/            # Stockfish 原典 (参照用)
```

## 言語設定

ユーザーへの返答は日本語で行うこと。
