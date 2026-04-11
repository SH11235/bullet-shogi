//! Threat pair 除外 profile (runtime 選択)
//!
//! 学習・評価時に CLI オプションで profile を指定する。
//! rshogi 側は compile-time (feature flag) で同一ロジックを実装。
//!
//! ## Profile 一覧
//!
//! | id | CLI 値 | 除外内容 |
//! |----|--------|---------|
//! | 0 | `full` | なし (Baseline) |
//! | 1 | `same-class` | 同種ペア全除外 |
//! | 2 | `same-class-major-pawn` | 同種 + 大駒→歩除外 |
//! | 10 | `cross-side` | cross-side 異種ペアのみ (増やすアプローチ) |
//!
//! 仕様: rshogi `docs/threat_spec.md` Exclusion profiles セクション

/// Threat pair 除外 profile
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreatProfile {
    /// Profile 0: 全 pair (Baseline)
    Full,
    /// Profile 1: 同種ペア全除外
    SameClass,
    /// Profile 2: 同種 + 大駒→歩除外
    SameClassMajorPawn,
    /// Profile 10: cross-side 異種ペアのみ
    CrossSide,
}

impl ThreatProfile {
    /// CLI 文字列から変換
    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Self::Full),
            "same-class" => Some(Self::SameClass),
            "same-class-major-pawn" => Some(Self::SameClassMajorPawn),
            "cross-side" => Some(Self::CrossSide),
            _ => None,
        }
    }

    /// quantised.bin に書き込む profile ID
    pub fn profile_id(self) -> u32 {
        match self {
            Self::Full => 0,
            Self::SameClass => 1,
            Self::SameClassMajorPawn => 2,
            Self::CrossSide => 10,
        }
    }

    /// pair を除外すべきかどうか判定する
    ///
    /// rshogi の `threat_exclusion::is_excluded` と同一ロジック。
    ///
    /// # 引数
    /// - `as_`: attacker side (0=friend, 1=enemy)
    /// - `ac`: attacker class index (0..8)
    /// - `ds`: attacked side (0=friend, 1=enemy)
    /// - `dc`: attacked class index (0..8)
    pub fn is_excluded(self, as_: usize, ac: usize, ds: usize, dc: usize) -> bool {
        match self {
            Self::Full => false,
            Self::SameClass => ac == dc,
            Self::SameClassMajorPawn => ac == dc || (ac >= 5 && dc == 0),
            Self::CrossSide => as_ == ds || ac == dc,
        }
    }

    /// 利用可能な profile 名の一覧（ヘルプ表示用）
    pub fn available() -> &'static str {
        "full, same-class, same-class-major-pawn, cross-side"
    }
}

impl std::fmt::Display for ThreatProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::SameClass => write!(f, "same-class"),
            Self::SameClassMajorPawn => write!(f, "same-class-major-pawn"),
            Self::CrossSide => write!(f, "cross-side"),
        }
    }
}
