use std::fmt::Debug;

use bullet_trainer::run::logger::ansi;

/// WDL lambda scheduling. Types implementing this trait output a WDL lambda
/// at each point in training, indexed by batch and superbatch.
///
/// The data loader may call `blend` with `superbatch > max`, because batches
/// are prefetched past `end_superbatch`. Implementations must saturate at
/// the endpoint value in that case, mirroring `LrScheduler` decay
/// schedulers. The downstream consumer (`value/loader.rs`) asserts the
/// returned lambda lies in `[0, 1]`; constructing a scheduler with endpoint
/// values outside `[0, 1]` therefore violates the consumer's contract.
pub trait WdlScheduler: Clone + Debug + Send + Sync + 'static {
    /// The WDL lambda for the current batch and superbatch.
    /// Most schedulers do not depend on the batch index.
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32;
    /// A colourful display representation of the WDL lambda scheduler.
    fn colourful(&self) -> String;
}

/// A WDL-lambda that stays constant throughout training.
#[derive(Clone, Debug)]
pub struct ConstantWDL {
    pub value: f32,
}

impl WdlScheduler for ConstantWDL {
    fn blend(&self, _batch: usize, _superbatch: usize, _max: usize) -> f32 {
        self.value
    }

    fn colourful(&self) -> String {
        format!("constant {}", ansi(self.value, 31))
    }
}

/// A WDL-lambda that transitions between a start and end value over training.
#[derive(Clone, Debug)]
pub struct LinearWDL {
    pub start: f32,
    pub end: f32,
}

impl WdlScheduler for LinearWDL {
    fn blend(&self, _batch: usize, superbatch: usize, max: usize) -> f32 {
        // Saturate at both ends. The LR-side schedulers only guard the high
        // end because their formula reads `superbatch as f32 / max as f32`
        // which already evaluates to `0.0` at `superbatch == 0`; this
        // scheduler interpolates over `(superbatch - 1)` instead, so
        // `superbatch == 0` would produce `start - grad` without the high
        // guard, and the order below matters: the overshoot check must run
        // before the low-end guard so that a single-superbatch schedule
        // (`max == 1`) still saturates to `end` when the loader prefetches
        // past the schedule end.
        if superbatch >= max.max(2) {
            return self.end;
        }
        if superbatch <= 1 {
            return self.start;
        }
        let grad = (self.end - self.start) / (max - 1) as f32;
        self.start + grad * (superbatch - 1) as f32
    }

    fn colourful(&self) -> String {
        format!("linear taper start {} end {}", ansi(self.start, 31), ansi(self.end, 31))
    }
}

/// Warm up to a sub-scheduler over `warmup_batches` batches.
#[derive(Clone, Debug)]
pub struct Warmup<WDL: WdlScheduler> {
    pub inner: WDL,
    pub warmup_batches: usize,
}

impl<WDL: WdlScheduler> WdlScheduler for Warmup<WDL> {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        let base_wdl = self.inner.blend(batch, superbatch, max);
        // batch loops within superbatches, so we must check we're
        // actually at the start of training to correctly implement
        // warmup.
        if superbatch == 1 && batch < self.warmup_batches {
            // linearly interpolate up from base_wdl / warmup_batches
            base_wdl / (self.warmup_batches - batch) as f32
        } else {
            base_wdl
        }
    }

    fn colourful(&self) -> String {
        // < BASE_SCHEDULER_TEXT >, warmup over {} batches
        format!("{}, warmup over {} batches", self.inner.colourful(), ansi(self.warmup_batches, 31))
    }
}

/// Sequence two sub-schedulers, switching over at `first_scheduler_final_superbatch`
#[derive(Clone, Debug)]
pub struct Sequence<First: WdlScheduler, Second: WdlScheduler> {
    pub first: First,
    pub second: Second,
    pub first_scheduler_final_superbatch: usize,
}

impl<First: WdlScheduler, Second: WdlScheduler> WdlScheduler for Sequence<First, Second> {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        // Clamp both `superbatch` and `midpoint` to the overall schedule
        // length so that:
        //   * prefetched overshoot (`superbatch > max`) saturates at the
        //     value the inner scheduler would produce at `superbatch = max`,
        //     even when that point still sits inside the first scheduler's
        //     range (i.e. `max < midpoint`).
        //   * misconfigured schedules where
        //     `end_superbatch < first_scheduler_final_superbatch` (which can
        //     arise when resuming with a shortened budget) never feed the
        //     second scheduler a zero-length budget or a wrapped `usize`.
        let midpoint = self.first_scheduler_final_superbatch.min(max);
        let superbatch = superbatch.min(max);

        if superbatch <= midpoint {
            self.first.blend(batch, superbatch, midpoint)
        } else {
            self.second.blend(batch, superbatch - midpoint, max - midpoint)
        }
    }

    fn colourful(&self) -> String {
        // < LEFT_SCHEDULER_TEXT >, then after {} superbatches, < RIGHT_SCHEDULER_TEXT>
        format!(
            "{}, then after {} superbatches, {}",
            self.first.colourful(),
            ansi(self.first_scheduler_final_superbatch, 32),
            self.second.colourful()
        )
    }
}

/// Enum wrapper for WDL schedulers to allow runtime selection.
/// This is useful when the scheduler type is determined at runtime
/// based on command-line arguments.
#[derive(Clone, Debug)]
pub enum WdlSchedulerEnum {
    Constant(ConstantWDL),
    Linear(LinearWDL),
}

impl WdlSchedulerEnum {
    /// Creates a constant WDL scheduler with the given value.
    pub fn constant(value: f32) -> Self {
        Self::Constant(ConstantWDL { value })
    }

    /// Creates a linear WDL scheduler with start and end values.
    pub fn linear(start: f32, end: f32) -> Self {
        Self::Linear(LinearWDL { start, end })
    }
}

impl WdlScheduler for WdlSchedulerEnum {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        match self {
            Self::Constant(s) => s.blend(batch, superbatch, max),
            Self::Linear(s) => s.blend(batch, superbatch, max),
        }
    }

    fn colourful(&self) -> String {
        match self {
            Self::Constant(s) => s.colourful(),
            Self::Linear(s) => s.colourful(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    #[test]
    fn linear_wdl_endpoints() {
        let s = LinearWDL { start: 0.2, end: 0.8 };
        approx_eq(s.blend(0, 1, 3), 0.2);
        approx_eq(s.blend(0, 2, 3), 0.5);
        approx_eq(s.blend(0, 3, 3), 0.8);
    }

    #[test]
    fn linear_wdl_saturates_when_superbatch_exceeds_max() {
        // Reproduces the dataloader-prefetch case where superbatch overruns
        // end_superbatch: pre-fix this returned 1.5 and tripped the
        // `blend in [0, 1]` assertion in `value/loader.rs`.
        let s = LinearWDL { start: 0.0, end: 1.0 };
        approx_eq(s.blend(0, 4, 3), 1.0);
        approx_eq(s.blend(0, 100, 3), 1.0);
    }

    #[test]
    fn linear_wdl_clamps_subzero_superbatch() {
        let s = LinearWDL { start: 0.2, end: 0.8 };
        approx_eq(s.blend(0, 0, 3), 0.2);
        approx_eq(s.blend(0, 1, 1), 0.2);
    }

    #[test]
    fn linear_wdl_one_superbatch_schedule_saturates_overshoot_to_end() {
        // For a single-superbatch schedule (`max == 1`) the lone superbatch
        // is both start and end, but any prefetched overshoot must saturate
        // to `end` rather than collapsing to `start`. (Regression coverage
        // for the Codex review on PR #15.)
        let s = LinearWDL { start: 0.0, end: 1.0 };
        approx_eq(s.blend(0, 1, 1), 0.0);
        approx_eq(s.blend(0, 2, 1), 1.0);
        approx_eq(s.blend(0, 100, 1), 1.0);
    }

    #[test]
    fn linear_wdl_output_always_in_unit_interval_for_valid_endpoints() {
        let s = LinearWDL { start: 0.1, end: 0.9 };
        for sb in 0..200 {
            let v = s.blend(0, sb, 10);
            assert!((0.0..=1.0).contains(&v), "blend({sb}) = {v} out of range");
        }
    }

    #[test]
    fn linear_wdl_descending_lambda_saturates_correctly() {
        // start > end: saturation must return the literal endpoints, not the
        // numeric min/max of the range.
        let s = LinearWDL { start: 0.8, end: 0.2 };
        approx_eq(s.blend(0, 1, 5), 0.8);
        approx_eq(s.blend(0, 5, 5), 0.2);
        approx_eq(s.blend(0, 100, 5), 0.2);
    }

    #[test]
    fn warmup_linear_wdl_overshoot_saturates_to_end() {
        // Warmup only adjusts the very first superbatch, so any overshoot
        // must come from the inner LinearWDL saturation.
        let inner = LinearWDL { start: 0.0, end: 1.0 };
        let warmup = Warmup { inner, warmup_batches: 4 };
        approx_eq(warmup.blend(0, 4, 3), 1.0);
        approx_eq(warmup.blend(0, 999, 3), 1.0);
    }

    #[test]
    fn sequence_overshoot_uses_inner_saturation() {
        // After the midpoint the second scheduler receives shifted
        // (superbatch - midpoint, max - midpoint); overshoot must saturate.
        let first = LinearWDL { start: 0.0, end: 0.5 };
        let second = LinearWDL { start: 0.5, end: 1.0 };
        let seq = Sequence { first, second, first_scheduler_final_superbatch: 3 };
        approx_eq(seq.blend(0, 3, 6), 0.5);
        approx_eq(seq.blend(0, 6, 6), 1.0);
        approx_eq(seq.blend(0, 100, 6), 1.0);
    }

    #[test]
    fn sequence_overshoot_through_first_scheduler_saturates_at_max() {
        // `max < midpoint` (or `superbatch == max + 1 <= midpoint`): the
        // overshoot still routes through the first scheduler, and must
        // saturate at the value that scheduler would produce at
        // `superbatch == max`, rather than continuing to interpolate past
        // the overall schedule end. (Copilot review on PR #15.)
        let first = LinearWDL { start: 0.0, end: 0.5 };
        let second = LinearWDL { start: 0.5, end: 1.0 };
        let seq = Sequence { first, second, first_scheduler_final_superbatch: 10 };
        // max = 8, sb = 9: clamped midpoint = sb = 8 → first.end = 0.5.
        approx_eq(seq.blend(0, 9, 8), 0.5);
        approx_eq(seq.blend(0, 100, 8), 0.5);
    }

    #[test]
    fn sequence_with_misconfigured_max_does_not_panic() {
        // max < midpoint can occur when resuming with a shortened schedule.
        // Clamping `superbatch` and `midpoint` to `max` keeps the second
        // scheduler from being selected with a zero-length budget and avoids
        // any `usize` underflow on `max - midpoint`.
        let first = LinearWDL { start: 0.0, end: 0.5 };
        let second = LinearWDL { start: 0.5, end: 1.0 };
        let seq = Sequence { first, second, first_scheduler_final_superbatch: 10 };
        // clamped midpoint = sb = 5 → first.blend(_, 5, 5) → first.end = 0.5.
        approx_eq(seq.blend(0, 12, 5), 0.5);
    }
}
