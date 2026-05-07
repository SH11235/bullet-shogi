use std::fmt::Debug;

use bullet_trainer::run::logger::ansi;

/// WDL lambda scheduling. Types implementing this trait output a WDL lambda
/// at each point in training, indexed by batch and superbatch.
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
        let grad = (self.end - self.start) / (max - 1).max(1) as f32;
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
        let midpoint = self.first_scheduler_final_superbatch;

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
