pub(crate) mod builder;
mod dataloader;
pub mod loader;
mod save;
pub mod validate;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};
use bullet_compiler::tensor::TValue;
use bullet_trainer::{
    Trainer,
    model::save::SavedFormat,
    optimiser::OptimiserState,
    run::{self, dataloader::PreparedBatchHost, logger},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::ExecutionContext,
    trainer::{
        schedule::{TrainingSchedule, lr::LrScheduler, wdl::WdlScheduler},
        settings::LocalSettings,
    },
    value::{
        dataloader::ValueDataLoader,
        loader::{DefaultDataLoader, LoadableDataType, WrmTargetParams},
    },
};

use crate::value::loader::PreparedData;

/// Value network trainer, generally for training NNUE networks.
pub struct ValueTrainer<
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
>(Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>);

impl<Opt, Inp, Out> std::ops::Deref for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    type Target = Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Opt, Inp, Out> std::ops::DerefMut for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;
type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

#[derive(Clone)]
pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    saved_format: Vec<SavedFormat>,
    /// `Some(params)` のとき教師 score を WRM target に変換する。
    /// builder の `use_win_rate_model(WrmTargetParams)` で設定。
    wrm_target: Option<WrmTargetParams>,
    wdl: bool,
    /// `Some(cap)` のとき `|score| >= cap` の局面を loss から除外。
    /// builder の `score_drop_abs(cap)` で設定。
    score_drop_abs: Option<u16>,
}

impl<Inp: SparseInputType, Out> ValueTrainerState<Inp, Out>
where
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn prepare(
        &self,
        batch: &[Inp::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> PreparedBatchHost {
        PreparedBatchHost::from(PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.wrm_target,
            self.wdl,
            batch,
            threads,
            blend,
            scale,
            self.score_drop_abs,
        ))
    }
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn run(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) {
        self.run_internal(schedule, settings, dataloader, None, 1, |_, _, _| {});
    }

    pub fn run_with_validation(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
        positions: &[Inp::RequiredDataType],
        batch_size: usize,
        callback: impl FnMut(usize, f64, Vec<f32>),
    ) {
        assert!(!positions.is_empty(), "validation positions must not be empty");
        assert!(batch_size > 0, "validation batch size must be positive");
        self.run_internal(schedule, settings, dataloader, Some(positions), batch_size, callback);
    }

    fn run_internal(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
        validation_positions: Option<&[Inp::RequiredDataType]>,
        validation_batch_size: usize,
        mut validation_callback: impl FnMut(usize, f64, Vec<f32>),
    ) {
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble", "34;1"));

        schedule.display();
        settings.display();

        if settings.test_set.is_some() {
            println!(
                "{}",
                logger::ansi("Warning: Validation data not currently implemented! Please bother me on discord.", "31")
            )
        }

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.wrm_target,
            self.state.wdl,
            schedule.eval_scale,
            self.state.score_drop_abs,
            dataloader.clone(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();

        let steps = schedule.steps;

        let error_record = RefCell::new(Vec::new());
        let validation_loss = RefCell::new((steps.start_superbatch, 0.0f64, 0usize));
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        self.train_custom(
            run::schedule::TrainingSchedule {
                steps,
                log_rate: 128,
                lr_schedule: Box::new(|a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader { steps, threads: settings.threads, dataloader, wdl: schedule.wdl_scheduler.clone() },
            |_, superbatch, curr_batch, error| {
                if validation_positions.is_some() {
                    let mut accumulated = validation_loss.borrow_mut();
                    if accumulated.0 != superbatch {
                        *accumulated = (superbatch, 0.0, 0);
                    }
                    accumulated.1 += f64::from(error);
                    accumulated.2 += 1;
                }
                loss_sum += error;
                ticks_since_last += 1.0;

                if curr_batch % 32 == 0
                    || (steps.batches_per_superbatch < 32 && curr_batch == steps.batches_per_superbatch)
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, steps.batches_per_superbatch as f32);

                    error_record.borrow_mut().push((superbatch, curr_batch, normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |trainer, superbatch| {
                if let Some(positions) = validation_positions {
                    let (_, sum, count) = *validation_loss.borrow();
                    let train_loss = if count == 0 { f64::NAN } else { sum / count as f64 };
                    let outputs = Self::eval_batch(trainer, positions, validation_batch_size);
                    validation_callback(superbatch, train_loss, outputs);
                }

                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));

                    if let Some(ref callback) = settings.on_checkpoint_saved {
                        callback(superbatch);
                    }
                }
            },
        )
        .unwrap();
    }

    fn eval_batch(
        trainer: &mut Trainer<ExecutionContext, Opt, ValueTrainerState<Inp, Out>>,
        positions: &[Inp::RequiredDataType],
        batch_size: usize,
    ) -> Vec<f32> {
        let mut all_outputs = Vec::with_capacity(positions.len());
        let device = trainer.optimiser.model.device();
        let stream = device.new_stream().unwrap();
        // Cache output tensors per chunk size: every chunk is `batch_size`
        // except possibly the final partial one, so this allocates at most
        // twice instead of once per chunk. `forward` overwrites the tensors,
        // but each chunk's values are copied to host before the next call.
        let mut cached_outputs: Option<(usize, _)> = None;
        for chunk in positions.chunks(batch_size) {
            let n = chunk.len();
            if cached_outputs.as_ref().map(|(size, _)| *size) != Some(n) {
                trainer.optimiser.model.set_fwd_batch_size(n).unwrap();
                let outputs = trainer.optimiser.model.make_forward_output_tensors(n).unwrap();
                cached_outputs = Some((n, outputs));
            }
            let (_, outputs) = cached_outputs.as_ref().unwrap();
            let host_data = trainer.state.prepare(chunk, 1, 1.0, 1.0);
            let model = &trainer.optimiser.model;
            let inputs = host_data.to_device(&device).unwrap();
            model.forward(&stream, &inputs, outputs).unwrap().value().unwrap();
            let output = outputs.get("outputs/output").unwrap().clone();
            let TValue::F32(values) = output.to_host().unwrap() else { panic!() };
            all_outputs.extend_from_slice(&values);
        }
        all_outputs
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        self.0.optimiser.model.set_fwd_batch_size(1).unwrap();

        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let host_data = self.state.prepare(&[pos], 1, 1.0, 1.0);

        let model = &self.optimiser.model;
        let device = model.device();
        let stream = device.new_stream().unwrap();

        let inputs = host_data.to_device(&device).unwrap();
        let outputs = model.make_forward_output_tensors(1).unwrap();
        model.forward(&stream, &inputs, &outputs).unwrap().value().unwrap();

        let output = outputs.get("outputs/output").unwrap().clone();
        let TValue::F32(output) = output.to_host().unwrap() else { panic!() };
        output
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let vals = self.eval_raw_output(fen);

        match vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                (win + draw / 2.0) / (win + draw + loss)
            }
            [score] => score,
            _ => panic!("Invalid output size!"),
        }
    }

    pub fn measure_max_cpu_throughput(
        &self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) {
        let steps = schedule.steps;
        let threads = settings.threads;
        let wdl = schedule.wdl_scheduler.clone();
        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.wrm_target,
            self.state.wdl,
            schedule.eval_scale,
            self.state.score_drop_abs,
            dataloader.clone(),
        );

        let dataloader = ValueDataLoader { steps, threads, dataloader, wdl };

        self.0.measure_max_cpu_throughput(dataloader, steps).unwrap()
    }
}
