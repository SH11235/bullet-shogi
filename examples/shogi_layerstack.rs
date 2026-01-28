/*
Shogi LayerStack NNUE Training Script

LayerStack アーキテクチャ（Stockfish NNUE v5+ 相当）を使用した将棋 NNUE 学習。
rshogi の推論実装と互換性のあるモデルを学習できる。

Usage:
    cargo run --release --example shogi_layerstack -- [OPTIONS]

Options:
    --features <SET>    Feature set (halfka-hm, halfka) (default: halfka-hm)
    --l0 <SIZE>         L0 (Feature Transformer) size (default: 1024)
    --l1 <SIZE>         L1 size (default: 16)
    --l2 <SIZE>         L2 size (default: 32)
    --data <PATH>       Training data path (comma-separated)
    --batch-size <N>    Batch size (default: 16384)
    --superbatches <N>  Number of superbatches (default: 100)
    --lr <RATE>         Initial learning rate (default: 0.001)
    --wdl <LAMBDA>      WDL lambda (default: 0.75)
    --scale <N>         Eval scale (default: 600)
    --save-rate <N>     Save interval (default: 10)
    --threads <N>       Number of threads (default: 4)
    --output <DIR>      Output directory (default: checkpoints)
    --net-id <NAME>     Network ID (default: shogi-layerstack)

Architecture:
    HalfKA_hm (73,305) -> L0 (1024) -> CReLU -> pairwise_mul -> 512
    concat(stm, ntm) -> 1024
    L1: 1024 -> 16×9buckets, select, SCReLU
    L2: 16 -> 32×9buckets, select, SCReLU
    L3: 32 -> 9buckets, select

Examples:
    # Train with default settings
    cargo run --release --example shogi_layerstack -- --data data/train.bin

    # Train with custom L0 size
    cargo run --release --example shogi_layerstack -- --l0 1536 --data data/train.bin

    # Train with HalfKA (non-mirrored)
    cargo run --release --example shogi_layerstack -- --features halfka --data data/train.bin
*/

use std::path::PathBuf;

use bullet_lib::{
    game::{
        inputs::{ShogiHalfKA, ShogiHalfKA_hm, SparseInputType},
        outputs::ShogiKingRankBucket,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{self, AdamWParams, RangerParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use clap::{Parser, ValueEnum};

// =============================================================================
// CLI Arguments
// =============================================================================

/// Feature set selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum FeatureSet {
    /// HalfKA_hm - Half-Mirrored King-All (73,305 dimensions)
    #[default]
    HalfkaHm,
    /// HalfKA - King-All non-mirrored (138,510 dimensions)
    Halfka,
}

/// Optimizer selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizerType {
    /// AdamW - fast convergence but may be unstable with sparse inputs
    AdamW,
    /// Ranger - RAdam + Lookahead (recommended by nnue-pytorch)
    #[default]
    Ranger,
}

#[derive(Parser, Debug)]
#[command(name = "shogi_layerstack")]
#[command(about = "Shogi LayerStack NNUE training script")]
struct Args {
    /// Feature set (halfka-hm, halfka)
    #[arg(long, value_enum, default_value = "halfka-hm")]
    features: FeatureSet,

    /// Optimizer (adamw, ranger)
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerType,

    /// L0 (Feature Transformer) size
    #[arg(long, default_value = "1024")]
    l0: usize,

    /// L1 (LayerStack first layer) size
    #[arg(long, default_value = "16")]
    l1: usize,

    /// L2 (LayerStack second layer) size
    #[arg(long, default_value = "32")]
    l2: usize,

    /// Training data path (comma-separated for multiple files)
    #[arg(long, default_value = "data/train.bin")]
    data: String,

    /// Batch size
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of superbatches
    #[arg(long, default_value = "100")]
    superbatches: usize,

    /// Initial learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// WDL lambda (0.0=eval only, 1.0=game result only)
    #[arg(long, default_value = "0.75")]
    wdl: f32,

    /// Eval scale for training target sigmoid(score / scale)
    #[arg(long, default_value = "600")]
    scale: i32,

    /// Save interval (superbatches)
    #[arg(long, default_value = "10")]
    save_rate: usize,

    /// Number of threads
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Output directory
    #[arg(long, default_value = "checkpoints")]
    output: PathBuf,

    /// Network ID
    #[arg(long, default_value = "shogi-layerstack")]
    net_id: String,

    /// Quantization factor QA (for L0)
    #[arg(long, default_value = "255")]
    qa: i16,

    /// Quantization factor QB (for later layers)
    #[arg(long, default_value = "64")]
    qb: i16,

    /// Weight decay (L2 regularization)
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Resume from checkpoint path
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Only re-quantise checkpoint (no training, requires --resume)
    #[arg(long)]
    quantise_only: bool,
}

// =============================================================================
// Main
// =============================================================================

/// 出力バケット数 (ShogiKingRankBucket と同じ値)
const NUM_BUCKETS: usize = 9;

fn main() {
    let args = Args::parse();

    // Architecture sizes (configurable via CLI)
    let l0_size = args.l0;
    let l1_size = args.l1; // L1 output size (includes +1 for skip connection)
    let l1_effective = l1_size - 1; // L1 effective output (excluding skip)
    let l2_input = l1_effective * 2; // SCReLU² + CReLU concat
    let l2_size = args.l2; // L2 output size

    // Quantization factors
    let qa = args.qa;
    let qb = args.qb;

    // Feature set info
    let (feature_name, input_size) = match args.features {
        FeatureSet::HalfkaHm => ("HalfKA_hm", ShogiHalfKA_hm.num_inputs()),
        FeatureSet::Halfka => ("HalfKA", ShogiHalfKA.num_inputs()),
    };

    // Optimizer name
    let optimizer_name = match args.optimizer {
        OptimizerType::AdamW => "AdamW",
        OptimizerType::Ranger => "Ranger",
    };

    // L1 input dimension after pairwise_mul and concat
    // L0 -> CReLU -> pairwise_mul (halves) -> concat(stm, ntm) -> L1 input
    let l1_input_dim = l0_size; // l0_size/2 * 2 = l0_size

    // Print configuration
    println!("=== Shogi LayerStack NNUE Training (sfnnwop-1536 compatible) ===");
    println!("Features: {} ({} dimensions)", feature_name, input_size);
    println!("Architecture:");
    println!("  L0: {} -> {} -> CReLU -> pairwise_mul -> {}", input_size, l0_size, l0_size / 2);
    println!("  concat(stm, ntm) -> {}", l1_input_dim);
    println!("  L1: {} -> {}×{} buckets, select, split [{}, 1]", l1_input_dim, l1_size, NUM_BUCKETS, l1_effective);
    println!("      [0..{}] -> SCReLU² + CReLU concat -> {}", l1_effective, l2_input);
    println!("      [{}] -> skip connection", l1_effective);
    println!("  L2: {} -> {}×{} buckets, select, CReLU", l2_input, l2_size, NUM_BUCKETS);
    println!("  L3: {} -> {} buckets, select + skip", l2_size, NUM_BUCKETS);
    println!("Output Buckets: {} (king rank based)", NUM_BUCKETS);
    println!("Optimizer: {}", optimizer_name);
    println!("Weight decay: {}", args.weight_decay);
    println!("Scale: {}", args.scale);
    println!("Quantization: QA={}, QB={}", qa, qb);
    println!("Batch size: {}", args.batch_size);
    println!("Superbatches: {}", args.superbatches);
    println!("Learning rate: {}", args.lr);
    println!("WDL lambda: {}", args.wdl);
    println!("Save rate: {}", args.save_rate);
    println!("Threads: {}", args.threads);
    println!("Output: {}", args.output.display());
    println!("Net ID: {}", args.net_id);
    println!("Data: {}", args.data);
    println!("=======================================");

    // Training schedule
    let schedule = TrainingSchedule {
        net_id: args.net_id,
        eval_scale: args.scale as f32,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch: 6104, // ~100M positions/superbatch
            start_superbatch: 1,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: args.wdl },
        lr_scheduler: lr::StepLR { start: args.lr, gamma: 0.3, step: 30 },
        save_rate: args.save_rate,
    };

    // Local settings
    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let settings =
        LocalSettings { threads: args.threads, test_set: None, output_directory: output_dir, batch_queue_size: 64 };

    // Data loader
    let data_files_owned: Vec<String> = if args.quantise_only {
        let resume_path = args.resume.as_ref().expect("--quantise-only requires --resume");
        let quantised = resume_path.join("quantised.bin");
        if quantised.exists() {
            vec![quantised.to_str().unwrap().to_string()]
        } else {
            vec![resume_path.join("raw.bin").to_str().unwrap().to_string()]
        }
    } else {
        args.data.split(',').map(|s| s.to_string()).collect()
    };
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();
    let data_loader = DirectSequentialDataLoader::new(&data_files_ref);

    // SavedFormat configuration
    // LayerStack format with NNUE header and Factorizer

    // NNUE version (YaneuraOu/Stockfish compatible)
    const NNUE_VERSION: u32 = 0x7AF32F16;

    // Calculate FV_SCALE = (127 × QB) / scale (rounded)
    // bullet 流: L1/L2/L3 全て同じ量子化スケール (127 × QB) を使用
    let bias_scale = 127 * i32::from(qb); // 8128
    let fv_scale = (bias_scale + args.scale / 2) / args.scale;

    // Build architecture string with LayerStack metadata
    let arch_str = format!(
        "Features={}[{}->{}]-LayerStack,fv_scale={},l0={},l1={},l1_effective={},l2_input={},l2={},buckets={},qa={},qb={},scale={}",
        feature_name,
        input_size,
        l0_size,
        fv_scale,
        l0_size,
        l1_size,
        l1_effective,
        l2_input,
        l2_size,
        NUM_BUCKETS,
        qa,
        qb,
        args.scale
    );
    let arch_bytes = arch_str.as_bytes();

    // Build NNUE header
    let mut header = Vec::new();
    header.extend_from_slice(&NNUE_VERSION.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // hash (dummy)
    header.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
    header.extend_from_slice(arch_bytes);

    // Layer hashes (dummy values)
    let ft_hash = 0u32.to_le_bytes().to_vec();
    let network_hash = 0u32.to_le_bytes().to_vec();

    println!("Architecture string: {}", arch_str);
    println!("Bias scale (L1/L2/L3): {}", bias_scale);
    println!("FV_SCALE: {}", fv_scale);

    let save_format: Vec<SavedFormat> = vec![
        // NNUE Header
        SavedFormat::custom(header),
        // FeatureTransformer layer hash
        SavedFormat::custom(ft_hash),
        // L0 (Feature Transformer) with Factorizer merged
        // Order: biases first, then weights (Stockfish/rshogi convention)
        SavedFormat::id("l0b").round().quantise::<i16>(qa),
        SavedFormat::id("l0w")
            .transform(|store, weights| {
                let factoriser = &store.get("l0f").values;
                // Factorizer is not bucketed, just add directly
                weights.into_iter().zip(factoriser.iter().cycle()).map(|(a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(qa),
        // Network layer hash
        SavedFormat::custom(network_hash),
        // L1-L3 (LayerStack layers) - all use same quantization (bullet 流)
        // bias: 127 × QB = 8128, weight: QB = 64
        // Order: biases first, then weights (transposed for row-major)
        SavedFormat::id("l1b").round().quantise::<i32>(bias_scale),
        SavedFormat::id("l1w").transpose().round().quantise::<i8>(qb),
        SavedFormat::id("l2b").round().quantise::<i32>(bias_scale),
        SavedFormat::id("l2w").transpose().round().quantise::<i8>(qb),
        SavedFormat::id("l3b").round().quantise::<i32>(bias_scale),
        SavedFormat::id("l3w").transpose().round().quantise::<i8>(qb),
    ];

    // Build trainer macro (network construction only)
    macro_rules! build_trainer {
        ($opt:expr, $input:expr) => {{
            ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .output_buckets(ShogiKingRankBucket)
                .save_format(&save_format)
                .loss_fn(|output, target| output.sigmoid().squared_error(target))
                .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                    // Factorizer for L0
                    let l0f = builder.new_weights("l0f", Shape::new(l0_size, input_size), InitSettings::Zeroed);

                    // L0 (Feature Transformer)
                    let mut l0 = builder.new_affine("l0", input_size, l0_size);
                    l0.init_with_effective_input_size(32);
                    l0.weights = l0.weights + l0f;

                    // LayerStack layers (each with NUM_BUCKETS outputs)
                    // Default (sfnnwop-1536 compatible): L1=16 (15+1 skip), L2_in=30, L2_out=32
                    let l1 = builder.new_affine("l1", l1_input_dim, NUM_BUCKETS * l1_size);
                    let l2 = builder.new_affine("l2", l2_input, NUM_BUCKETS * l2_size);
                    let l3 = builder.new_affine("l3", l2_size, NUM_BUCKETS);

                    // Forward pass
                    // L0: input -> affine -> CReLU -> pairwise_mul
                    let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
                    let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();

                    // Concatenate perspectives
                    let combined = stm_hidden.concat(ntm_hidden);

                    // L1: select → slice で分割 [l1_effective, 1]
                    let l1_out = l1.forward(combined).select(output_buckets);
                    let l1_main = l1_out.slice_rows(0, l1_effective); // [0..l1_effective]
                    let l1_skip = l1_out.slice_rows(l1_effective, l1_size); // [l1_effective..l1_size]

                    // SCReLU² + CReLU concat
                    // screlu() = clamp(x, 0, 1)² なので追加の二乗は不要
                    let l1_sqr = l1_main.screlu(); // clamp(x, 0, 1)² - 既に二乗済み
                    let l1_crelu = l1_main.crelu(); // clamp(x, 0, 1)
                    let l2_input = l1_sqr.concat(l1_crelu); // [30]

                    // L2: select → CReLU
                    let l2_out = l2.forward(l2_input).select(output_buckets).crelu();

                    // L3: select + skip connection
                    let l3_out = l3.forward(l2_out).select(output_buckets);
                    l3_out + l1_skip // Skip connection (+ 演算子で加算)
                })
        }};
    }

    macro_rules! maybe_run_or_quantise {
        ($trainer:expr) => {{
            if args.quantise_only {
                let resume_path = args.resume.as_ref().expect("--quantise-only requires --resume");
                let resume_str = resume_path.to_str().unwrap();
                println!("Loading checkpoint from {}...", resume_str);
                $trainer.load_from_checkpoint(resume_str);

                let output_dir = args.output.to_str().unwrap_or("checkpoints");
                let output_path = format!("{}/requantised.bin", output_dir);
                std::fs::create_dir_all(output_dir).unwrap_or(());

                println!("Saving re-quantised weights to {}...", output_path);
                $trainer.save_quantised(&output_path).expect("Failed to save quantised weights");
                println!("Done!");
            } else {
                if let Some(ref resume_path) = args.resume {
                    let resume_str = resume_path.to_str().unwrap();
                    println!("Resuming from checkpoint: {}", resume_str);
                    $trainer.load_from_checkpoint(resume_str);
                }
                $trainer.run(&schedule, &settings, &data_loader);
            }
        }};
    }

    // Run training based on feature set and optimizer
    match (args.features, args.optimizer) {
        (FeatureSet::HalfkaHm, OptimizerType::AdamW) => {
            let mut trainer = build_trainer!(optimiser::AdamW, ShogiHalfKA_hm);
            trainer.optimiser.set_params(AdamWParams { decay: args.weight_decay, ..Default::default() });
            // Stricter weight clipping for factorized weights
            let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
            trainer.optimiser.set_params_for_weight("l0w", stricter);
            trainer.optimiser.set_params_for_weight("l0f", stricter);
            maybe_run_or_quantise!(trainer);
        }
        (FeatureSet::HalfkaHm, OptimizerType::Ranger) => {
            let mut trainer = build_trainer!(optimiser::Ranger, ShogiHalfKA_hm);
            trainer.optimiser.set_params(RangerParams { decay: args.weight_decay, ..Default::default() });
            // Stricter weight clipping for factorized weights
            let stricter = RangerParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
            trainer.optimiser.set_params_for_weight("l0w", stricter);
            trainer.optimiser.set_params_for_weight("l0f", stricter);
            maybe_run_or_quantise!(trainer);
        }
        (FeatureSet::Halfka, OptimizerType::AdamW) => {
            let mut trainer = build_trainer!(optimiser::AdamW, ShogiHalfKA);
            trainer.optimiser.set_params(AdamWParams { decay: args.weight_decay, ..Default::default() });
            // Stricter weight clipping for factorized weights
            let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
            trainer.optimiser.set_params_for_weight("l0w", stricter);
            trainer.optimiser.set_params_for_weight("l0f", stricter);
            maybe_run_or_quantise!(trainer);
        }
        (FeatureSet::Halfka, OptimizerType::Ranger) => {
            let mut trainer = build_trainer!(optimiser::Ranger, ShogiHalfKA);
            trainer.optimiser.set_params(RangerParams { decay: args.weight_decay, ..Default::default() });
            // Stricter weight clipping for factorized weights
            let stricter = RangerParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
            trainer.optimiser.set_params_for_weight("l0w", stricter);
            trainer.optimiser.set_params_for_weight("l0f", stricter);
            maybe_run_or_quantise!(trainer);
        }
    }
}
