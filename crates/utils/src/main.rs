mod convert;
mod count_buckets;
mod interleave;
mod montybinpack;
mod shuffle;
mod validate;
mod viribinpack;

use structopt::StructOpt;

#[derive(StructOpt)]
pub enum Options {
    Convert(convert::ConvertOptions),
    Interleave(interleave::InterleaveOptions),
    Shuffle(shuffle::ShuffleOptions),
    Validate(validate::ValidateOptions),
    BucketCount(count_buckets::ValidateOptions),
    Montybinpack(montybinpack::MontyBinpackOptions),
    Viribinpack(viribinpack::ViriBinpackOptions),
}

fn main() -> anyhow::Result<()> {
    match Options::from_args() {
        Options::Convert(options) => options.run(),
        Options::Interleave(options) => options.run(),
        Options::Shuffle(options) => options.run(),
        Options::Validate(options) => options.run(),
        Options::BucketCount(options) => options.run(),
        Options::Montybinpack(options) => options.run(),
        Options::Viribinpack(options) => options.run(),
    }
}

struct Rand(u64);

impl Default for Rand {
    fn default() -> Self {
        Self::with_seed(Self::random_seed())
    }
}

impl Rand {
    const FALLBACK_SEED: u64 = 0xA076_1D64_78BD_642F;

    fn with_seed(seed: u64) -> Self {
        Self(if seed == 0 { Self::FALLBACK_SEED } else { seed })
    }

    fn random_seed() -> u64 {
        (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("valid").as_nanos()
            & 0xFFFF_FFFF_FFFF_FFFF) as u64
    }

    fn derive_seed(seed: u64, stream: u64) -> u64 {
        let mut x = seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        let x = x ^ (x >> 31);
        if x == 0 { Self::FALLBACK_SEED } else { x }
    }

    fn rand(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
