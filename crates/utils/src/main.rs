mod convert;
mod count_buckets;
mod interleave;
mod montybinpack;
mod output;
mod shuffle;
mod validate;
mod viribinpack;

use clap::{CommandFactory, Parser, error::ErrorKind};

#[derive(Parser)]
#[command(version, propagate_version = true)]
pub enum Options {
    Convert(convert::ConvertOptions),
    Interleave(interleave::InterleaveOptions),
    Shuffle(shuffle::ShuffleOptions),
    Validate(validate::ValidateOptions),
    BucketCount(count_buckets::ValidateOptions),
    #[command(subcommand)]
    Montybinpack(montybinpack::MontyBinpackOptions),
    #[command(subcommand)]
    Viribinpack(viribinpack::ViriBinpackOptions),
}

impl Options {
    const MIN_INTERLEAVE_INPUTS: usize = 2;

    fn parse_checked<I, T>(arguments: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let options = Self::try_parse_from(arguments)?;
        let (subcommand, inputs): (&[&str], usize) = match &options {
            Options::Interleave(options) => (&["interleave"], options.inputs.len()),
            Options::Montybinpack(montybinpack::MontyBinpackOptions::Interleave(options)) => {
                (&["montybinpack", "interleave"], options.inputs.len())
            }
            Options::Viribinpack(viribinpack::ViriBinpackOptions::Interleave(options)) => {
                (&["viribinpack", "interleave"], options.inputs.len())
            }
            _ => return Ok(options),
        };
        if inputs >= Self::MIN_INTERLEAVE_INPUTS {
            return Ok(options);
        }
        // Reported through the subcommand so the message carries its usage line and exit status.
        let mut command = Self::command();
        command.build();
        let mut command = &mut command;
        for name in subcommand {
            command = command.find_subcommand_mut(name).expect("subcommand is defined");
        }
        Err(command.error(
            ErrorKind::TooFewValues,
            format!("at least {} inputs are required; only {inputs} was provided", Self::MIN_INTERLEAVE_INPUTS),
        ))
    }
}

fn main() -> anyhow::Result<()> {
    match Options::parse_checked(std::env::args_os()).unwrap_or_else(|error| error.exit()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_schema_is_consistent() {
        Options::command().debug_assert();
    }

    #[test]
    fn shuffle_defaults_and_long_flags_are_compatible() {
        let Options::Shuffle(options) = Options::try_parse_from([
            "bullet-utils",
            "shuffle",
            "--input",
            "input",
            "--output",
            "output",
            "--mem-used-mb",
            "1",
        ])
        .unwrap() else {
            panic!("expected shuffle command");
        };
        assert_eq!(options.record_size, 32);
        assert_eq!(options.interleave_block_mb, 8);
        assert_eq!(options.interleave_mode, interleave::InterleaveMode::Block);
        assert_eq!(options.seed, None);
    }

    #[test]
    fn interleave_short_flags_and_case_insensitive_mode_are_compatible() {
        let Options::Interleave(options) = Options::try_parse_from([
            "bullet-utils",
            "interleave",
            "a",
            "b",
            "-o",
            "output",
            "--mode",
            "ReCoRd",
            "--seed",
            "123",
            "--record-size",
            "40",
        ])
        .unwrap() else {
            panic!("expected interleave command");
        };
        assert_eq!(options.inputs, ["a", "b"].map(std::path::PathBuf::from));
        assert_eq!(options.mode, interleave::InterleaveMode::Record);
        assert_eq!(options.seed, Some(123));
        assert_eq!(options.record_size, 40);
        assert_eq!(options.block_mb, 8);
    }

    #[test]
    fn nested_binpack_commands_and_positional_arity_are_compatible() {
        for format in ["montybinpack", "viribinpack"] {
            for arguments in [
                vec!["bullet-utils", format, "count", "input"],
                vec!["bullet-utils", format, "head", "input", "-o", "output", "-g", "2"],
                vec!["bullet-utils", format, "interleave", "a", "b", "-o", "output"],
            ] {
                assert!(Options::try_parse_from(arguments).is_ok());
            }
            assert!(Options::parse_checked(["bullet-utils", format, "interleave", "a", "-o", "output"]).is_err());
            assert!(Options::parse_checked(["bullet-utils", format, "interleave", "a", "-o", "output", "b"]).is_ok());
        }
        assert!(Options::try_parse_from(["bullet-utils", "viribinpack", "splat", "input", "output", "config"]).is_ok());
        let error = Options::parse_checked(["bullet-utils", "interleave", "a", "-o", "output"]).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::TooFewValues);
        assert_eq!(error.exit_code(), 2);
        let Ok(Options::Interleave(options)) =
            Options::parse_checked(["bullet-utils", "interleave", "a", "-o", "output", "b", "--seed", "1", "c"])
        else {
            panic!("expected interleave command");
        };
        assert_eq!(options.inputs, ["a", "b", "c"].map(std::path::PathBuf::from));
        assert!(Options::try_parse_from(["bullet-utils", "bucket-count", "a", "-b", "buckets"]).is_ok());
        assert!(
            Options::try_parse_from([
                "bullet-utils",
                "convert",
                "-f",
                "text",
                "-i",
                "input",
                "-o",
                "output",
                "-t",
                "2"
            ])
            .is_ok()
        );
    }
}
