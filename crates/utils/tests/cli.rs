use std::{
    fs,
    path::Path,
    process::{Command, Output, Stdio},
};

fn command(directory: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_bullet-utils"));
    command.current_dir(directory);
    command
}

fn assert_success(output: Output) {
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

fn records(size: usize, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|id| {
            let mut record = vec![id as u8; size];
            record[..8].copy_from_slice(&(id as u64).to_le_bytes());
            record
        })
        .collect()
}

#[test]
fn seeded_shuffle_preserves_historical_order_for_chess_and_shogi_records() {
    for size in [32, 40] {
        let directory = tempfile::tempdir().unwrap();
        let input = records(size, 10);
        fs::write(directory.path().join("input.bin"), &input).unwrap();
        assert_success(
            command(directory.path())
                .args(["shuffle", "-i", "input.bin", "-o", "output.bin", "-m", "1", "--seed", "123", "--record-size"])
                .arg(size.to_string())
                .output()
                .unwrap(),
        );
        let output = fs::read(directory.path().join("output.bin")).unwrap();
        let expected: Vec<u8> = [0, 5, 7, 1, 3, 8, 2, 4, 6, 9]
            .into_iter()
            .flat_map(|id| input[id * size..(id + 1) * size].iter().copied())
            .collect();
        assert_eq!(output, expected);
        assert_eq!(fs::read(directory.path().join("input.bin")).unwrap(), input);
    }
}

#[test]
fn concurrent_disk_shuffles_preserve_records_and_unrelated_temporary_files() {
    for size in [32, 40] {
        let directory = tempfile::tempdir().unwrap();
        let input = records(size, 70_000);
        fs::write(directory.path().join("input.bin"), &input).unwrap();
        let existing_temporary = directory.path().join("tmp");
        fs::create_dir(&existing_temporary).unwrap();
        fs::write(existing_temporary.join("part_1.bin"), b"unrelated").unwrap();
        let mut children = Vec::new();
        for output in ["a.bin", "b.bin"] {
            children.push(
                command(directory.path())
                    .args(["shuffle", "-i", "input.bin", "-o", output, "-m", "1", "--seed", "123", "--record-size"])
                    .arg(size.to_string())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap(),
            );
        }
        for child in children {
            assert_success(child.wait_with_output().unwrap());
        }
        let output = fs::read(directory.path().join("a.bin")).unwrap();
        assert_eq!(output, fs::read(directory.path().join("b.bin")).unwrap());
        let mut actual: Vec<_> = output.chunks_exact(size).collect();
        let mut expected: Vec<_> = input.chunks_exact(size).collect();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
        assert_eq!(fs::read(directory.path().join("input.bin")).unwrap(), input);
        assert_eq!(fs::read(existing_temporary.join("part_1.bin")).unwrap(), b"unrelated");
        assert!(
            fs::read_dir(directory.path())
                .unwrap()
                .all(|entry| { !entry.unwrap().file_name().to_string_lossy().starts_with(".bullet-") })
        );
    }
}

#[test]
fn invalid_record_settings_and_incomplete_inputs_leave_existing_output_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("input.bin"), [1; 33]).unwrap();
    fs::write(directory.path().join("valid.bin"), [2; 32]).unwrap();
    fs::write(directory.path().join("output.bin"), b"existing").unwrap();
    for arguments in [
        vec!["shuffle", "-i", "input.bin", "-o", "output.bin", "-m", "1"],
        vec!["shuffle", "-i", "valid.bin", "-o", "output.bin", "-m", "0"],
        vec!["shuffle", "-i", "valid.bin", "-o", "output.bin", "-m", "1", "--record-size", "0"],
        vec!["interleave", "input.bin", "valid.bin", "-o", "output.bin", "--mode", "record"],
        vec!["interleave", "input.bin", "valid.bin", "-o", "output.bin", "--mode", "block"],
        vec!["interleave", "valid.bin", "missing.bin", "-o", "output.bin", "--mode", "concat"],
        vec!["interleave", "valid.bin", "valid.bin", "-o", "output.bin", "--record-size", "0"],
    ] {
        let output = command(directory.path()).args(arguments).output().unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
        assert_eq!(fs::read(directory.path().join("output.bin")).unwrap(), b"existing");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 3);
    }
}

#[test]
fn interleave_rejects_outputs_that_alias_an_input() {
    let directory = tempfile::tempdir().unwrap();
    let input = records(32, 10);
    fs::write(directory.path().join("input.bin"), &input).unwrap();
    fs::hard_link(directory.path().join("input.bin"), directory.path().join("alias.bin")).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("input.bin", directory.path().join("link.bin")).unwrap();
    for output in ["input.bin", "alias.bin", "link.bin"] {
        if !directory.path().join(output).exists() {
            continue;
        }
        for mode in ["record", "block", "concat"] {
            let result = command(directory.path())
                .args(["interleave", "input.bin", "input.bin", "-o", output, "--mode", mode])
                .output()
                .unwrap();
            assert_eq!(result.status.code(), Some(1));
            assert_eq!(fs::read(directory.path().join("input.bin")).unwrap(), input);
        }
    }
}

#[test]
fn shuffle_in_place_matches_shuffle_to_a_separate_output() {
    // 70,000 records exceed a 1 MiB budget, so the larger case takes the disk-backed path.
    for (size, count) in [(32, 10), (40, 10), (32, 70_000), (40, 70_000)] {
        let directory = tempfile::tempdir().unwrap();
        let input = records(size, count);
        fs::write(directory.path().join("input.bin"), &input).unwrap();
        fs::write(directory.path().join("in-place.bin"), &input).unwrap();
        for (source, destination) in [("input.bin", "separate.bin"), ("in-place.bin", "in-place.bin")] {
            assert_success(
                command(directory.path())
                    .args(["shuffle", "-i", source, "-o", destination, "-m", "1", "--seed", "123", "--record-size"])
                    .arg(size.to_string())
                    .output()
                    .unwrap(),
            );
        }
        let separate = fs::read(directory.path().join("separate.bin")).unwrap();
        assert_ne!(separate, input);
        assert_eq!(fs::read(directory.path().join("in-place.bin")).unwrap(), separate);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 3);
    }
}

#[cfg(unix)]
#[test]
fn symbolic_link_outputs_are_written_through_and_kept() {
    let directory = tempfile::tempdir().unwrap();
    let input = records(40, 70_000);
    fs::write(directory.path().join("input.bin"), &input).unwrap();
    fs::create_dir(directory.path().join("real")).unwrap();
    std::os::unix::fs::symlink("real/data.bin", directory.path().join("link.bin")).unwrap();
    std::os::unix::fs::symlink("input.bin", directory.path().join("self.bin")).unwrap();

    // The first pass creates the missing link target; the second replaces it.
    for budget in ["100", "1"] {
        assert_success(
            command(directory.path())
                .args(["shuffle", "-i", "input.bin", "-o", "link.bin", "--record-size", "40", "--seed", "123", "-m"])
                .arg(budget)
                .output()
                .unwrap(),
        );
        assert!(directory.path().join("link.bin").is_symlink());
        assert_eq!(fs::metadata(directory.path().join("real/data.bin")).unwrap().len(), input.len() as u64);
        assert_eq!(fs::read_dir(directory.path().join("real")).unwrap().count(), 1);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 4);
    }

    assert_success(
        command(directory.path())
            .args(["shuffle", "-i", "input.bin", "-o", "self.bin", "-m", "100", "--record-size", "40", "--seed", "123"])
            .output()
            .unwrap(),
    );
    assert!(directory.path().join("self.bin").is_symlink());
    let shuffled = fs::read(directory.path().join("input.bin")).unwrap();
    assert_ne!(shuffled, input);
    assert_eq!(shuffled.len(), input.len());
}

#[test]
fn interleave_accepts_inputs_split_around_options_and_requires_two() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("a.bin"), records(32, 3)).unwrap();
    fs::write(directory.path().join("b.bin"), records(32, 5)).unwrap();
    for arguments in [
        vec!["interleave", "a.bin", "b.bin", "-o", "grouped.bin", "--seed", "5", "--mode", "record"],
        vec!["interleave", "a.bin", "-o", "split.bin", "--seed", "5", "b.bin", "--mode", "record"],
        vec!["interleave", "-o", "leading.bin", "a.bin", "--seed", "5", "--mode", "record", "b.bin"],
    ] {
        assert_success(command(directory.path()).args(arguments).output().unwrap());
    }
    let grouped = fs::read(directory.path().join("grouped.bin")).unwrap();
    assert_eq!(grouped.len(), 8 * 32);
    assert_eq!(fs::read(directory.path().join("split.bin")).unwrap(), grouped);
    assert_eq!(fs::read(directory.path().join("leading.bin")).unwrap(), grouped);

    let single = command(directory.path()).args(["interleave", "a.bin", "-o", "single.bin"]).output().unwrap();
    assert_eq!(single.status.code(), Some(2));
    let message = String::from_utf8_lossy(&single.stderr);
    assert!(message.contains("at least 2 inputs are required"), "{message}");
    assert!(message.contains("bullet-utils interleave"), "{message}");
    assert!(!directory.path().join("single.bin").exists());
}

#[test]
fn concat_copies_bytes_without_consulting_record_size() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("a.bin"), b"abc").unwrap();
    fs::write(directory.path().join("b.bin"), b"defgh").unwrap();
    assert_success(
        command(directory.path())
            .args(["interleave", "a.bin", "b.bin", "-o", "output.bin", "--mode", "concat", "--record-size", "0"])
            .output()
            .unwrap(),
    );
    assert_eq!(fs::read(directory.path().join("output.bin")).unwrap(), b"abcdefgh");
}

#[test]
fn convert_publishes_only_complete_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let line = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 12 | 0.5\n";
    fs::write(directory.path().join("input.txt"), line.repeat(3)).unwrap();
    fs::write(directory.path().join("output.bin"), b"existing").unwrap();

    let unknown = command(directory.path())
        .args(["convert", "-f", "unknown", "-i", "input.txt", "-o", "output.bin"])
        .output()
        .unwrap();
    assert_eq!(unknown.status.code(), Some(1));
    assert_eq!(fs::read(directory.path().join("output.bin")).unwrap(), b"existing");
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);

    assert_success(
        command(directory.path())
            .args(["convert", "-f", "text", "-i", "input.txt", "-o", "output.bin"])
            .output()
            .unwrap(),
    );
    assert_eq!(fs::metadata(directory.path().join("output.bin")).unwrap().len(), 3 * 32);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
}

#[test]
fn disk_shuffle_failure_cleans_only_its_own_files() {
    let directory = tempfile::tempdir().unwrap();
    let input = records(32, 70_000);
    fs::write(directory.path().join("input.bin"), &input).unwrap();
    fs::write(directory.path().join("output.bin"), b"existing").unwrap();
    fs::create_dir(directory.path().join("tmp")).unwrap();
    fs::write(directory.path().join("tmp/part_1.bin"), b"unrelated").unwrap();
    let output = command(directory.path())
        .args(["shuffle", "-i", "input.bin", "-o", "output.bin", "-m", "1", "--interleave-block-mb", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(fs::read(directory.path().join("output.bin")).unwrap(), b"existing");
    assert_eq!(fs::read(directory.path().join("input.bin")).unwrap(), input);
    assert_eq!(fs::read(directory.path().join("tmp/part_1.bin")).unwrap(), b"unrelated");
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 3);
}

#[cfg(unix)]
#[test]
fn restrictive_umask_applies_to_new_outputs_and_preserves_existing_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("input.bin"), records(32, 10)).unwrap();
    let existing = directory.path().join("existing.bin");
    fs::write(&existing, b"existing").unwrap();
    fs::set_permissions(&existing, fs::Permissions::from_mode(0o644)).unwrap();
    for destination in ["existing.bin", "new.bin"] {
        let output = Command::new("sh")
            .current_dir(directory.path())
            .args(["-c", "umask 077; exec \"$@\"", "bullet-umask-test"])
            .arg(env!("CARGO_BIN_EXE_bullet-utils"))
            .args(["shuffle", "-i", "input.bin", "-o", destination, "-m", "1", "--seed", "123"])
            .output()
            .unwrap();
        assert_success(output);
    }
    assert_eq!(fs::metadata(existing).unwrap().permissions().mode() & 0o777, 0o644);
    assert_eq!(fs::metadata(directory.path().join("new.bin")).unwrap().permissions().mode() & 0o777, 0o600);
}

#[test]
fn all_subcommands_keep_help_and_version_flags() {
    let directory = tempfile::tempdir().unwrap();
    for subcommand in [
        vec![],
        vec!["convert"],
        vec!["interleave"],
        vec!["shuffle"],
        vec!["validate"],
        vec!["bucket-count"],
        vec!["montybinpack"],
        vec!["montybinpack", "head"],
        vec!["montybinpack", "interleave"],
        vec!["montybinpack", "count"],
        vec!["viribinpack"],
        vec!["viribinpack", "head"],
        vec!["viribinpack", "interleave"],
        vec!["viribinpack", "count"],
        vec!["viribinpack", "splat"],
    ] {
        for flag in ["-h", "--help", "-V", "--version"] {
            assert_success(command(directory.path()).args(&subcommand).arg(flag).output().unwrap());
        }
    }
}

#[test]
fn usage_errors_return_two_and_missing_input_errors_identify_the_path() {
    let directory = tempfile::tempdir().unwrap();
    let usage = command(directory.path()).arg("shuffle").output().unwrap();
    assert_eq!(usage.status.code(), Some(2));

    fs::write(directory.path().join("output.bin"), b"existing").unwrap();
    fs::write(directory.path().join("valid.bin"), records(32, 1)).unwrap();
    let output = command(directory.path())
        .args(["interleave", "missing.bin", "valid.bin", "-o", "output.bin"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing.bin"));
    assert_eq!(fs::read(directory.path().join("output.bin")).unwrap(), b"existing");
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
}

#[cfg(unix)]
#[test]
fn readonly_outputs_are_rejected_and_device_outputs_are_written_in_place() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("input.bin"), records(32, 1)).unwrap();
    let output_path = directory.path().join("output.bin");
    fs::write(&output_path, b"existing").unwrap();
    fs::set_permissions(&output_path, fs::Permissions::from_mode(0o444)).unwrap();
    for (destination, accepted) in [("output.bin", false), ("/dev/null", true)] {
        let output = command(directory.path())
            .args(["shuffle", "-i", "input.bin", "-o", destination, "-m", "1"])
            .output()
            .unwrap();
        assert_eq!(output.status.success(), accepted, "{}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(fs::read(&output_path).unwrap(), b"existing");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }
    assert!(!fs::metadata("/dev/null").unwrap().is_file());
    fs::set_permissions(output_path, fs::Permissions::from_mode(0o644)).unwrap();
}

#[test]
fn shuffle_memory_budget_must_fit_a_complete_record() {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("input.bin"), []).unwrap();
    fs::write(directory.path().join("output.bin"), b"existing").unwrap();
    let output = command(directory.path())
        .args(["shuffle", "-i", "input.bin", "-o", "output.bin", "-m", "1", "--record-size", "1048577"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("memory limit must include at least one record"));
    assert_eq!(fs::read(directory.path().join("output.bin")).unwrap(), b"existing");
}
