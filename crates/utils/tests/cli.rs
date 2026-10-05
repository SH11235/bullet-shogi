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
fn input_aliases_are_rejected_without_modifying_records() {
    let directory = tempfile::tempdir().unwrap();
    let input = records(32, 10);
    fs::write(directory.path().join("input.bin"), &input).unwrap();
    fs::hard_link(directory.path().join("input.bin"), directory.path().join("alias.bin")).unwrap();
    for output in ["input.bin", "alias.bin"] {
        for arguments in [
            vec!["shuffle", "-i", "input.bin", "-o", output, "-m", "1"],
            vec!["interleave", "input.bin", "input.bin", "-o", output, "--mode", "record"],
            vec!["interleave", "input.bin", "input.bin", "-o", output, "--mode", "block"],
            vec!["interleave", "input.bin", "input.bin", "-o", output, "--mode", "concat"],
        ] {
            assert!(!command(directory.path()).args(arguments).output().unwrap().status.success());
            assert_eq!(fs::read(directory.path().join("input.bin")).unwrap(), input);
        }
    }
}

#[cfg(unix)]
#[test]
fn symbolic_links_cannot_alias_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let input = records(40, 10);
    fs::write(directory.path().join("input.bin"), &input).unwrap();
    std::os::unix::fs::symlink("input.bin", directory.path().join("alias.bin")).unwrap();
    let output = command(directory.path())
        .args(["shuffle", "-i", "input.bin", "-o", "alias.bin", "-m", "1", "--record-size", "40"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(fs::read(directory.path().join("input.bin")).unwrap(), input);
    assert!(directory.path().join("alias.bin").is_symlink());
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
fn readonly_and_device_outputs_are_rejected_without_replacement() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("input.bin"), records(32, 1)).unwrap();
    let output_path = directory.path().join("output.bin");
    fs::write(&output_path, b"existing").unwrap();
    fs::set_permissions(&output_path, fs::Permissions::from_mode(0o444)).unwrap();
    for destination in ["output.bin", "/dev/null"] {
        let output = command(directory.path())
            .args(["shuffle", "-i", "input.bin", "-o", destination, "-m", "1"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert_eq!(fs::read(&output_path).unwrap(), b"existing");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }
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
