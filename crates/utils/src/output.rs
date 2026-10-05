use std::{
    fs::{self, File, Permissions},
    io,
    path::{Path, PathBuf},
};

use anyhow::{Context, ensure};
use tempfile::{Builder, NamedTempFile};

pub fn ensure_distinct_output(inputs: &[PathBuf], output: &Path) -> anyhow::Result<()> {
    match fs::metadata(output) {
        Ok(_) => {
            for input in inputs {
                let aliases = same_file::is_same_file(input, output).with_context(|| {
                    format!("failed to compare input {} with output {}", input.display(), output.display())
                })?;
                ensure!(!aliases, "output must differ from input: {}", input.display());
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

/// Hidden name prefix that ties a temporary entry to the destination it was created for,
/// so anything an interrupted run leaves behind can be attributed and deleted by hand.
pub fn temporary_prefix(kind: &str, destination: &Path) -> anyhow::Result<String> {
    // Long destination names are shortened so the prefix plus the random suffix stays a valid file name.
    const MAX_NAME_CHARS: usize = 64;
    let name = destination
        .file_name()
        .with_context(|| format!("output has no file name: {}", destination.display()))?
        .to_string_lossy();
    let name: String = name.chars().take(MAX_NAME_CHARS).collect();
    Ok(format!(".bullet-{kind}-{name}."))
}

/// Follows a symbolic link at the destination so the file it names is the one replaced.
/// A dangling link resolves to the missing target, which is then created.
fn resolve_destination(destination: &Path) -> anyhow::Result<PathBuf> {
    const MAX_LINK_DEPTH: usize = 40;
    let mut resolved = destination.to_owned();
    for _ in 0..MAX_LINK_DEPTH {
        match fs::symlink_metadata(&resolved) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let target = fs::read_link(&resolved)?;
                resolved = resolved.parent().unwrap_or(Path::new("")).join(target);
            }
            _ => return Ok(resolved),
        }
    }
    anyhow::bail!("too many levels of symbolic links: {}", destination.display())
}

fn parent_directory(path: &Path) -> &Path {
    path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new("."))
}

enum Target {
    Replace { temporary: NamedTempFile, destination: PathBuf, permissions: Option<Permissions> },
    // Devices, FIFOs and similar destinations cannot be replaced by a rename, so they are written in place.
    Direct { file: File, path: PathBuf },
}

pub struct AtomicOutput {
    target: Target,
}

impl AtomicOutput {
    pub fn new(destination: &Path) -> anyhow::Result<Self> {
        // Checked through the path as given: resolving links by hand does not work for
        // the magic links behind names such as `/dev/stdout`.
        if let Ok(metadata) = fs::metadata(destination)
            && !metadata.is_file()
        {
            ensure!(!metadata.is_dir(), "output is a directory: {}", destination.display());
            let file = fs::OpenOptions::new()
                .write(true)
                .open(destination)
                .with_context(|| format!("failed to open output {}", destination.display()))?;
            return Ok(Self { target: Target::Direct { file, path: destination.to_owned() } });
        }
        let destination = resolve_destination(destination)?;
        let permissions = match fs::metadata(&destination) {
            Ok(metadata) => {
                ensure!(metadata.is_file(), "output must be a regular file: {}", destination.display());
                ensure!(!metadata.permissions().readonly(), "output is read-only: {}", destination.display());
                Some(metadata.permissions())
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let mut builder = Builder::new();
        let prefix = temporary_prefix("output", &destination)?;
        builder.prefix(&prefix);
        if let Some(permissions) = &permissions {
            builder.permissions(permissions.clone());
        } else {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                builder.permissions(Permissions::from_mode(0o666));
            }
        }
        let temporary =
            builder.tempfile_in(parent_directory(&destination)).context("failed to create temporary output")?;
        Ok(Self { target: Target::Replace { temporary, destination, permissions } })
    }

    pub fn file(&mut self) -> &mut File {
        match &mut self.target {
            Target::Replace { temporary, .. } => temporary.as_file_mut(),
            Target::Direct { file, .. } => file,
        }
    }

    /// Path that currently receives the bytes, for writers that insist on opening a path themselves.
    pub fn path(&self) -> &Path {
        match &self.target {
            Target::Replace { temporary, .. } => temporary.path(),
            Target::Direct { path, .. } => path,
        }
    }

    pub fn commit(self) -> anyhow::Result<()> {
        let Target::Replace { temporary, destination, permissions } = self.target else {
            return Ok(());
        };
        if let Some(permissions) = permissions {
            temporary.as_file().set_permissions(permissions)?;
        }
        temporary.as_file().sync_all()?;
        temporary.persist(&destination).map_err(|error| error.error).context("failed to publish output")?;
        // Without syncing the directory the rename itself can be lost on power failure even
        // though the data was synced. Some filesystems reject directory sync, and the output
        // is already in place by then, so a failure is reported without failing the command.
        #[cfg(unix)]
        if let Err(error) = File::open(parent_directory(&destination)).and_then(|directory| directory.sync_all()) {
            eprintln!("warning: could not sync the directory of {}: {error}", destination.display());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn failure_preserves_output_and_removes_temporary_file() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("data.bin");
        fs::write(&destination, b"existing").unwrap();
        {
            let mut output = AtomicOutput::new(&destination).unwrap();
            output.file().write_all(b"partial").unwrap();
        }
        assert_eq!(fs::read(&destination).unwrap(), b"existing");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn publication_replaces_output_without_changing_hard_links() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("data.bin");
        let link = directory.path().join("link.bin");
        fs::write(&destination, b"existing").unwrap();
        fs::hard_link(&destination, &link).unwrap();
        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file().write_all(b"complete").unwrap();
        output.commit().unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"complete");
        assert_eq!(fs::read(&link).unwrap(), b"existing");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }

    #[test]
    fn output_cannot_alias_an_input() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.bin");
        let link = directory.path().join("link.bin");
        fs::write(&input, b"existing").unwrap();
        fs::hard_link(&input, &link).unwrap();
        assert!(ensure_distinct_output(std::slice::from_ref(&input), &input).is_err());
        assert!(ensure_distinct_output(std::slice::from_ref(&input), &link).is_err());
        assert_eq!(fs::read(&input).unwrap(), b"existing");
    }

    #[cfg(unix)]
    #[test]
    fn replacement_preserves_existing_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("data.bin");
        fs::write(&destination, b"existing").unwrap();
        fs::set_permissions(&destination, Permissions::from_mode(0o640)).unwrap();
        let output = AtomicOutput::new(&destination).unwrap();
        output.commit().unwrap();
        assert_eq!(fs::metadata(&destination).unwrap().permissions().mode() & 0o777, 0o640);
    }

    #[cfg(unix)]
    #[test]
    fn temporary_output_never_widens_existing_read_permissions() {
        use std::os::unix::fs::PermissionsExt;

        const CHILD: &str = "BULLET_UTILS_TEMP_PERMISSION_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new("sh")
                .args(["-c", "umask 022; exec \"$@\"", "bullet-permission-test"])
                .arg(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "output::tests::temporary_output_never_widens_existing_read_permissions",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
            return;
        }

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("data.bin");
        fs::write(&destination, b"existing").unwrap();
        fs::set_permissions(&destination, Permissions::from_mode(0o600)).unwrap();
        let mut output = AtomicOutput::new(&destination).unwrap();
        output.file().write_all(b"private").unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"existing");
        assert_eq!(output.file().metadata().unwrap().permissions().mode() & 0o077, 0);
        output.commit().unwrap();
        assert_eq!(fs::metadata(destination).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn temporary_names_identify_their_destination() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("data.bin");
        let output = AtomicOutput::new(&destination).unwrap();
        let name = output.path().file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(".bullet-output-data.bin."), "{name}");
        assert_eq!(output.path().parent(), destination.parent());

        let long = directory.path().join("x".repeat(250));
        let mut output = AtomicOutput::new(&long).unwrap();
        output.file().write_all(b"complete").unwrap();
        output.commit().unwrap();
        assert_eq!(fs::read(long).unwrap(), b"complete");
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_destinations_are_written_through() {
        let directory = tempfile::tempdir().unwrap();
        let real = directory.path().join("real");
        fs::create_dir(&real).unwrap();
        let link = directory.path().join("link.bin");
        std::os::unix::fs::symlink("real/data.bin", &link).unwrap();
        let chained = directory.path().join("chained.bin");
        std::os::unix::fs::symlink("link.bin", &chained).unwrap();

        for (destination, contents) in [(&link, b"dangling"), (&link, b"existing"), (&chained, b"chained!")] {
            let mut output = AtomicOutput::new(destination).unwrap();
            assert_eq!(output.path().parent(), Some(real.as_path()));
            output.file().write_all(contents).unwrap();
            output.commit().unwrap();
            assert_eq!(fs::read(real.join("data.bin")).unwrap(), contents);
            assert!(link.is_symlink() && chained.is_symlink());
            assert_eq!(fs::read_dir(&real).unwrap().count(), 1);
            assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 3);
        }

        let cycle = directory.path().join("cycle.bin");
        std::os::unix::fs::symlink("cycle.bin", &cycle).unwrap();
        assert!(AtomicOutput::new(&cycle).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn non_regular_destinations_are_written_in_place() {
        let mut output = AtomicOutput::new(Path::new("/dev/null")).unwrap();
        assert_eq!(output.path(), Path::new("/dev/null"));
        output.file().write_all(b"discarded").unwrap();
        output.commit().unwrap();
        assert!(!fs::metadata("/dev/null").unwrap().is_file());

        let directory = tempfile::tempdir().unwrap();
        assert!(AtomicOutput::new(directory.path()).is_err());
    }
}
