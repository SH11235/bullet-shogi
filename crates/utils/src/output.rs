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

pub struct AtomicOutput {
    temporary: NamedTempFile,
    destination: PathBuf,
    permissions: Option<Permissions>,
}

impl AtomicOutput {
    pub fn new(destination: &Path) -> anyhow::Result<Self> {
        let permissions = match fs::metadata(destination) {
            Ok(metadata) => {
                ensure!(metadata.is_file(), "output must be a regular file: {}", destination.display());
                ensure!(!metadata.permissions().readonly(), "output is read-only: {}", destination.display());
                Some(metadata.permissions())
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let parent = destination.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let mut builder = Builder::new();
        builder.prefix(".bullet-output-");
        if let Some(permissions) = &permissions {
            builder.permissions(permissions.clone());
        } else {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                builder.permissions(Permissions::from_mode(0o666));
            }
        }
        let temporary = builder.tempfile_in(parent).context("failed to create temporary output")?;
        Ok(Self { temporary, destination: destination.to_owned(), permissions })
    }

    pub fn file(&mut self) -> &mut File {
        self.temporary.as_file_mut()
    }

    pub fn commit(self) -> anyhow::Result<()> {
        if let Some(permissions) = self.permissions {
            self.temporary.as_file().set_permissions(permissions)?;
        }
        self.temporary.as_file().sync_all()?;
        self.temporary.persist(&self.destination).map_err(|error| error.error).context("failed to publish output")?;
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
}
