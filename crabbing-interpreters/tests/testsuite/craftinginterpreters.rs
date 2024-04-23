use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::rstest;

use crate::filter_pointers;
use crate::relative_to;
use crate::PointerFilter;

#[rstest]
fn tests(
    _filter_pointers: PointerFilter,
    #[files("../craftinginterpreters/test/**/*.lox")]
    #[exclude("/scanning/")]
    #[exclude("/expressions/")]
    #[exclude("/benchmark/")]
    path: PathBuf,
) {
    let path = relative_to(
        &path,
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    );
    assert_cmd_snapshot!(
        path.display().to_string(),
        Command::new(get_cargo_bin("crabbing-interpreters"))
            .arg(path)
            .current_dir("..")
    )
}

#[rstest]
fn scope(
    #[files("../craftinginterpreters/test/**/*.lox")]
    #[files("tests/cases/**/*.lox")]
    #[exclude("/benchmark/")]
    #[exclude("loop_too_large\\.lox$")]
    path: PathBuf,
) {
    let path = relative_to(
        &path,
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    );
    let command = || {
        let mut command = Command::new(get_cargo_bin("crabbing-interpreters"));
        command
            .current_dir("..")
            .arg(path)
            .arg("--scopes")
            .arg("--stop-at=scopes");
        command
    };
    if command()
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success()
    {
        assert_cmd_snapshot!(format!("scope-{}", path.display()), command());
    }
}

#[rstest]
fn closure_compiler(
    _filter_pointers: PointerFilter,
    #[files("../craftinginterpreters/test/**/*.lox")]
    #[exclude("/scanning/")]
    #[exclude("/expressions/")]
    #[exclude("/benchmark/")]
    path: PathBuf,
) {
    let path = relative_to(
        &path,
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    );
    assert_cmd_snapshot!(
        path.display().to_string(),
        Command::new(get_cargo_bin("crabbing-interpreters"))
            .current_dir("..")
            .arg("--loop=closures")
            .arg(path),
    );
}
