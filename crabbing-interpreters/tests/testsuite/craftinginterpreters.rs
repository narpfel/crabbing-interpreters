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
    #[files("../craftinginterpreters/test/*.lox")]
    #[files("../craftinginterpreters/test/assignment/**/*.lox")]
    #[files("../craftinginterpreters/test/block/**/*.lox")]
    #[files("../craftinginterpreters/test/bool/**/*.lox")]
    #[files("../craftinginterpreters/test/comments/**/*.lox")]
    #[files("../craftinginterpreters/test/for/**/*.lox")]
    #[files("../craftinginterpreters/test/function/**/*.lox")]
    #[files("../craftinginterpreters/test/if/**/*.lox")]
    #[files("../craftinginterpreters/test/logical_operator/**/*.lox")]
    #[files("../craftinginterpreters/test/nil/**/*.lox")]
    #[files("../craftinginterpreters/test/number/**/*.lox")]
    #[files("../craftinginterpreters/test/operator/**/*.lox")]
    #[files("../craftinginterpreters/test/print/**/*.lox")]
    #[files("../craftinginterpreters/test/string/**/*.lox")]
    #[files("../craftinginterpreters/test/variable/**/*.lox")]
    #[files("../craftinginterpreters/test/while/**/*.lox")]
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
