use std::path::PathBuf;
use std::process::Command;

use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::rstest;

use crate::as_relative;

#[rstest]
fn tests(
    #[files("craftinginterpreters/test/*.lox")]
    #[files("craftinginterpreters/test/assignment/**/*.lox")]
    #[files("craftinginterpreters/test/block/**/*.lox")]
    #[files("craftinginterpreters/test/bool/**/*.lox")]
    #[files("craftinginterpreters/test/comments/**/*.lox")]
    #[files("craftinginterpreters/test/if/**/*.lox")]
    #[files("craftinginterpreters/test/logical_operator/**/*.lox")]
    #[files("craftinginterpreters/test/nil/**/*.lox")]
    #[files("craftinginterpreters/test/number/**/*.lox")]
    #[files("craftinginterpreters/test/operator/**/*.lox")]
    #[files("craftinginterpreters/test/print/**/*.lox")]
    #[files("craftinginterpreters/test/string/**/*.lox")]
    path: PathBuf,
) {
    let path = as_relative(&path);
    assert_cmd_snapshot!(
        path.display().to_string(),
        Command::new(get_cargo_bin("crabbing-interpreters")).arg(path)
    )
}
