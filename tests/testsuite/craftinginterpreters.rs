use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::rstest;

fn as_relative(path: &Path) -> &Path {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap()
}

#[rstest]
fn tests(
    #[files("craftinginterpreters/test/*.lox")]
    #[files("craftinginterpreters/test/assignment/**/*.lox")]
    #[files("craftinginterpreters/test/block/**/*.lox")]
    #[files("craftinginterpreters/test/bool/**/*.lox")]
    #[files("craftinginterpreters/test/comments/**/*.lox")]
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
