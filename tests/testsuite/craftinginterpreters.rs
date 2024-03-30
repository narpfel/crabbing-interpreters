use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::rstest;

use crate::testname;

fn as_relative(path: &Path) -> &Path {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap()
}

#[rstest]
fn operator_tests(
    testname: String,
    #[files("craftinginterpreters/test/operator/**/*.lox")] path: PathBuf,
) {
    assert_cmd_snapshot!(
        testname,
        Command::new(get_cargo_bin("crabbing-interpreters")).arg(as_relative(&path))
    )
}
