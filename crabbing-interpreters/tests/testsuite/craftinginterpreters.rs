use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

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
    #[files("../craftinginterpreters/test/function/**/*.lox")]
    #[files("../craftinginterpreters/test/if/**/*.lox")]
    #[files("../craftinginterpreters/test/logical_operator/**/*.lox")]
    #[files("../craftinginterpreters/test/nil/**/*.lox")]
    #[files("../craftinginterpreters/test/number/**/*.lox")]
    #[files("../craftinginterpreters/test/operator/**/*.lox")]
    #[files("../craftinginterpreters/test/print/**/*.lox")]
    #[files("../craftinginterpreters/test/string/**/*.lox")]
    #[files("../craftinginterpreters/test/variable/**/*.lox")]
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
