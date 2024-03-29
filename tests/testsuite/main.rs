use std::process::Command;

use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::fixture;
use rstest::rstest;

#[fixture]
fn testname() -> String {
    std::thread::current()
        .name()
        .unwrap()
        .to_string()
        .replace(":", "_")
}

#[test]
fn repl_evaluates_expression() {
    assert_cmd_snapshot!(Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin("1 + 2"))
}

#[test]
fn repl_produces_error_message() {
    assert_cmd_snapshot!(Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin("nil * 2"))
}

#[rstest]
#[case::nil("nil")]
#[case::bool_true("true")]
#[case::negative_nil("- nil")]
#[case::divide_by_string(r#"42 / "string""#)]
fn repl(testname: String, #[case] src: &str) {
    assert_cmd_snapshot!(
        testname,
        Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin(src)
    )
}
