use std::process::Command;

use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;

#[test]
fn repl_evaluates_expression() {
    assert_cmd_snapshot!(Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin("1 + 2"))
}

#[test]
fn repl_produces_error_message() {
    assert_cmd_snapshot!(Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin("nil * 2"))
}
