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
    assert_cmd_snapshot!(Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin("1 + 2;"))
}

#[test]
fn repl_produces_error_message() {
    assert_cmd_snapshot!(Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin("nil * 2;"))
}

#[rstest]
#[case::nil("nil;")]
#[case::bool_true("true;")]
#[case::negative_nil("- nil;")]
#[case::divide_by_string(r#"42 / "string";"#)]
#[case::comment("// comment\n4 + 4;")]
#[case::multiline_input("4\n**2;")]
#[case::multiline_input("4<\n2;")]
#[case::multiline_input("4\n>\n2;")]
#[case::print_statement("print nil;")]
#[case::print_statement("print 42 > 27;")]
#[case::print_multiline("print  \n  \"string\";")]
#[case::multiple_statements("print 42; print 27;")]
#[case::print_print("print print 42;")]
#[case::variable_declaration("var x = 42;")]
#[case::use_variable("var x = 27; print x;")]
#[case::multiple_variables("var x = 27; var y = x + 42; print y;")]
#[case::multiple_variables("var x = 42; var y = 27; print x; print y; print x + y;")]
#[case::undefined_variable("x;")]
#[case::undefined_variable("var x = 42; print x + y;")]
#[case::long_undefined_variable_name("42 + long_variable_name;")]
fn repl(testname: String, #[case] src: &str) {
    assert_cmd_snapshot!(
        testname,
        Command::new(get_cargo_bin("crabbing-interpreters")).pass_stdin(src)
    )
}
