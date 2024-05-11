use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use crabbing_interpreters::Loop;
use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::fixture;
use rstest::rstest;
// FIXME: rstest PR 244
#[allow(clippy::single_component_path_imports)]
use rstest_reuse;
use rstest_reuse::apply;
use rstest_reuse::template;

#[template]
fn test_cases(
    #[files("../craftinginterpreters/test/**/*.lox")]
    #[files("tests/cases/**/*.lox")]
    #[exclude("/scanning/")]
    #[exclude("/expressions/")]
    #[exclude("/benchmark/")]
    #[exclude("/stack_overflow.lox$")]
    path: PathBuf,
) {
}

#[fixture]
fn testname() -> String {
    std::thread::current()
        .name()
        .unwrap()
        .to_string()
        .replace(':', "_")
}

#[derive(Debug, Clone, Copy)]
enum Interpreter {
    Native(Loop),
    #[cfg(feature = "miri_tests")]
    Miri(Loop),
}

impl From<Interpreter> for Command {
    fn from(interpreter: Interpreter) -> Self {
        match interpreter {
            Interpreter::Native(Loop::Ast) => Command::new(get_cargo_bin("crabbing-interpreters")),
            Interpreter::Native(Loop::Closures) => {
                let mut command = Command::new(get_cargo_bin("crabbing-interpreters"));
                command.arg("--loop=closures");
                command
            }
            Interpreter::Native(Loop::Bytecode) => {
                let mut command = Command::new(get_cargo_bin("crabbing-interpreters"));
                command.arg("--loop=bytecode");
                command
            }
            Interpreter::Native(Loop::Threaded) => {
                let mut command = Command::new(get_cargo_bin("crabbing-interpreters"));
                command.arg("--loop=threaded");
                command
            }
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(Loop::Ast) => {
                let mut command = Command::new("cargo");
                command
                    .args(&["miri", "run", "-q", "--"])
                    .env("MIRIFLAGS", "-Zmiri-disable-isolation");
                command
            }
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(Loop::Closures) => {
                let mut command = Command::new("cargo");
                command
                    .args(&["miri", "run", "-q", "--", "--loop=closures"])
                    .env("MIRIFLAGS", "-Zmiri-disable-isolation");
                command
            }
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(Loop::Bytecode) => {
                let mut command = Command::new("cargo");
                command
                    .args(&["miri", "run", "-q", "--", "--loop=bytecode"])
                    .env("MIRIFLAGS", "-Zmiri-disable-isolation");
                command
            }
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(Loop::Threaded) => {
                let mut command = Command::new("cargo");
                command
                    .args(&["miri", "run", "-q", "--", "--loop=threaded"])
                    .env("MIRIFLAGS", "-Zmiri-disable-isolation");
                command
            }
        }
    }
}

#[template]
#[rstest]
#[case::native_ast(Interpreter::Native(Loop::Ast))]
#[case::native_closures(Interpreter::Native(Loop::Closures))]
#[case::native_bytecode(Interpreter::Native(Loop::Bytecode))]
#[case::native_threaded(Interpreter::Native(Loop::Threaded))]
#[cfg_attr(feature = "miri_tests", case::miri_ast(Interpreter::Miri(Loop::Ast)))]
#[cfg_attr(
    feature = "miri_tests",
    case::miri_closures(Interpreter::Miri(Loop::Closures))
)]
#[cfg_attr(
    feature = "miri_tests",
    case::miri_bytecode(Interpreter::Miri(Loop::Bytecode))
)]
#[cfg_attr(
    feature = "miri_tests",
    case::miri_threaded(Interpreter::Miri(Loop::Threaded))
)]
fn interpreter(#[case] interpreter: Interpreter) {}

type OutputFilter = insta::internals::SettingsBindDropGuard;

#[fixture]
fn filter_output() -> OutputFilter {
    let mut settings = insta::Settings::clone_current();
    settings.add_filter("0x[[:xdigit:]]{4,16}", "[POINTER]");
    settings.bind_to_scope()
}

fn relative_to(path: &Path, target: impl AsRef<Path>) -> &Path {
    path.strip_prefix(target.as_ref()).unwrap()
}

#[apply(interpreter)]
fn repl_evaluates_expression(
    _filter_output: OutputFilter,
    testname: String,
    interpreter: Interpreter,
) {
    assert_cmd_snapshot!(testname, Command::from(interpreter).pass_stdin("1 + 2;"))
}

#[apply(interpreter)]
fn repl_produces_error_message(
    _filter_output: OutputFilter,
    testname: String,
    interpreter: Interpreter,
) {
    assert_cmd_snapshot!(testname, Command::from(interpreter).pass_stdin("nil * 2;"))
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
#[case::incomplete_statement("print")]
#[case::unterminated_string_literal(r#""unterminated string literal"#)]
#[case::unterminated_string_literal_doesnt_break_repl("\"unterminated string literal\n4 + 5;")]
#[case::type_error_for_plus_mentions_strings("nil + 42;")]
#[case::add_bools("true + false;")]
#[case::var_statement_without_initialiser("var x; print x; print x == nil;")]
fn repl(_filter_output: OutputFilter, #[by_ref] testname: &str, #[case] src: &str) {
    insta::allow_duplicates! {
        for interpreter in [
            Interpreter::Native(Loop::Ast),
            Interpreter::Native(Loop::Closures),
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(Loop::Ast),
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(Loop::Closures),
        ] {
            assert_cmd_snapshot!(
                testname,
                Command::from(interpreter).pass_stdin(src),
            )
        }
    }
}

#[apply(test_cases)]
#[apply(interpreter)]
fn tests(_filter_output: OutputFilter, path: PathBuf, interpreter: Interpreter) {
    let path = relative_to(
        &path,
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    );

    assert_cmd_snapshot!(
        path.display().to_string(),
        Command::from(interpreter).current_dir("..").arg(path),
    )
}

#[apply(interpreter)]
fn nonexistent_file(_filter_output: OutputFilter, testname: String, interpreter: Interpreter) {
    assert_cmd_snapshot!(testname, Command::from(interpreter).arg("nonexistent_file"));
}

#[apply(test_cases)]
#[rstest]
fn scope(#[exclude("loop_too_large\\.lox$")] path: PathBuf) {
    let path = relative_to(
        &path,
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    );
    let command = || {
        let mut command = Command::from(Interpreter::Native(Loop::Ast));
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

#[test]
#[ignore]
fn test_that_threaded_interpreter_is_properly_tailrecursive() {
    assert_cmd_snapshot!(Command::new("cargo")
        .args([
            "run",
            "--profile=perf",
            "--quiet",
            "--features=count_bytecode_execution",
            "--",
            "--loop=threaded",
            "--show-bytecode-execution-counts",
            "tests/tailrec/test_tailrec.lox",
        ])
        .stdout(Stdio::null()));
}
