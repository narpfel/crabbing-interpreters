#![feature(closure_lifetime_binder)]

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use crabbing_interpreters::Loop;
use insta_cmd::assert_cmd_snapshot;
use insta_cmd::get_cargo_bin;
use rstest::fixture;
use rstest::rstest;
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

fn loop_as_arg(l: Loop) -> &'static str {
    match l {
        Loop::Ast => "ast",
        Loop::Closures => "closures",
        Loop::Bytecode => "bytecode",
        Loop::Threaded => "threaded",
    }
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
            Interpreter::Native(r#loop) => {
                let mut command = Command::new(get_cargo_bin("crabbing-interpreters"));
                command.arg(format!("--loop={}", loop_as_arg(r#loop)));
                command
            }
            #[cfg(feature = "miri_tests")]
            Interpreter::Miri(r#loop) => {
                let mut command = Command::new("cargo");
                command
                    .env(
                        "MIRIFLAGS",
                        "-Zmiri-disable-isolation -Zmiri-deterministic-floats",
                    )
                    .args([
                        "miri",
                        "run",
                        "-q",
                        "--features",
                        &std::env::var("TEST_FEATURES").unwrap_or_else(|_| "".to_string()),
                        "--",
                        &format!("--loop={}", loop_as_arg(r#loop)),
                    ]);
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
    settings.add_filter(
        r"Could not read file `(?<filename>.*?)`: .*? \(os error 2\) \(in native function call to `read_file`\)",
        "Could not read file `$filename`: No such file or directory (os error 2) (in native function call to `read_file`)",
    );
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
#[case::use_variable_in_multiple_lines("var x = 42;\nprint x;")]
#[case::use_variable_in_multiple_lines_and_use_builtin(
    "var x = 42; var y = clock;\nprint x; print y; native_function_test(x, y);"
)]
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
                Command::from(interpreter).arg("--scopes").pass_stdin(src),
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

#[rstest]
#[case::panic_unwind(for <'a> |cmd: &'a mut Command| -> &'a mut Command { cmd })]
#[case::panic_abort(
    for <'a> |cmd: &'a mut Command| -> &'a mut Command {
        cmd
            .env("RUSTFLAGS", "-Z unstable-options -C panic=immediate-abort")
            .args([
                "-Zbuild-std=panic_abort,std",
                "--target=x86_64-unknown-linux-gnu",
            ])
    }
)]
#[ignore]
fn test_that_threaded_interpreter_is_properly_tailrecursive(
    #[case] apply_panic_strategy: impl for<'a> FnOnce(&'a mut Command) -> &'a mut Command,
) {
    let additional_features = std::env::var("TEST_FEATURES").unwrap_or_else(|_| "".to_string());
    assert_cmd_snapshot!(
        "that_threaded_interpreter_is_properly_tailrecursive",
        apply_panic_strategy(Command::new("cargo").arg("run"))
            .args([
                "--profile=perf",
                "--quiet",
                &format!("--features=count_bytecode_execution,{additional_features}"),
                "--",
                "--loop=threaded",
                "--show-bytecode-execution-counts",
                "tests/tailrec/test_tailrec.lox",
            ])
            .stdout(Stdio::null()),
    );
}

#[apply(interpreter)]
fn test_that_constants_are_deduplicated(interpreter: Interpreter) {
    assert_cmd_snapshot!(
        "test_that_constants_are_deduplicated",
        Command::from(interpreter).args([
            "--scopes",
            "--bytecode",
            "--stop-at=bytecode",
            "tests/cases/deduplicated_constants.lox",
        ]),
    )
}
