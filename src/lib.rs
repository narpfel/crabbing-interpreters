#![feature(closure_lifetime_binder)]
#![feature(error_generic_member_access)]
#![feature(lint_reasons)]
#![feature(never_type)]

use std::ffi::OsStr;
use std::ffi::OsString;
use std::io::stdin;
use std::io::stdout;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use ariadne::Fmt;
use ariadne::Label;
use bumpalo::Bump;
use clap::Parser;

use crate::eval::eval;
use crate::eval::execute;
pub use crate::lex::lex;
use crate::parse::expression;
use crate::parse::parse;
use crate::parse::program;
use crate::parse::Expression;

mod eval;
mod lex;
mod parse;

pub trait AllocPath {
    fn alloc_path(&self, path: impl AsRef<Path>) -> &Path;
}

impl AllocPath for Bump {
    fn alloc_path(&self, path: impl AsRef<Path>) -> &Path {
        Path::new(unsafe {
            OsStr::from_encoded_bytes_unchecked(
                self.alloc_slice_copy(path.as_ref().as_os_str().as_encoded_bytes()),
            )
        })
    }
}

/// Crabbing Interpreters
#[derive(Debug, Parser)]
struct Args {
    /// filename
    filename: Option<PathBuf>,
}

fn repl() -> Result<(), Box<dyn std::error::Error>> {
    let bump = &mut Bump::new();
    let mut line = String::new();
    loop {
        bump.reset();
        line.clear();
        print!("\x1B[0mλ» \x1B[1m");
        stdout().flush()?;
        // TODO: read until the lexer finds a semicolon token (or the parser doesn’t
        // error with an unexpected EOF)
        let len = stdin().read_line(&mut line)?;
        print!("\x1B[0m");
        if len == 0 {
            return Ok(());
        }
        let tokens = match lex(bump, "<input>", &line) {
            Ok(tokens) => tokens,
            Err(err) => {
                eprintln!("{:?}", err);
                continue;
            }
        };
        let expr = match parse(expression, bump, tokens) {
            Ok(expr) => expr,
            Err(err) => {
                println!("Parse error\n{:?}", err);
                continue;
            }
        };
        println!("\x1B[3m{:#?}\x1B[0m", expr);
        let result = eval(&expr);
        match result {
            Ok(value) => println!("\x1B[38;2;170;034;255m\x1B[1m=> {}\x1B[0m", value),
            Err(eval::TypeError::InvalidUnaryOp { op, value, at }) => {
                let Expression::Unary(operator, operand) = at
                else {
                    unreachable!()
                };

                let operand_color = ariadne::Color::Blue;
                let operator_color = ariadne::Color::Green;
                let expected_type_color = ariadne::Color::Magenta;

                let message = format!(
                     "type error in unary operator `{op}`: operand has type `{actual}` but `{op}` requires type `{expected}`",
                     op = operator.token.slice().fg(operator_color),
                     actual = value.typ().fg(operand_color),
                     expected = "Number".fg(expected_type_color),
                 );

                let labels = [
                    Label::new(operand.loc())
                        .with_message(format!(
                            "this is of type `{}`",
                            value.typ().fg(operand_color),
                        ))
                        .with_color(operand_color)
                        .with_order(1),
                    Label::new(operator.loc())
                        .with_message("the operator in question")
                        .with_color(operator_color)
                        .with_order(2),
                ];

                at.loc()
                    .report(ariadne::ReportKind::Error)
                    .with_message(message)
                    .with_labels(labels)
                    .with_help(format!(
                        "operator `{}` can only be applied to numbers",
                        op.token.slice().fg(operator_color),
                    ))
                    .finish()
                    .print(at.loc().cache())?;
            }
            Err(eval::TypeError::InvalidBinaryOp { lhs, op, rhs, at, .. }) => {
                let Expression::Binary { lhs: lhs_expr, rhs: rhs_expr, .. } = at
                else {
                    unreachable!()
                };

                let lhs_color = ariadne::Color::Blue;
                let op_color = ariadne::Color::Magenta;
                let rhs_color = ariadne::Color::Green;

                let message = format!(
                    "type error in binary operator `{}`: lhs has type `{}`, but rhs has type `{}`",
                    op.token.slice().fg(op_color),
                    lhs.typ().fg(lhs_color),
                    rhs.typ().fg(rhs_color),
                );

                let labels = [
                    Label::new(rhs_expr.loc())
                        .with_message(format!("this is of type `{}`", rhs.typ().fg(rhs_color)))
                        .with_color(rhs_color)
                        .with_order(1),
                    Label::new(op.loc())
                        .with_message("the operator in question")
                        .with_color(op_color)
                        .with_order(2),
                    Label::new(lhs_expr.loc())
                        .with_message(format!("this is of type `{}`", lhs.typ().fg(lhs_color)))
                        .with_color(lhs_color)
                        .with_order(3),
                ];

                let or_strings = if matches!(op.kind, crate::parse::BinOpKind::Plus) {
                    " or strings"
                }
                else {
                    ""
                };

                at.loc()
                    .report(ariadne::ReportKind::Error)
                    .with_message(message)
                    .with_labels(labels)
                    .with_help(format!(
                        "operator `{}` can only be applied to numbers{or_strings}",
                        op.token.slice().fg(op_color),
                    ))
                    .finish()
                    .print(at.loc().cache())?;
            }
        }
    }
}

pub fn run<'a>(
    bump: &'a Bump,
    args: impl IntoIterator<Item = impl Into<OsString> + Clone>,
) -> Result<(), Box<dyn std::error::Error + 'a>> {
    let args = Args::parse_from(args);
    if let Some(filename) = args.filename {
        let tokens = lex(
            bump,
            bump.alloc_path(&filename),
            bump.alloc_str(&std::fs::read_to_string(filename)?),
        )?;
        let ast = parse(program, bump, tokens)?;
        execute(ast)?;
    }
    else {
        repl()?;
    }
    Ok(())
}
