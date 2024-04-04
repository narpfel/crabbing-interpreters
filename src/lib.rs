#![feature(closure_lifetime_binder)]
#![feature(error_generic_member_access)]
#![feature(lint_reasons)]
#![feature(never_type)]

use std::ffi::OsStr;
use std::ffi::OsString;
use std::io;
use std::io::stdin;
use std::io::stdout;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use bumpalo::Bump;
use clap::Parser;

use crate::eval::execute;
use crate::eval::Environment;
use crate::eval::Value;
pub use crate::lex::lex;
use crate::parse::parse;
use crate::parse::program;

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

pub trait Report {
    fn print(&self);
    fn exit_code(&self) -> i32;
}

impl<'a, T> From<T> for Box<dyn Report + 'a>
where
    T: Report + 'a,
{
    fn from(value: T) -> Self {
        Box::new(value)
    }
}

impl Report for io::Error {
    fn print(&self) {
        eprintln!("{self:?}");
    }

    fn exit_code(&self) -> i32 {
        74
    }
}

struct IoError {
    path: PathBuf,
    io_error: io::Error,
}

impl Report for IoError {
    fn print(&self) {
        match self.io_error.kind() {
            io::ErrorKind::NotFound => eprintln!("{}: `{}`", self.io_error, self.path.display()),
            _ => eprintln!(
                "{} while trying to read file `{}`",
                self.io_error,
                self.path.display(),
            ),
        }
    }

    fn exit_code(&self) -> i32 {
        74
    }
}

/// Crabbing Interpreters
#[derive(Debug, Parser)]
struct Args {
    /// filename
    filename: Option<PathBuf>,
}

fn repl() -> Result<(), Box<dyn Report>> {
    let bump = &mut Bump::new();
    let mut globals = Environment::new();
    let mut line = String::new();
    'repl: loop {
        line.clear();
        let mut first = true;
        let stmts = loop {
            if first {
                print!("\x1B[0mλ» \x1B[1m");
            }
            else {
                print!("\x1B[0m.. \x1B[1m");
            }
            stdout().flush()?;
            let len = stdin().read_line(&mut line)?;
            print!("\x1B[0m");
            if len == 0 {
                if first {
                    return Ok(());
                }
                else {
                    println!();
                    break &[][..];
                }
            }
            first = false;
            let tokens = match lex(bump, "<input>", &line) {
                Ok(tokens) => tokens,
                Err(err) => {
                    err.print();
                    continue 'repl;
                }
            };
            match parse(program, bump, tokens) {
                Ok(stmts) => break stmts,
                Err(parse::Error::Eof(_)) => (),
                Err(err) => {
                    err.print();
                    continue 'repl;
                }
            };
        };
        let result = execute(&mut globals, stmts);
        match result {
            Ok(value) =>
                if !matches!(value, Value::Nil) {
                    println!("\x1B[38;2;170;034;255m\x1B[1m=> {}\x1B[0m", value);
                },
            Err(err) => err.print(),
        }
    }
}

pub fn run<'a>(
    bump: &'a Bump,
    args: impl IntoIterator<Item = impl Into<OsString> + Clone>,
) -> Result<(), Box<dyn Report + 'a>> {
    let args = Args::parse_from(args);
    if let Some(filename) = args.filename {
        let tokens = lex(
            bump,
            bump.alloc_path(&filename),
            bump.alloc_str(
                &std::fs::read_to_string(&filename)
                    .map_err(|err| IoError { path: filename, io_error: err })?,
            ),
        )?;
        let ast = parse(program, bump, tokens)?;
        let mut globals = Environment::new();
        execute(&mut globals, ast)?;
    }
    else {
        repl()?;
    }
    Ok(())
}
