#![feature(closure_lifetime_binder)]
#![feature(lint_reasons)]
#![feature(never_type)]
#![feature(stmt_expr_attributes)]

use std::cell::Cell;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt::Display;
use std::fmt::Write as _;
use std::io;
use std::io::stdin;
use std::io::stdout;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use bumpalo::Bump;
use clap::Parser;
use clap::ValueEnum;
pub(crate) use crabbing_interpreters_lex as lex;
pub use crabbing_interpreters_lex::lex;

use crate::eval::execute;
use crate::eval::ControlFlow;
use crate::eval::Environment;
use crate::eval::Value;
use crate::interner::Interner;
use crate::lex::Loc;
use crate::parse::parse;
use crate::parse::program;
use crate::parse::Name;
use crate::scope::resolve_names;

mod clone_from_cell;
mod eval;
mod interner;
mod nonempty;
mod parse;
mod rc_str;
mod rc_value;
mod scope;

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

trait Sliced {
    fn slice(&self) -> impl Display + '_;
}

impl Sliced for &str {
    fn slice(&self) -> impl Display + '_ {
        self
    }
}

impl Sliced for String {
    fn slice(&self) -> impl Display + '_ {
        self
    }
}

impl Sliced for usize {
    fn slice(&self) -> impl Display + '_ {
        *self
    }
}

impl Sliced for &usize {
    fn slice(&self) -> impl Display + '_ {
        self
    }
}

impl Sliced for &Value<'_> {
    fn slice(&self) -> impl Display + '_ {
        self
    }
}

trait IndentLines {
    fn indent_lines(&self, indent: usize) -> String;
}

impl IndentLines for str {
    fn indent_lines(&self, indent: usize) -> String {
        self.lines().fold(String::new(), |mut s, line| {
            writeln!(s, "{:indent$}{line}", "").unwrap();
            s
        })
    }
}

/// Crabbing Interpreters
#[derive(Debug, Parser)]
struct Args {
    /// filename
    filename: Option<PathBuf>,
    #[arg(short, long)]
    scopes: bool,
    #[arg(short, long)]
    times: bool,
    #[arg(long)]
    stop_at: Option<StopAt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StopAt {
    Scopes,
}

fn repl() -> Result<(), Box<dyn Report>> {
    let bump = &mut Bump::new();
    let mut interner = Interner::default();
    let clock = interner.intern("clock");
    let globals_names =
        bump.alloc_slice_copy(&[Name::new(clock, bump.alloc(Loc::debug_loc(bump, "clock")))]);
    let mut globals = Environment::new([(clock, 0)].into_iter().collect());
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
            let (tokens, eof_loc) = lex(bump, Path::new("<input>"), &line);
            match parse(program, bump, tokens, eof_loc, &mut interner) {
                Ok(stmts) => break stmts,
                Err(parse::Error::Eof { at: _ }) => (),
                Err(err) => {
                    err.print();
                    continue 'repl;
                }
            };
        };
        let (stmts, _, global_cell_count) = match resolve_names(bump, globals_names, stmts) {
            Ok(stmts) => stmts,
            Err(err) => {
                err.print();
                continue 'repl;
            }
        };
        let global_cells: Vec<_> = (0..global_cell_count)
            .map(|_| Cell::new(Rc::new(Cell::new(Value::Nil))))
            .collect();
        let result = execute(&mut globals, 0, stmts, &global_cells);
        match result {
            Ok(value) | Err(ControlFlow::Return(value)) =>
                if !matches!(value, Value::Nil) {
                    println!("\x1B[38;2;170;034;255m\x1B[1m=> {}\x1B[0m", value);
                },
            Err(ControlFlow::Error(err)) => err.print(),
        }
    }
}

fn time<T>(step: &str, print: bool, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = f();
    if print {
        println!("{step}: {:?}", start.elapsed());
    }
    result
}

pub fn run<'a>(
    bump: &'a Bump,
    args: impl IntoIterator<Item = impl Into<OsString> + Clone>,
) -> Result<(), Box<dyn Report + 'a>> {
    let args = Args::parse_from(args);
    if let Some(filename) = args.filename {
        let (tokens, eof_loc) = time("lex", args.times, || -> Result<_, Box<dyn Report>> {
            Ok(lex(
                bump,
                bump.alloc_path(&filename),
                bump.alloc_str(
                    &std::fs::read_to_string(&filename)
                        .map_err(|err| IoError { path: filename, io_error: err })?,
                ),
            ))
        })?;
        let mut interner = Interner::default();
        let ast = time("ast", args.times, || {
            parse(program, bump, tokens, eof_loc, &mut interner)
        })?;
        let globals = bump.alloc_slice_copy(&[Name::new(
            interner.intern("clock"),
            bump.alloc(Loc::debug_loc(bump, "clock")),
        )]);
        let (scoped_ast, global_name_offsets, global_cell_count) =
            time("scp", args.times, move || {
                scope::resolve_names(bump, globals, ast)
            })?;
        if args.scopes {
            print!("(program");
            if !scoped_ast.is_empty() {
                println!();
            }
            let mut sexpr = String::new();
            for stmt in scoped_ast {
                write!(sexpr, "{}", stmt.as_sexpr(3)).unwrap();
            }
            println!("{})", sexpr.trim_end());
            if args.stop_at == Some(StopAt::Scopes) {
                return Ok(());
            }
        }
        let global_name_offsets = global_name_offsets
            .iter()
            .map(|(&name, v)| match v.target() {
                scope::Target::GlobalBySlot(slot) => (name, slot),
                _ => unreachable!(),
            })
            .collect();
        let mut stack = time("stk", args.times, || Environment::new(global_name_offsets));
        let global_cells: Vec<_> = (0..global_cell_count)
            .map(|_| Cell::new(Rc::new(Cell::new(Value::Nil))))
            .collect();
        match time("exe", args.times, || {
            execute(&mut stack, 0, scoped_ast, &global_cells)
        }) {
            Ok(_) | Err(ControlFlow::Return(_)) => (),
            Err(ControlFlow::Error(err)) => Err(err)?,
        };
    }
    else {
        repl()?;
    }
    Ok(())
}
