#![feature(closure_lifetime_binder)]
#![feature(closure_track_caller)]
#![feature(debug_closure_helpers)]
#![feature(explicit_tail_calls)]
#![feature(macro_metavar_expr)]
#![feature(never_type)]
#![feature(ptr_metadata)]
#![feature(rust_cold_cc)]
#![feature(slice_from_ptr_range)]
#![feature(slice_ptr_get)]
#![feature(stmt_expr_attributes)]
#![feature(substr_range)]
#![warn(clippy::as_conversions)]
#![expect(incomplete_features)]

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
use std::time::Instant;

use bumpalo::Bump;
use clap::Parser;
use clap::ValueEnum;
pub(crate) use crabbing_interpreters_lex as lex;
pub use crabbing_interpreters_lex::lex;
use itertools::Itertools as _;

use crate::bytecode::compile_program;
use crate::bytecode::run_bytecode;
use crate::bytecode::Bytecode;
use crate::bytecode::CompiledBytecodes;
use crate::bytecode::Vm;
use crate::closure_compiler::compile_block;
use crate::closure_compiler::State;
use crate::environment::Environment;
use crate::eval::execute;
use crate::eval::ControlFlow;
pub use crate::gc::Gc;
use crate::gc::GcRef;
use crate::interner::Interner;
use crate::lex::Loc;
use crate::parse::parse;
use crate::parse::program;
use crate::parse::Name;
use crate::scope::resolve_names;
use crate::value::Value;

mod bytecode;
mod closure_compiler;
mod environment;
mod eval;
mod gc;
mod hash_map;
mod interner;
mod nonempty;
mod parse;
mod scope;
mod value;

const EMPTY: &str = "";
const DEBUG_INDENT: usize = 8;

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
    #[arg(long)]
    /// print stack layout
    layout: bool,
    #[arg(short, long)]
    scopes: bool,
    /// print bytecode
    #[arg(short, long)]
    bytecode: bool,
    #[arg(short, long)]
    times: bool,
    #[arg(long)]
    stop_at: Option<StopAt>,
    #[arg(short, long, value_enum, default_value_t)]
    r#loop: Loop,
    #[arg(long)]
    show_bytecode_execution_counts: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StopAt {
    Ast,
    Scopes,
    Bytecode,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Loop {
    #[default]
    Ast,
    Closures,
    Bytecode,
    Threaded,
}

fn repl(args: &Args) -> Result<(), Box<dyn Report>> {
    let bump = &mut Bump::new();
    let mut interner = Interner::default();
    let clock = interner.intern("clock");
    let native_function_test = interner.intern("native_function_test");
    let read_file = interner.intern("read_file");
    let mut globals_names = bump.alloc_slice_copy(&[
        Name::new(clock, bump.alloc(Loc::debug_loc(bump, "clock"))),
        Name::new(
            native_function_test,
            bump.alloc(Loc::debug_loc(bump, "native_function_test")),
        ),
        Name::new(read_file, bump.alloc(Loc::debug_loc(bump, "read_file"))),
    ]);
    let gc = &Gc::default();
    let mut globals = Environment::new(
        gc,
        [clock, native_function_test, read_file]
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect(),
        GcRef::from_iter_in(gc, [].into_iter()),
    );
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
        let program = match resolve_names(bump, globals_names, stmts) {
            Ok(program) => program,
            Err(err) => {
                err.print();
                continue 'repl;
            }
        };
        if args.scopes {
            println!("{}", program.stmts_as_sexpr(3));
        }
        let global_cells = GcRef::from_iter_in(
            globals.gc,
            (0..program.global_cell_count)
                .map(|_| Cell::new(GcRef::new_in(gc, Cell::new(Value::Nil.into_nanboxed())))),
        );
        let result = execute(&mut globals, 0, program.stmts, global_cells, &|| ());
        match result {
            Ok(value) | Err(ControlFlow::Return(value)) =>
                if !matches!(value, Value::Nil) {
                    println!("\x1B[38;2;170;034;255m\x1B[1m=> {value}\x1B[0m");
                },
            Err(ControlFlow::Error(err)) => err.print(),
        }
        globals_names = bump.alloc_slice_copy(
            &globals_names
                .iter()
                .chain(
                    program
                        .global_name_offsets
                        .values()
                        .map(|variable| variable.name),
                )
                .copied()
                .unique_by(Name::id)
                .collect_vec(),
        );
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
    gc: &'a Gc,
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

        if args.stop_at == Some(StopAt::Ast) {
            return Ok(());
        }

        let globals = &*bump.alloc_slice_copy(&[
            Name::new(
                interner.intern("clock"),
                bump.alloc(Loc::debug_loc(bump, "clock")),
            ),
            Name::new(
                interner.intern("native_function_test"),
                bump.alloc(Loc::debug_loc(bump, "native_function_test")),
            ),
            Name::new(
                interner.intern("read_file"),
                bump.alloc(Loc::debug_loc(bump, "read_file")),
            ),
        ]);
        let program = time("scp", args.times, move || {
            scope::resolve_names(bump, globals, ast)
        })?;

        if args.layout {
            println!("{}\n", program.scopes.as_sexpr("layout", 3));
        }

        if args.scopes {
            println!("{}", program.stmts_as_sexpr(3));
        }
        if args.stop_at == Some(StopAt::Scopes) {
            return Ok(());
        }
        if args.scopes {
            println!();
        }

        let global_name_offsets = program
            .global_name_offsets
            .iter()
            .map(|(&name, v)| match v.target() {
                scope::Target::GlobalBySlot(slot) => (name, slot),
                _ => unreachable!(),
            })
            .collect();
        let global_cells = GcRef::from_iter_in(
            gc,
            (0..program.global_cell_count)
                .map(|_| Cell::new(GcRef::new_in(gc, Cell::new(Value::Nil.into_nanboxed())))),
        );
        let mut stack = time("stk", args.times, || {
            Environment::new(gc, global_name_offsets, global_cells)
        });
        let execute_closures = time("clo", args.times, || compile_block(bump, program.stmts));
        let (bytecode, constants, metadata, error_locations) =
            time("cmp", args.times, || compile_program(gc, program.stmts));
        let compiled_bytecodes = time("thr", args.times, || {
            bytecode
                .iter()
                .map(|bytecode| bytecode.compile())
                .collect_vec()
        });
        let compiled_bytecodes = CompiledBytecodes::new(&compiled_bytecodes);

        if args.bytecode {
            println!("Interned strings");
            interner.print_interned_strings();
            println!();

            println!("Metadata");
            for (i, metadata) in metadata.iter().enumerate() {
                print!("{i:>DEBUG_INDENT$}:  ");
                let metadata_string = format!("{metadata:#?}");
                let mut lines = metadata_string.lines();
                println!("{}", lines.next().unwrap());
                for line in lines {
                    println!("{EMPTY:>DEBUG_INDENT$}|  {line}");
                }
            }
            println!();

            println!("Constants");
            for (i, constant) in constants.iter().enumerate() {
                println!("{i:>DEBUG_INDENT$}:  {}", constant.parse().lox_debug());
            }
            println!();

            println!("Bytecode");
            for (i, bytecode) in bytecode.iter().enumerate() {
                println!("{i:>DEBUG_INDENT$}   {bytecode}");
            }
        }
        if args.stop_at == Some(StopAt::Bytecode) {
            return Ok(());
        }
        if args.bytecode {
            println!()
        }

        let execution_result = time(
            "exe",
            args.times,
            || -> Result<Value, ControlFlow<Value, Box<dyn Report>>> {
                let result = match args.r#loop {
                    Loop::Ast => execute(&mut stack, 0, program.stmts, global_cells, &|| ())?,
                    Loop::Closures => execute_closures(&mut State {
                        env: &mut stack,
                        offset: 0,
                        cell_vars: global_cells,
                        trace_call_stack: &|| (),
                    })?,
                    Loop::Bytecode => run_bytecode(&mut Vm::new(
                        &bytecode,
                        &constants,
                        &metadata,
                        &error_locations,
                        stack,
                        global_cells,
                        compiled_bytecodes,
                    )?)?,
                    Loop::Threaded => {
                        let mut vm = Vm::new(
                            &bytecode,
                            &constants,
                            &metadata,
                            &error_locations,
                            stack,
                            global_cells,
                            compiled_bytecodes,
                        )?;
                        vm.run_threaded();

                        if args.show_bytecode_execution_counts {
                            let max_len = Bytecode::all_discriminants()
                                .map(Bytecode::name)
                                .map(str::len)
                                .into_iter()
                                .max()
                                .unwrap();
                            eprintln!("Bytecode execution counts");
                            for (discriminant, count) in vm.execution_counts().iter().enumerate() {
                                eprintln!(
                                    "{:>max_len$} ({discriminant:>2}): {count:>12}",
                                    Bytecode::name(discriminant),
                                );
                            }
                            eprintln!(
                                "{:>max_len$}     : {:>12}",
                                "Total",
                                vm.execution_counts().iter().sum::<u64>(),
                            );
                        }

                        match vm.error() {
                            Some(error) => Err(error)?,
                            None => Value::Nil,
                        }
                    }
                };
                Ok(result)
            },
        );
        match execution_result {
            Ok(_) | Err(ControlFlow::Return(_)) => (),
            Err(ControlFlow::Error(err)) => Err(err)?,
        };
    }
    else {
        repl(&args)?;
    }
    Ok(())
}
