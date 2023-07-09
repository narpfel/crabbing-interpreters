#![feature(closure_lifetime_binder)]
#![feature(error_generic_member_access)]
#![feature(lint_reasons)]
#![feature(never_type)]

use std::ffi::OsStr;
use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;

use bumpalo::Bump;
use clap::Parser;

pub use crate::lex::lex;
pub use crate::parse::parse;

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
    /// output token stream (for testing the lexer)
    #[arg(long)]
    test_lexer: bool,
    /// output parsed expression as an s-expression (for testing the parser)
    #[arg(long)]
    test_parser: bool,
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
        if args.test_lexer {
            for token in tokens {
                println!("{}", token.as_debug_string());
            }
            println!("EOF  null");
            return Ok(());
        }
        let ast = parse(bump, tokens)?;
        if args.test_parser {
            println!("{}", ast.as_sexpr());
            return Ok(());
        }
    }
    else {
        todo!("repl");
    }
    Ok(())
}
