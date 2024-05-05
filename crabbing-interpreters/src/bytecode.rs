mod compiler;
mod vm;

pub(crate) use compiler::compile_program;
use compiler::Bytecode;
pub(crate) use vm::run_bytecode;
