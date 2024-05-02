use std::process::exit;

use bumpalo::Bump;
use crabbing_interpreters::Gc;

fn main() {
    let bump = &Bump::new();
    let gc = &Gc::default();
    let result = crabbing_interpreters::run(bump, gc, std::env::args_os());
    if let Err(err) = result {
        err.print();
        exit(err.exit_code());
    }
}
