use std::process::exit;

use bumpalo::Bump;

fn main() {
    let bump = &Bump::new();
    let result = crabbing_interpreters::run(bump, std::env::args_os());
    if let Err(err) = result {
        err.print();
        exit(err.exit_code());
    }
}