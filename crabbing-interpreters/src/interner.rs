use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;

use crate::DEBUG_INDENT;

pub mod interned {
    use super::InternedString;

    pub const CLOCK: InternedString = InternedString(0);
    pub const INIT: InternedString = InternedString(1);
    pub const THIS: InternedString = InternedString(2);
    pub const SUPER: InternedString = InternedString(3);
    pub const NATIVE_FUNCTION_TEST: InternedString = InternedString(4);
    pub const READ_FILE: InternedString = InternedString(5);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternedString(u32);

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("s.{}", self.0))
    }
}

#[derive(Clone)]
pub struct Interner<'a> {
    interned_strings: HashMap<&'a str, InternedString>,
}

impl Default for Interner<'_> {
    fn default() -> Self {
        Self {
            interned_strings: [
                ("clock", interned::CLOCK),
                ("init", interned::INIT),
                ("this", interned::THIS),
                ("super", interned::SUPER),
                ("native_function_test", interned::NATIVE_FUNCTION_TEST),
                ("read_file", interned::READ_FILE),
            ]
            .into_iter()
            .collect(),
        }
    }
}

impl<'a> Interner<'a> {
    pub fn intern(&mut self, s: &'a str) -> InternedString {
        let next_index = self.interned_strings.len();
        *self
            .interned_strings
            .entry(s)
            .or_insert(InternedString(next_index.try_into().unwrap()))
    }

    pub(crate) fn print_interned_strings(&self) {
        let interned_strings = self
            .interned_strings
            .iter()
            .sorted_by_key(|(&s, &interned)| (interned, s));
        for (s, interned) in interned_strings {
            println!("{interned:>DEBUG_INDENT$}:  {s:?}");
        }
    }
}
