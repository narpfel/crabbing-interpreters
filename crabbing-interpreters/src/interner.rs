use rustc_hash::FxHashMap as HashMap;

pub mod interned {
    use super::InternedString;

    pub const CLOCK: InternedString = InternedString(0);
    pub const INIT: InternedString = InternedString(1);
    pub const THIS: InternedString = InternedString(2);
    pub const SUPER: InternedString = InternedString(3);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InternedString(u32);

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s.{}", self.0)
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
}
