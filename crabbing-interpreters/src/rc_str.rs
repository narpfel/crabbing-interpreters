use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum RcStr<'a> {
    Borrowed(&'a str),
    Owned(Rc<str>),
}

impl Deref for RcStr<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match self {
            RcStr::Borrowed(s) => s,
            RcStr::Owned(s) => s,
        }
    }
}

impl PartialEq for RcStr<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for RcStr<'_> {}

impl PartialOrd for RcStr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RcStr<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.deref().cmp(other)
    }
}

impl std::fmt::Display for RcStr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}
