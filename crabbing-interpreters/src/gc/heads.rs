use std::cell::Cell;
use std::ptr::NonNull;

use crate::gc::Action;
use crate::gc::GcHead;
use crate::gc::State;

type HeadPtr = NonNull<Cell<GcHead>>;

#[derive(Default)]
pub(super) struct Heads(Cell<Option<HeadPtr>>);

impl Heads {
    pub(super) fn push(&self, head_ptr: NonNull<Cell<GcHead>>) {
        let first = self.0.get();
        let head_ref = unsafe { head_ptr.as_ref() };
        let mut head = head_ref.get();
        debug_assert!(head.next.is_none());
        head.next = first;
        head_ref.set(head);
        self.0.set(Some(head_ptr));
    }

    /// # Safety
    ///
    /// `head` and `prev` must be contained in `self`.
    unsafe fn remove<'a>(&'a self, head: &'a Cell<GcHead>, prev: Option<HeadPtr>) {
        let next = head.get().next;
        head.set(GcHead { next: None, ..head.get() });
        match prev {
            Some(prev) => {
                let prev = unsafe { prev.as_ref() };
                debug_assert_eq!(prev.get().next, Some(NonNull::from_ref(head)));
                prev.set(GcHead { next, ..prev.get() });
            }
            None => self.0.set(next),
        }
    }

    pub(super) fn iter_with<F>(&self, f: impl Fn(State) -> Action<F>)
    where
        F: Fn(HeadPtr),
    {
        let mut ptr = self.0.get();
        let mut prev = None;
        while let Some(head_ptr) = ptr {
            let head_ref = unsafe { head_ptr.as_ref() };
            let head = head_ref.get();
            match f(head.state()) {
                Action::Keep => {
                    head_ref.set(head.with_state(State::Unvisited));
                    prev = ptr;
                }
                Action::Drop => {
                    unsafe { self.remove(head_ref, prev) };

                    // SAFETY: we have removed the value from the list of gc’d values, so it will
                    // not be seen again in a future collection. Therefore, this is the only `drop`
                    // call for this value.
                    unsafe { (head.drop)(head_ptr.cast(), head.length()) }
                }
                Action::Immortalise(immortalise) => {
                    unsafe { self.remove(head_ref, prev) };
                    immortalise(head_ptr);
                }
            }
            ptr = head.next;
        }
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.0.get().is_none()
    }
}

impl Drop for Heads {
    fn drop(&mut self) {
        self.iter_with(|_| match false {
            // hack so that the generic parameter can be inferred
            true => Action::Immortalise(|_| unreachable!()),
            false => Action::Drop,
        });
    }
}
