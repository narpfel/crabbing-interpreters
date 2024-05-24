use std::fmt;

use crate::gc::Trace;

pub(crate) struct Stack<T> {
    stack: Box<[T; super::USEABLE_STACK_SIZE_IN_ELEMENTS]>,
    pointer: usize,
}

impl<T> Stack<T>
where
    T: Copy,
{
    pub(super) fn new(default_value: T) -> Self {
        Self {
            stack: Box::try_from(
                vec![default_value; super::USEABLE_STACK_SIZE_IN_ELEMENTS].into_boxed_slice(),
            )
            .unwrap_or_else(|_| unreachable!()),
            pointer: 0,
        }
    }

    pub(super) fn push(&mut self, value: T) {
        self.stack[self.pointer] = value;
        self.pointer += 1;
    }

    pub(super) fn pop(&mut self) -> T {
        self.pointer -= 1;
        self.stack[self.pointer]
    }

    pub(super) fn peek(&self) -> T {
        self.short_peek_at(0)
    }

    pub(super) fn short_peek_at(&self, index: u32) -> T {
        self.peek_at(index)
    }

    pub(super) fn peek_at(&self, index: u32) -> T {
        self.stack[self.pointer - 1 - usize::try_from(index).unwrap()]
    }

    pub(super) fn peek_at_mut(&mut self, index: u32) -> &mut T {
        &mut self.stack[self.pointer - 1 - usize::try_from(index).unwrap()]
    }

    #[must_use]
    pub(super) fn is_empty(&self) -> bool {
        self.pointer == 0
    }
}

unsafe impl<T> Trace for Stack<T>
where
    T: Trace,
{
    fn trace(&self) {
        self.stack[..self.pointer].trace()
    }
}

impl<T> fmt::Debug for Stack<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.stack[..self.pointer]).finish()
    }
}
