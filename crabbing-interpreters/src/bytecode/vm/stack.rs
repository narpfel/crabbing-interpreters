use std::fmt;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::panic::Location;
use std::ptr::NonNull;
use std::slice::SliceIndex;

use crate::gc::Trace;

pub(crate) struct Stack<T> {
    stack: AbortOnOutOfBounds<T, { super::USEABLE_STACK_SIZE_IN_ELEMENTS }>,
    pointer: usize,
}

impl<T> Stack<T> {
    pub(super) fn into_raw_parts(self) -> (NonNull<T>, NonNull<T>) {
        assert!(self.pointer < self.stack.len());
        let ptr = ManuallyDrop::new(self.stack).0.cast();
        (ptr, unsafe { ptr.add(self.pointer) })
    }

    pub(super) unsafe fn from_raw_parts(base: NonNull<T>, sp: NonNull<T>) -> Self {
        let pointer = unsafe { sp.sub_ptr(base) };
        Self {
            stack: AbortOnOutOfBounds(base.cast()),
            pointer,
        }
    }

    #[track_caller]
    pub(super) unsafe fn swap(&mut self, i: u32, j: u32) -> Result<(), ()> {
        self.stack.swap(
            self.pointer - 1 - usize::try_from(i).unwrap(),
            self.pointer - 1 - usize::try_from(j).unwrap(),
        )
    }
}

impl<T> Stack<T>
where
    T: Copy,
{
    pub(super) fn new(default_value: T) -> Self {
        Self {
            stack: AbortOnOutOfBounds::from(
                Box::try_from(
                    vec![default_value; super::USEABLE_STACK_SIZE_IN_ELEMENTS].into_boxed_slice(),
                )
                .unwrap_or_else(|_| unreachable!()),
            ),
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

struct AbortOnOutOfBounds<T, const N: usize>(NonNull<[T; N]>);

impl<T, const N: usize> AbortOnOutOfBounds<T, N> {
    #[track_caller]
    fn swap(&mut self, i: usize, j: usize) -> Result<(), ()> {
        if i != j {
            match self.deref_mut().get_many_mut([i, j]) {
                Ok([x, y]) => {
                    std::mem::swap(x, y);
                    Ok(())
                }
                Err(_) => {
                    index_failed::<T, _>(self.len(), i, Location::caller());
                    index_failed::<T, _>(self.len(), j, Location::caller());
                    Err(())
                }
            }
        }
        else {
            Ok(())
        }
    }
}

impl<T, const N: usize> Drop for AbortOnOutOfBounds<T, N> {
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.0.as_ptr()) })
    }
}

impl<T, const N: usize> From<Box<[T; N]>> for AbortOnOutOfBounds<T, N> {
    fn from(boxed_array: Box<[T; N]>) -> Self {
        let ptr = NonNull::new(Box::into_raw(boxed_array)).unwrap();
        AbortOnOutOfBounds(ptr)
    }
}

impl<T, const N: usize> Deref for AbortOnOutOfBounds<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.cast().as_ref() }
    }
}

impl<T, const N: usize> DerefMut for AbortOnOutOfBounds<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.cast().as_mut() }
    }
}

impl<T, Idx, const N: usize> Index<Idx> for AbortOnOutOfBounds<T, N>
where
    Idx: SliceIndex<[T]> + std::fmt::Debug + Clone,
{
    type Output = <[T; N] as Index<Idx>>::Output;

    #[track_caller]
    fn index(&self, index: Idx) -> &Self::Output {
        self.get(index.clone()).unwrap_or_else(
            #[track_caller]
            || {
                index_failed(self.len(), index, Location::caller());
                std::process::abort()
            },
        )
    }
}

impl<T, Idx, const N: usize> IndexMut<Idx> for AbortOnOutOfBounds<T, N>
where
    Idx: SliceIndex<[T]> + std::fmt::Debug + Clone,
{
    #[track_caller]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        let len = self.len();
        self.get_mut(index.clone()).unwrap_or_else(
            #[track_caller]
            || {
                index_failed(len, index, Location::caller());
                std::process::abort()
            },
        )
    }
}

#[cold]
#[inline(never)]
extern "C" fn index_failed<T, Idx>(len: usize, index: Idx, location: &Location)
where
    Idx: SliceIndex<[T]> + std::fmt::Debug + Clone,
{
    eprintln!(
        "thread '{}' panicked at {}:\nindex operation failed: the len is {} but the index is `{:?}`",
        std::thread::current().name().unwrap_or("<unknown>"),
        location,
        len,
        index,
    )
}
