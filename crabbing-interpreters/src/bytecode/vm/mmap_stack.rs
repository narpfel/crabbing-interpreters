use std::fmt;
use std::io;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use crate::gc::Trace;

pub(crate) struct Stack<T> {
    stack: NonNull<T>,
    pointer: NonNull<T>,
}

impl<T> Stack<T> {
    pub(super) fn new(_default_value: T) -> Self {
        let stack = unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                Self::SIZE_IN_BYTES,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                panic!("mmap failed: {}", io::Error::last_os_error());
            }

            #[cfg(not(miri))]
            {
                let result = libc::mprotect(ptr, Self::START_OFFSET, libc::PROT_NONE);
                if result != 0 {
                    panic!("first mprotect failed: {}", io::Error::last_os_error());
                }

                let result = libc::mprotect(
                    ptr.byte_add(Self::START_OFFSET + Self::USEABLE_SIZE_IN_BYTES),
                    Self::PAGE_SIZE * Self::GUARD_PAGE_COUNT,
                    libc::PROT_NONE,
                );
                if result != 0 {
                    panic!("second mprotect failed: {}", io::Error::last_os_error());
                }
            }

            NonNull::new(ptr).unwrap().cast()
        };

        Self {
            stack,
            pointer: unsafe { stack.byte_add(Self::START_OFFSET) },
        }
    }

    pub(super) fn into_raw_parts(self) -> (NonNull<T>, NonNull<T>) {
        let me = ManuallyDrop::new(self);
        (me.useable_stack_start(), me.pointer)
    }

    pub(super) unsafe fn from_raw_parts(stack: NonNull<T>, pointer: NonNull<T>) -> Self {
        Self {
            stack: unsafe { stack.byte_sub(Self::START_OFFSET) },
            pointer,
        }
    }
}

impl<T> Drop for Stack<T> {
    fn drop(&mut self) {
        unsafe {
            NonNull::from(self.used_stack_mut()).drop_in_place();

            let result = libc::munmap(self.stack.as_ptr().cast(), Self::SIZE_IN_BYTES);
            if result != 0 {
                panic!("munmap failed: {}", io::Error::last_os_error());
            }
        }
    }
}

impl<T> Stack<T>
where
    T: Copy,
{
    pub(super) fn push(&mut self, value: T) {
        debug_assert!(self.is_in_bounds(0));
        unsafe {
            self.pointer.write(value);
            self.pointer = self.pointer.add(1);
        }
    }

    pub(super) fn pop(&mut self) -> T {
        debug_assert!(self.is_in_bounds(-1));
        unsafe {
            self.pointer = self.pointer.sub(1);
            self.pointer.read()
        }
    }

    pub(super) fn peek(&self) -> T {
        self.short_peek_at(0)
    }

    pub(super) fn short_peek_at(&self, index: u32) -> T {
        let offset = peek_offset(index);
        debug_assert!(offset.unsigned_abs() < Self::ELEMENT_COUNT_IN_GUARD_AREA);
        unsafe { self.peek_at_unchecked(offset) }
    }

    pub(super) fn peek_at(&self, index: u32) -> T {
        let offset = peek_offset(index);
        assert!(self.is_in_bounds(offset));
        unsafe { self.peek_at_unchecked(offset) }
    }

    unsafe fn peek_at_unchecked(&self, offset: isize) -> T {
        debug_assert!(self.is_in_bounds(offset));
        unsafe { self.pointer.offset(offset).read() }
    }
}

impl<T> Stack<T> {
    pub(super) fn peek_at_mut(&mut self, index: u32) -> &mut T {
        let offset = peek_offset(index);
        assert!(self.is_in_bounds(offset));
        unsafe { self.pointer.offset(offset).as_mut() }
    }

    #[must_use]
    fn is_in_bounds(&self, offset: isize) -> bool {
        let target_ptr = self.pointer.as_ptr().wrapping_offset(offset);
        self.useable_stack_start().as_ptr() <= target_ptr
            && target_ptr
                < self
                    .useable_stack_start()
                    .as_ptr()
                    .wrapping_byte_add(Self::USEABLE_SIZE_IN_BYTES)
    }

    #[must_use]
    pub(super) fn is_empty(&self) -> bool {
        self.pointer == self.useable_stack_start()
    }

    fn useable_stack_start(&self) -> NonNull<T> {
        unsafe { self.stack.byte_add(Self::START_OFFSET) }
    }

    fn used_stack(&self) -> &[T] {
        unsafe {
            std::slice::from_ptr_range(self.useable_stack_start().as_ptr()..self.pointer.as_ptr())
        }
    }

    fn used_stack_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_mut_ptr_range(
                self.useable_stack_start().as_ptr()..self.pointer.as_ptr(),
            )
        }
    }
}

unsafe impl<T> Trace for Stack<T>
where
    T: Trace,
{
    fn trace(&self) {
        self.used_stack().trace()
    }
}

impl<T> fmt::Debug for Stack<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.used_stack()).finish()
    }
}

fn peek_offset(index: u32) -> isize {
    -1 - isize::try_from(index).unwrap()
}
