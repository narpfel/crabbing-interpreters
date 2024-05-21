use std::fmt;
use std::io;
use std::ptr::NonNull;

use crate::environment::ENV_SIZE;
use crate::gc::Trace;

pub(super) struct Stack<T> {
    stack: NonNull<T>,
    pointer: NonNull<T>,
}

impl<T> Stack<T> {
    #[cfg(not(miri))]
    const GUARD_PAGE_COUNT_AFTER: usize = 1;
    #[cfg(miri)]
    const GUARD_PAGE_COUNT_AFTER: usize = 0;
    #[cfg(not(miri))]
    const GUARD_PAGE_COUNT_BEFORE: usize = 1;
    #[cfg(miri)]
    const GUARD_PAGE_COUNT_BEFORE: usize = 0;
    const PAGE_SIZE: usize = 4096;
    const SIZE_IN_BYTES: usize = Self::SIZE_IN_PAGES * Self::PAGE_SIZE;
    const SIZE_IN_PAGES: usize = Self::GUARD_PAGE_COUNT_BEFORE
        + Self::GUARD_PAGE_COUNT_AFTER
        + Self::USEABLE_SIZE_IN_BYTES / Self::PAGE_SIZE;
    const START_OFFSET: usize = Self::PAGE_SIZE * Self::GUARD_PAGE_COUNT_BEFORE;
    const USEABLE_SIZE_IN_BYTES: usize = ENV_SIZE.next_power_of_two() * std::mem::size_of::<T>();
    const _ASSERT_CORRECT_ALIGNMENT: () = assert!(Self::PAGE_SIZE >= std::mem::align_of::<T>());
    const _ASSERT_PAGE_SIZE_IS_MULTIPLE_OF_ELEMENT_SIZE: () =
        assert!(Self::PAGE_SIZE % std::mem::size_of::<T>() == 0);
    const _ASSERT_STACK_HAS_SIZE: () = assert!(Self::SIZE_IN_PAGES > 2);

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
                    Self::PAGE_SIZE * Self::GUARD_PAGE_COUNT_AFTER,
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
        unsafe {
            self.pointer.write(value);
            self.pointer = self.pointer.add(1);
        }
    }

    pub(super) fn pop(&mut self) -> T {
        unsafe {
            self.pointer = self.pointer.sub(1);
            self.pointer.read()
        }
    }

    pub(super) fn peek(&self) -> T {
        self.short_peek_at(0)
    }

    pub(super) fn short_peek_at(&self, index: u32) -> T {
        debug_assert!(index < 256);
        debug_assert!(self.is_in_bounds(index));
        unsafe { self.peek_at_unchecked(index) }
    }

    pub(super) fn peek_at(&self, index: u32) -> T {
        assert!(self.is_in_bounds(index));
        unsafe { self.peek_at_unchecked(index) }
    }

    unsafe fn peek_at_unchecked(&self, index: u32) -> T {
        debug_assert!(self.is_in_bounds(index));
        let index = usize::try_from(index).unwrap();
        unsafe { self.pointer.sub(1 + index).read() }
    }
}

impl<T> Stack<T> {
    pub(super) fn peek_at_mut(&mut self, index: u32) -> &mut T {
        assert!(self.is_in_bounds(index));
        let index = usize::try_from(index).unwrap();
        unsafe { self.pointer.sub(1 + index).as_mut() }
    }

    #[must_use]
    fn is_in_bounds(&self, index: u32) -> bool {
        let index = usize::try_from(index).unwrap();
        let target_ptr = self.pointer.as_ptr().wrapping_sub(1 + index);
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
