use core::fmt;
use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::ptr::{self};

#[cfg(not(miri))]
const COLLECTION_INTERVAL: usize = 100_000;

#[cfg(miri)]
const COLLECTION_INTERVAL: usize = 1;

type BoxedValue<'a, T> = Box<GcValue<'a, T>>;

enum Action {
    Keep,
    Drop,
}

// FIXME: document safety
#[expect(clippy::missing_safety_doc)]
pub(crate) unsafe trait Trace {
    fn trace(&self, f: &dyn Fn(GcRoot));
}

unsafe impl Trace for () {
    fn trace(&self, _f: &dyn Fn(GcRoot)) {}
}

unsafe impl Trace for u64 {
    fn trace(&self, _f: &dyn Fn(GcRoot)) {}
}

unsafe impl Trace for String {
    fn trace(&self, _f: &dyn Fn(GcRoot)) {}
}

unsafe impl<T> Trace for Cell<T>
where
    T: Copy + Trace,
{
    fn trace(&self, tracer: &dyn Fn(GcRoot)) {
        let value = self.get();
        value.trace(tracer);
        self.set(value);
    }
}

unsafe impl<T> Trace for GcRef<'_, T>
where
    T: Trace,
{
    fn trace(&self, tracer: &dyn Fn(GcRoot)) {
        tracer(self.as_root());
        (self.0.head.get().vtable.iter_children)(NonNull::from(self.0).cast(), tracer);
    }
}

#[derive(Default)]
pub struct Gc {
    last: Cell<Option<NonNull<Cell<GcHead>>>>,
    allocation_count: Cell<usize>,
}

impl Gc {
    fn alloc<'a, T>(&'a self, value: T) -> &'a GcValue<'a, T>
    where
        T: Trace,
    {
        self.allocation_count.set(self.allocation_count.get() + 1);
        let last = self.last.get();

        let gc_value = BoxedValue::<'a, T>::into_raw(BoxedValue::<'a, T>::new(GcValue {
            head: Cell::new(GcHead {
                prev: last,
                next: None,
                vtable: &VTable {
                    drop: |p| drop(unsafe { BoxedValue::<'a, T>::from_raw(p.cast().as_ptr()) }),
                    iter_children: |head_ptr, tracer| {
                        (unsafe { head_ptr.cast::<GcValue<'a, T>>().as_ref() })
                            .value
                            .trace(tracer)
                    },
                },
                state: State::Unvisited,
            }),
            value,
            _gc: PhantomData,
        }));

        let head_ptr = NonNull::new(unsafe { ptr::addr_of_mut!((*gc_value).head) });

        if let Some(last) = last {
            let last = unsafe { last.as_ref() };
            last.set(GcHead { next: head_ptr, ..last.get() });
        }

        self.last.set(head_ptr);
        unsafe { &*gc_value }
    }

    #[expect(clippy::only_used_in_recursion)]
    pub(crate) fn mark_recursively(&self, root: GcRoot) {
        let root_ptr = root.0;
        let root = unsafe { root_ptr.as_ref() };
        let state = root.get().state;
        match state {
            State::Unvisited => {
                root.set(GcHead {
                    state: State::VisitingChildren,
                    ..root.get()
                });
                (root.get().vtable.iter_children)(root_ptr.cast(), &|root| {
                    self.mark_recursively(root)
                });
                root.set(GcHead { state: State::Done, ..root.get() });
            }
            State::VisitingChildren => (),
            State::Done => (),
        }
    }

    pub(crate) unsafe fn sweep(&self) {
        self.allocation_count.set(0);
        self.iter_with(|head| match head.state {
            State::Unvisited => Action::Drop,
            State::Done => Action::Keep,
            State::VisitingChildren => unreachable!(),
        });
    }

    fn iter_with(&self, f: impl Fn(GcHead) -> Action) {
        let mut ptr = self.last.get();
        while let Some(head_ptr) = ptr {
            let head_ref = unsafe { head_ptr.as_ref() };
            let head = head_ref.get();
            ptr = head.prev;
            match f(head) {
                Action::Keep => head_ref.set(GcHead { state: State::Unvisited, ..head }),
                Action::Drop => {
                    let prev = head.prev;
                    let next = head.next;

                    if let Some(prev) = prev {
                        let prev = unsafe { prev.as_ref() };
                        prev.set(GcHead { next, ..prev.get() });
                    }
                    else {
                        // `head` is first
                        self.last.set(next);
                    }

                    if let Some(next) = next {
                        let next = unsafe { next.as_ref() };
                        next.set(GcHead { prev, ..next.get() });
                    }
                    else {
                        // `head` is last
                        self.last.set(ptr);
                    }

                    (head.vtable.drop)(head_ptr.cast())
                }
            }
        }
    }

    pub(crate) fn collection_necessary(&self) -> bool {
        self.allocation_count.get() >= COLLECTION_INTERVAL
    }
}

impl Drop for Gc {
    fn drop(&mut self) {
        self.iter_with(|_| Action::Drop);
    }
}

#[repr(C)]
struct GcValue<'gc, T> {
    head: Cell<GcHead>,
    value: T,
    _gc: PhantomData<&'gc Gc>,
}

#[derive(Debug, Clone, Copy)]
struct GcHead {
    prev: Option<NonNull<Cell<GcHead>>>,
    next: Option<NonNull<Cell<GcHead>>>,
    vtable: &'static VTable,
    state: State,
}

#[derive(Debug, Clone, Copy)]
struct VTable {
    drop: fn(NonNull<()>),
    iter_children: fn(NonNull<()>, &dyn Fn(GcRoot)),
}

#[derive(Debug, Clone, Copy)]
enum State {
    Unvisited,
    VisitingChildren,
    Done,
}

#[derive(Clone, Copy)]
pub struct GcRoot(NonNull<Cell<GcHead>>);

impl fmt::Debug for GcRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("GcRoot")
            .field(unsafe { self.0.as_ref() })
            .finish()
    }
}

pub struct GcRef<'gc, T>(&'gc GcValue<'gc, T>);

impl<'gc, T> GcRef<'gc, T> {
    pub(crate) fn new_in(gc: &'gc Gc, value: T) -> Self
    where
        T: Trace,
    {
        Self(gc.alloc(value))
    }

    pub(crate) fn as_ptr(this: &Self) -> *const T {
        &this.0.value
    }

    pub(crate) fn as_root(&self) -> GcRoot {
        GcRoot(NonNull::from(self.0).cast())
    }
}

impl<T> Clone for GcRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GcRef<'_, T> {}

impl<T> PartialEq for GcRef<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        GcRef::as_ptr(self) == GcRef::as_ptr(other)
    }
}

impl<T> Eq for GcRef<'_, T> {}

impl<T> PartialOrd for GcRef<'_, T> {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        None
    }
}

impl<T> Deref for GcRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.value
    }
}

#[derive(Clone)]
pub struct GcStr<'a>(GcRef<'a, String>);

impl<'a> GcStr<'a> {
    pub(crate) fn new_in(gc: &'a Gc, s: String) -> Self {
        Self(GcRef::new_in(gc, s))
    }

    pub(crate) fn as_root(&self) -> GcRoot {
        self.0.as_root()
    }
}

impl Copy for GcStr<'_> {}

impl PartialEq for GcStr<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0 .0.value == other.0 .0.value
    }
}

impl PartialOrd for GcStr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0 .0.value.partial_cmp(&other.0 .0.value)
    }
}

impl fmt::Debug for GcStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0 .0.value.fmt(f)
    }
}

impl fmt::Display for GcStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0 .0.value.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    struct SharedValue(Rc<Cell<u64>>);

    impl Drop for SharedValue {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    unsafe impl Trace for SharedValue {
        fn trace(&self, _f: &dyn Fn(GcRoot)) {}
    }

    #[test]
    fn dropping_the_gc_doesnt_leak() {
        let gc = Gc::default();
        for i in 0..10 {
            GcRef::new_in(&gc, 42);
            GcRef::new_in(&gc, format!("{i}"));
        }
    }

    #[test]
    fn dropping_the_gc_runs_drop() {
        let gc = Gc::default();

        let number = Rc::new(Cell::new(0));
        let gc_number = GcRef::new_in(&gc, SharedValue(Rc::clone(&number)));

        assert_eq!(gc_number.deref().0.get(), 0);

        drop(gc);

        assert_eq!(number.get(), 1);
    }

    #[test]
    fn collect_finds_unreferenced_values() {
        let gc = Gc::default();
        let numbers = [(); 10].map(|()| Rc::new(Cell::new(0)));
        let mut gc_numbers = numbers
            .iter()
            .map(|n| GcRef::new_in(&gc, SharedValue(Rc::clone(n))))
            .collect::<Vec<_>>();

        let to_remove = [7, 6, 4, 1];
        for i in to_remove {
            gc_numbers.swap_remove(i);
        }

        for gc_number in &gc_numbers {
            gc.mark_recursively(gc_number.as_root());
        }
        unsafe {
            gc.sweep();
        }

        for (i, n) in numbers.iter().enumerate() {
            if to_remove.contains(&i) {
                assert_eq!(n.get(), 1, "{i}th element was deallocated");
            }
            else {
                assert_eq!(n.get(), 0, "{i}th element was not deallocated");
            }
        }

        drop(gc);

        for n in &numbers {
            assert_eq!(n.get(), 1);
        }
    }

    #[test]
    fn collecting_everything_resets_last() {
        let gc = Gc::default();
        GcRef::new_in(&gc, ());
        GcRef::new_in(&gc, ());
        GcRef::new_in(&gc, ());
        unsafe {
            gc.sweep();
        }
        assert!(gc.last.get().is_none());
        unsafe {
            gc.sweep();
        }
    }

    #[test]
    fn can_allocate_after_collection() {
        let gc = Gc::default();
        GcRef::new_in(&gc, ());
        let value = GcRef::new_in(&gc, 27);
        GcRef::new_in(&gc, ());
        gc.mark_recursively(value.as_root());
        unsafe {
            gc.sweep();
        }
        assert_eq!(*value, 27);
        let value = GcRef::new_in(&gc, 42);
        assert_eq!(*value, 42);
    }

    #[test]
    fn can_collect_after_collection() {
        let gc = Gc::default();
        GcRef::new_in(&gc, ());
        let value = GcRef::new_in(&gc, ());
        GcRef::new_in(&gc, ());
        gc.mark_recursively(value.as_root());
        unsafe {
            gc.sweep();
        }
        let value = GcRef::new_in(&gc, 42);
        assert_eq!(*value, 42);
        unsafe {
            gc.sweep();
        }
    }

    #[test]
    fn it_runs_collection_after_collection_interval() {
        let gc = Gc::default();
        for _ in 0..COLLECTION_INTERVAL * 2 {
            GcRef::new_in(&gc, ());
            if gc.collection_necessary() {
                unsafe {
                    gc.sweep();
                }
            }
        }
        assert!(gc.last.get().is_none());
    }
}
