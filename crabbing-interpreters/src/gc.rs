use std::alloc::Layout;
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;
use std::ptr::NonNull;

#[cfg(not(miri))]
const COLLECTION_INTERVAL: usize = 100_000;

#[cfg(miri)]
const COLLECTION_INTERVAL: usize = 1;

type BoxedValue<'a, T> = Box<GcValue<'a, T>>;

enum Action {
    Keep,
    Drop,
}

/// # Safety
///
/// Implementors must make sure to trace into all children that contain [`GcRef`]s. Missed children
/// will be collected during the next sweep, leading to dangling references.
pub(crate) unsafe trait Trace {
    fn trace(&self);
}

unsafe impl Trace for () {
    fn trace(&self) {}
}

unsafe impl Trace for u8 {
    fn trace(&self) {}
}

unsafe impl Trace for u32 {
    fn trace(&self) {}
}

unsafe impl Trace for u64 {
    fn trace(&self) {}
}

unsafe impl Trace for usize {
    fn trace(&self) {}
}

unsafe impl Trace for String {
    fn trace(&self) {}
}

unsafe impl Trace for str {
    fn trace(&self) {}
}

unsafe impl<T> Trace for Cell<T>
where
    T: Copy + Trace,
{
    fn trace(&self) {
        self.get().trace();
    }
}

unsafe impl<T> Trace for GcRef<'_, T>
where
    T: Trace + ?Sized,
{
    fn trace(&self) {
        let head = &self.value().head;
        let state = head.get().state;
        match state {
            State::Unvisited => {
                head.set(GcHead {
                    state: State::VisitingChildren,
                    ..head.get()
                });
                self.value().value.trace();
                head.set(GcHead { state: State::Done, ..head.get() });
            }
            State::VisitingChildren => (),
            State::Done => (),
        }
    }
}

unsafe impl Trace for GcStr<'_> {
    fn trace(&self) {
        self.0.trace();
    }
}

unsafe impl<T, U, V> Trace for (T, U, V)
where
    T: Trace,
    U: Trace,
    V: Trace,
{
    fn trace(&self) {
        self.0.trace();
        self.1.trace();
        self.2.trace();
    }
}

unsafe impl<T> Trace for [T]
where
    T: Trace,
{
    fn trace(&self) {
        for value in self {
            value.trace();
        }
    }
}

#[derive(Default)]
pub struct Gc {
    last: Cell<Option<NonNull<Cell<GcHead>>>>,
    allocation_count: Cell<usize>,
}

impl Gc {
    fn alloc<'a, T>(&'a self, value: T) -> *const GcValue<'a, T>
    where
        T: Trace,
    {
        let gc_value = BoxedValue::<'a, T>::into_raw(BoxedValue::<'a, T>::new(GcValue {
            head: Cell::new(GcHead {
                prev: None,
                next: None,
                length: 0,
                drop: |p, _| drop(unsafe { BoxedValue::<'a, T>::from_raw(p.cast().as_ptr()) }),
                state: State::Unvisited,
            }),
            _gc: PhantomData,
            value,
        }));

        unsafe {
            self.adopt(gc_value);
        }
        gc_value
    }

    unsafe fn adopt<'a, T>(&'a self, value: *mut GcValue<'a, T>)
    where
        T: ?Sized,
    {
        self.allocation_count.set(self.allocation_count.get() + 1);
        let last = self.last.get();
        unsafe {
            let value_ref = &*value;
            let head_ptr = NonNull::new(ptr::addr_of_mut!((*value).head));
            let mut head = value_ref.head.get();
            debug_assert!(head.prev.is_none());
            debug_assert!(head.next.is_none());
            head.prev = last;
            value_ref.head.set(head);

            if let Some(last) = last {
                let last = last.as_ref();
                last.set(GcHead { next: head_ptr, ..last.get() });
            }

            self.last.set(head_ptr);
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

                    if let Some(next) = next {
                        let next = unsafe { next.as_ref() };
                        next.set(GcHead { prev, ..next.get() });
                    }
                    else {
                        // `head` is last
                        self.last.set(prev);
                    }

                    (head.drop)(head_ptr.cast(), head.length)
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
struct GcValue<'gc, T>
where
    T: ?Sized,
{
    head: Cell<GcHead>,
    _gc: PhantomData<&'gc Gc>,
    value: T,
}

impl<T> GcValue<'_, [T]> {
    fn ptr_from_raw_parts(memory: NonNull<u8>, length: usize) -> *mut Self {
        NonNull::from_raw_parts(memory.cast(), length).as_ptr()
    }
}

#[derive(Debug, Clone, Copy)]
struct GcHead {
    prev: Option<NonNull<Cell<GcHead>>>,
    next: Option<NonNull<Cell<GcHead>>>,
    length: usize,
    drop: fn(NonNull<()>, usize),
    state: State,
}

#[derive(Debug, Clone, Copy)]
enum State {
    Unvisited,
    VisitingChildren,
    Done,
}

pub struct GcRef<'gc, T>(*const GcValue<'gc, T>, PhantomData<&'gc GcValue<'gc, T>>)
where
    T: ?Sized;

impl<'gc, T> GcRef<'gc, T> {
    pub(crate) fn new_in(gc: &'gc Gc, value: T) -> Self
    where
        T: Trace,
    {
        Self(gc.alloc(value), PhantomData)
    }

    pub(crate) fn as_ptr(this: &Self) -> *const T {
        &GcRef::value(this).value
    }
}

impl<'gc, T> GcRef<'gc, T>
where
    T: ?Sized,
{
    fn value(&self) -> &'gc GcValue<'gc, T> {
        // SAFETY: This is safe for uncollected `GcRef`s. Callers of `Gc::sweep` must ensure that
        // no reachable `GcRef`s are collected.
        unsafe { &*self.0 }
    }
}

impl<'gc, T> GcRef<'gc, [T]> {
    pub(crate) fn from_iter_in(gc: &'gc Gc, mut iterator: impl ExactSizeIterator<Item = T>) -> Self
    where
        T: Trace,
    {
        let length = iterator.len();

        unsafe {
            let layout = Self::compute_layout(length);
            let memory = NonNull::new(std::alloc::alloc(layout))
                .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
            let gc_value = Self::value_ptr_from_raw_parts(memory, length);

            ptr::addr_of_mut!((*gc_value).head).write(Cell::new(GcHead {
                prev: None,
                next: None,
                length,
                drop: |memory, length| {
                    let gc_value = Self::value_ptr_from_raw_parts(memory.cast(), length);
                    ptr::drop_in_place(gc_value);
                    std::alloc::dealloc(memory.cast().as_ptr(), Self::compute_layout(length));
                },
                state: State::Unvisited,
            }));
            ptr::addr_of_mut!((*gc_value)._gc).write(PhantomData);

            // FIXME: a panic here until the adoption leaks `gc_value`
            // TODO: add a test with a broken `ExactSizeIterator` that reports an incorrect size
            let values = ptr::addr_of_mut!((*gc_value).value);
            for i in 0..length {
                values
                    .get_unchecked_mut(i)
                    .write(iterator.next().expect("iterator was long enough"));
            }
            assert!(iterator.next().is_none(), "iterator was too long");

            // adopt after initialising the slice to prevent dropping uninitialised values when the
            // loop above panics
            gc.adopt(gc_value);
            Self(gc_value, PhantomData)
        }
    }

    fn compute_layout(length: usize) -> Layout {
        Layout::new::<GcValue<'gc, ()>>()
            .extend(Layout::array::<T>(length).unwrap())
            .unwrap()
            .0
            .pad_to_align()
    }

    fn value_ptr_from_raw_parts(memory: NonNull<u8>, length: usize) -> *mut GcValue<'gc, [T]> {
        GcValue::ptr_from_raw_parts(memory, length)
    }
}

impl<T> Clone for GcRef<'_, T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GcRef<'_, T> where T: ?Sized {}

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

impl<T> Deref for GcRef<'_, T>
where
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value().value
    }
}

impl<'a, T> IntoIterator for GcRef<'a, T>
where
    T: ?Sized,
    &'a T: IntoIterator,
{
    type IntoIter = <&'a T as IntoIterator>::IntoIter;
    type Item = <&'a T as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.value().value.into_iter()
    }
}

impl<T> fmt::Debug for GcRef<'_, T>
where
    T: ?Sized + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

#[derive(Clone, Copy)]
pub struct GcStr<'a>(GcRef<'a, [u8]>);

impl<'a> GcStr<'a> {
    pub(crate) fn new_in(gc: &'a Gc, s: &str) -> Self {
        Self(GcRef::from_iter_in(gc, s.bytes()))
    }

    fn str(&self) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.0.value().value) }
    }
}

impl PartialEq for GcStr<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.value().value == other.0.value().value
    }
}

impl PartialOrd for GcStr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.value().value.partial_cmp(&other.0.value().value)
    }
}

impl fmt::Debug for GcStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.str().fmt(f)
    }
}

impl fmt::Display for GcStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.str().fmt(f)
    }
}

impl Deref for GcStr<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.str()
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
        fn trace(&self) {}
    }

    unsafe impl<T> Trace for &T
    where
        T: Trace,
    {
        fn trace(&self) {
            (*self).trace();
        }
    }

    unsafe impl<T> Trace for Option<T>
    where
        T: Trace,
    {
        fn trace(&self) {
            if let Some(value) = self {
                value.trace();
            }
        }
    }

    #[test]
    fn dropping_the_gc_doesnt_leak() {
        let gc = Gc::default();
        for i in 0..10 {
            GcRef::new_in(&gc, 42_u64);
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
            gc_number.trace();
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
        let value = GcRef::new_in(&gc, 27_u64);
        GcRef::new_in(&gc, ());
        value.trace();
        unsafe {
            gc.sweep();
        }
        assert_eq!(*value, 27);
        let value = GcRef::new_in(&gc, 42_u64);
        assert_eq!(*value, 42);
    }

    #[test]
    fn can_collect_after_collection() {
        let gc = Gc::default();
        GcRef::new_in(&gc, ());
        let value = GcRef::new_in(&gc, ());
        GcRef::new_in(&gc, ());
        value.trace();
        unsafe {
            gc.sweep();
        }
        let value = GcRef::new_in(&gc, 42_u64);
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

    #[test]
    fn circular_references_in_cells() {
        #[derive(Clone, Copy)]
        struct Value<'a> {
            n: u64,
            maybe_value: Option<&'a Cell<GcRef<'a, Cell<Value<'a>>>>>,
        }
        unsafe impl Trace for Value<'_> {
            fn trace(&self) {
                self.n.trace();
                self.maybe_value.trace();
            }
        }
        let gc = &Gc::default();
        let initial = Cell::new(GcRef::new_in(
            gc,
            Cell::new(Value { n: 17, maybe_value: None }),
        ));
        let x = Cell::new(GcRef::new_in(
            gc,
            Cell::new(Value { n: 42, maybe_value: Some(&initial) }),
        ));
        let y = Cell::new(GcRef::new_in(
            gc,
            Cell::new(Value { n: 27, maybe_value: Some(&x) }),
        ));
        x.get().set(Value {
            n: x.get().get().n,
            maybe_value: Some(&y),
        });
        x.trace();
        unsafe {
            gc.sweep();
        }
        assert_eq!(x.get().get().n, 42);
        assert_eq!(x.get().get().maybe_value.unwrap().get().get().n, 27);
        assert_eq!(y.get().get().n, 27);
        assert_eq!(y.get().get().maybe_value.unwrap().get().get().n, 42);
    }
}
