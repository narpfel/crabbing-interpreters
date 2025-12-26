use std::alloc::Layout;
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;
use std::ptr::NonNull;
use std::ptr::Pointee;

mod heads;

#[cfg(not(miri))]
const COLLECTION_INTERVAL: usize = 100_000;

#[cfg(miri)]
const COLLECTION_INTERVAL: usize = 1;

type BoxedValue<'a, T> = Box<GcValue<'a, T>>;

enum Action<F> {
    Keep,
    Drop,
    Immortalise(F),
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
        let state = head.get().state();
        match state {
            State::Unvisited => {
                head.set(head.get().with_state(State::Marked));
                self.value().value.trace();
            }
            State::Marked => (),
            State::Immortal => (),
        }
    }
}

unsafe impl<T> Trace for GcThin<'_, T>
where
    T: Trace + ?Sized,
    <T as Pointee>::Metadata: IsUsize,
{
    fn trace(&self) {
        self.as_gc_ref().trace();
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
    allocated_heads: heads::Heads,
    allocation_count: Cell<usize>,
    // TODO: only the first 128 entries are used, but reducing the size introduces an unnecessary
    // bounds check in `GcStr::new_in`
    small_string_cache: [Cell<Option<NonNull<()>>>; 256] = [const { Cell::new(None) }; _],
    immortals: heads::Heads,
}

impl Gc {
    fn alloc<'a, T>(&'a self, value: T) -> NonNull<GcValue<'a, T>>
    where
        T: Trace,
    {
        let gc_value = BoxedValue::<'a, T>::into_raw(BoxedValue::<'a, T>::new(GcValue {
            head: Cell::new(GcHead {
                next: None,
                length_and_state: LengthAndState::new(0),
                drop: |p, _| drop(unsafe { BoxedValue::<'a, T>::from_raw(p.cast().as_ptr()) }),
            }),
            _gc: PhantomData,
            value,
        }));
        let gc_value = NonNull::new(gc_value).unwrap();

        unsafe {
            self.adopt(gc_value);
        }
        gc_value
    }

    unsafe fn adopt<'a, T>(&'a self, value: NonNull<GcValue<'a, T>>)
    where
        T: ?Sized,
    {
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.allocated_heads.push(value.cast());
    }

    fn immortalise<'a, T>(&'a self, value: &GcValue<'a, T>)
    where
        T: ?Sized,
    {
        let head = &value.head;
        head.set(head.get().with_state(State::Immortal));
    }

    pub(crate) unsafe fn sweep(&self) {
        self.allocation_count.set(0);
        self.allocated_heads.iter_with(|state| match state {
            State::Unvisited => Action::Drop,
            State::Marked => Action::Keep,
            State::Immortal => Action::Immortalise(|head_ptr| self.immortals.push(head_ptr)),
        });
    }

    pub(crate) fn collection_necessary(&self) -> bool {
        self.allocation_count.get() >= COLLECTION_INTERVAL
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
    fn ptr_from_raw_parts(memory: NonNull<u8>, length: usize) -> NonNull<Self> {
        NonNull::from_raw_parts(memory, length)
    }
}

#[derive(Debug, Clone, Copy)]
struct LengthAndState(usize);

impl LengthAndState {
    fn new(length: usize) -> Self {
        Self(length.shl_exact(2).unwrap() | usize::from(State::Unvisited))
    }

    fn length(self) -> usize {
        self.0 >> 2
    }

    fn state(self) -> State {
        match self.0 & 0b11 {
            0 => State::Unvisited,
            1 => State::Marked,
            2 => State::Immortal,
            3 => State::Immortal,
            _ => unreachable!(),
        }
    }

    fn with_state(&self, state: State) -> Self {
        Self((self.0 & !0b11) | usize::from(state))
    }
}

#[derive(Debug, Clone, Copy)]
struct GcHead {
    next: Option<NonNull<Cell<GcHead>>>,
    length_and_state: LengthAndState,
    /// SAFETY: This must only be called once, and it must be a deallocation function matching the
    /// allocation function used to allocate this gc value.
    drop: unsafe fn(NonNull<()>, usize),
}

impl GcHead {
    fn length(&self) -> usize {
        self.length_and_state.length()
    }

    fn state(&self) -> State {
        self.length_and_state.state()
    }

    fn with_state(&self, state: State) -> Self {
        Self {
            length_and_state: self.length_and_state.with_state(state),
            ..*self
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum State {
    Unvisited,
    Marked,
    Immortal,
}

impl From<State> for usize {
    fn from(value: State) -> Self {
        #[expect(
            clippy::as_conversions,
            reason = "converting an enum to its underlying type is okay"
        )]
        Self::from(value as u8)
    }
}

pub struct GcRef<'gc, T>(NonNull<GcValue<'gc, T>>, PhantomData<&'gc GcValue<'gc, T>>)
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

    pub(crate) unsafe fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr.cast(), PhantomData)
    }

    pub(crate) fn as_inner(this: Self) -> NonNull<()> {
        this.0.cast()
    }
}

impl<'gc, T> GcRef<'gc, T>
where
    T: ?Sized,
{
    fn value(&self) -> &'gc GcValue<'gc, T> {
        // SAFETY: This is safe for uncollected `GcRef`s. Callers of `Gc::sweep` must ensure that
        // no reachable `GcRef`s are collected.
        unsafe { self.0.as_ref() }
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
            let nonnull_gc_value = Self::value_ptr_from_raw_parts(memory, length);
            let gc_value = nonnull_gc_value.as_ptr();

            ptr::addr_of_mut!((*gc_value).head).write(Cell::new(GcHead {
                next: None,
                length_and_state: LengthAndState::new(length),
                drop: |memory, length| {
                    let gc_value = Self::value_ptr_from_raw_parts(memory.cast(), length);
                    ptr::drop_in_place(gc_value.as_ptr());
                    std::alloc::dealloc(memory.cast().as_ptr(), Self::compute_layout(length));
                },
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
            gc.adopt(nonnull_gc_value);
            Self(nonnull_gc_value, PhantomData)
        }
    }

    fn compute_layout(length: usize) -> Layout {
        Layout::new::<GcValue<'gc, ()>>()
            .extend(Layout::array::<T>(length).unwrap())
            .unwrap()
            .0
            .pad_to_align()
    }

    fn value_ptr_from_raw_parts(memory: NonNull<u8>, length: usize) -> NonNull<GcValue<'gc, [T]>> {
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

trait IsUsize {}

impl IsUsize for usize {}

struct GcThin<'a, T: ?Sized>(NonNull<()>, PhantomData<GcRef<'a, T>>);

impl<T> Clone for GcThin<'_, T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GcThin<'_, T> where T: ?Sized {}

impl<'a, T> GcThin<'a, T>
where
    T: ?Sized,
    <T as Pointee>::Metadata: IsUsize,
{
    fn from_ref(gc_ref: GcRef<'a, T>) -> Self {
        let (ptr, _metadata) = gc_ref.0.to_raw_parts();
        Self(ptr.cast(), PhantomData)
    }

    fn as_inner(this: Self) -> NonNull<()> {
        this.0
    }

    unsafe fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(ptr, PhantomData)
    }

    fn as_gc_ref(self) -> GcRef<'a, T> {
        let head_ptr: NonNull<Cell<GcHead>> = self.0.cast();
        unsafe {
            let length = head_ptr.as_ref().get().length();
            // SAFETY: this pointer cast is safe because we restrict `<T as Pointee>::Metadata` to
            // be `usize`
            let ptr = NonNull::from_raw_parts(self.0, *ptr::from_ref(&length).cast());
            GcRef(ptr, PhantomData)
        }
    }
}

#[derive(Clone, Copy)]
pub struct GcStr<'a>(GcThin<'a, [u8]>);

impl<'a> GcStr<'a> {
    pub(crate) fn new_in(gc: &'a Gc, s: &str) -> Self {
        match s.len() {
            1 => {
                let cached_string = &gc.small_string_cache[usize::from(s.as_bytes()[0])];
                match cached_string.get() {
                    Some(string) => unsafe { Self::from_ptr(string) },
                    None => {
                        let string = Self(GcThin::from_ref(GcRef::from_iter_in(gc, s.bytes())));
                        gc.immortalise(string.0.as_gc_ref().value());
                        cached_string.set(Some(GcStr::as_inner(string)));
                        string
                    }
                }
            }
            _ => Self(GcThin::from_ref(GcRef::from_iter_in(gc, s.bytes()))),
        }
    }

    pub(crate) unsafe fn from_ptr(ptr: NonNull<()>) -> Self {
        Self(GcThin::from_ptr(ptr))
    }

    pub(crate) fn as_inner(this: Self) -> NonNull<()> {
        GcThin::as_inner(this.0)
    }

    fn str(&self) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.0.as_gc_ref().value().value) }
    }
}

impl PartialEq for GcStr<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.str() == other.str()
    }
}

impl PartialOrd for GcStr<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.str().partial_cmp(other.str())
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
        assert!(gc.allocated_heads.is_empty());
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
        assert!(gc.allocated_heads.is_empty());
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
