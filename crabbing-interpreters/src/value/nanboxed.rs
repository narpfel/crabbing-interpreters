use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ptr::NonNull;

use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::gc::Trace;
use crate::value::BoundMethod;
use crate::value::Class;
use crate::value::Function;
use crate::value::Instance;
use crate::value::NativeFnPtr;
use crate::value::Value as Unboxed;

const NAN_PREFIX_LENGTH: u8 = 1 + 11;
/// 47 bits for the address and one extra for the sign bit
const NANBOX_TAG_OFFSET: u64 = 48;
const NAN_BITS: u64 = !((1 << (64 - NAN_PREFIX_LENGTH)) - 1);

fn extend_leftmost_pointer_bit(n: u64) -> u64 {
    ((n.cast_signed() << (64 - NANBOX_TAG_OFFSET)) >> (64 - NANBOX_TAG_OFFSET)).cast_unsigned()
}

#[derive(Clone, Copy)]
union F64WithProvenance {
    float: f64,
    pointer: *const (),
}

impl F64WithProvenance {
    fn pointer(self) -> *const () {
        const {
            assert!(
                size_of::<f64>() == size_of::<usize>(),
                "nan-boxing only works on 64-bit systems",
            )
        }
        unsafe { self.pointer }
    }

    fn data(self) -> f64 {
        unsafe { self.float }
    }

    fn addr(self) -> usize {
        self.pointer().addr()
    }

    fn with_addr(self, addr: usize) -> *const () {
        self.pointer().with_addr(addr)
    }
}

impl PartialEq for F64WithProvenance {
    fn eq(&self, other: &Self) -> bool {
        self.addr() == other.addr()
    }
}

impl Eq for F64WithProvenance {}

impl From<*const ()> for F64WithProvenance {
    fn from(pointer: *const ()) -> Self {
        Self { pointer }
    }
}

impl fmt::Debug for F64WithProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.pointer())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Value<'a> {
    data: F64WithProvenance,
    _value: PhantomData<Unboxed<'a>>,
}

impl<'a> Value<'a> {
    pub(super) fn new(data: *const ()) -> Self {
        Self {
            data: F64WithProvenance { pointer: data },
            _value: PhantomData,
        }
    }

    pub(crate) unsafe fn from_f64_unchecked(number: f64) -> Self {
        debug_assert!(!number.is_nan());
        Self {
            data: F64WithProvenance { float: number },
            _value: PhantomData,
        }
    }

    pub(crate) fn data(self) -> f64 {
        self.data.data()
    }

    #[inline(always)]
    pub fn parse(self) -> Unboxed<'a> {
        let data = u64::try_from(self.data.addr()).unwrap();

        if (data & NAN_BITS) != NAN_BITS || data == NAN_BITS {
            Unboxed::Number(f64::from_bits(data))
        }
        else {
            let tag = (data & !NAN_BITS) >> NANBOX_TAG_OFFSET;
            let tag = unsafe { NanBoxTag::try_from(tag).unwrap_unchecked() };
            let pointer = extend_leftmost_pointer_bit(data);
            let pointer = self.data.with_addr(usize::try_from(pointer).unwrap());
            // SAFETY: `pointer` is not null here because `data != NAN_BITS`
            let pointer = unsafe { NonNull::new_unchecked(pointer.cast_mut()) };
            match tag {
                NanBoxTag::Nil => Unboxed::Nil,
                NanBoxTag::False => Unboxed::Bool(false),
                NanBoxTag::True => Unboxed::Bool(true),
                NanBoxTag::String => Unboxed::String(unsafe { GcStr::from_ptr(pointer) }),
                NanBoxTag::Function => Unboxed::Function(unsafe { GcRef::from_ptr(pointer) }),
                NanBoxTag::NativeFunction => Unboxed::NativeFunction(unsafe {
                    transmute::<NonNull<()>, NativeFnPtr>(pointer)
                }),
                NanBoxTag::Class => Unboxed::Class(unsafe { GcRef::from_ptr(pointer) }),
                NanBoxTag::Instance => Unboxed::Instance(unsafe { GcRef::from_ptr(pointer) }),
                NanBoxTag::BoundMethod => Unboxed::BoundMethod(unsafe { GcRef::from_ptr(pointer) }),
                NanBoxTag::NaN => Unboxed::Number(f64::from_bits(data)),
            }
        }
    }

    pub fn is_truthy(self) -> bool {
        self.parse().is_truthy()
    }

    pub(crate) fn eq_nanboxed(self, other: Self) -> bool {
        self.data == other.data
    }
}

unsafe impl Trace for Value<'_> {
    fn trace(&self) {
        self.parse().trace();
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NaN;

#[repr(u8)]
enum NanBoxTag {
    Nil = 1,
    False,
    True,
    String,
    Function,
    NativeFunction,
    Class,
    Instance,
    BoundMethod,
    NaN,
}

impl NanBoxTag {
    fn tag(self) -> u64 {
        #[expect(
            clippy::as_conversions,
            reason = "casting a `repr(u8)` enum to `u64` is always okay"
        )]
        ((self as u64) << NANBOX_TAG_OFFSET)
    }
}

impl TryFrom<u64> for NanBoxTag {
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Nil,
            2 => Self::False,
            3 => Self::True,
            4 => Self::String,
            5 => Self::Function,
            6 => Self::NativeFunction,
            7 => Self::Class,
            8 => Self::Instance,
            9 => Self::BoundMethod,
            10 => Self::NaN,
            _ => Err(value)?,
        })
    }
}

impl From<()> for NanBoxTag {
    fn from(_value: ()) -> Self {
        NanBoxTag::Nil
    }
}

impl From<NaN> for NanBoxTag {
    fn from(_value: NaN) -> Self {
        NanBoxTag::NaN
    }
}

impl From<bool> for NanBoxTag {
    fn from(value: bool) -> Self {
        match value {
            true => NanBoxTag::True,
            false => NanBoxTag::False,
        }
    }
}

impl From<GcStr<'_>> for NanBoxTag {
    fn from(_value: GcStr) -> Self {
        NanBoxTag::String
    }
}

impl From<Function<'_>> for NanBoxTag {
    fn from(_value: Function<'_>) -> Self {
        NanBoxTag::Function
    }
}

impl From<NativeFnPtr> for NanBoxTag {
    fn from(_value: NativeFnPtr) -> Self {
        NanBoxTag::NativeFunction
    }
}

impl From<Class<'_>> for NanBoxTag {
    fn from(_value: Class<'_>) -> Self {
        NanBoxTag::Class
    }
}

impl From<Instance<'_>> for NanBoxTag {
    fn from(_value: Instance<'_>) -> Self {
        NanBoxTag::Instance
    }
}

impl From<BoundMethod<'_>> for NanBoxTag {
    fn from(_value: BoundMethod<'_>) -> Self {
        NanBoxTag::BoundMethod
    }
}

pub(super) trait AsNanBoxed {
    fn into_nanboxed(self) -> *const ();
}

impl<T> AsNanBoxed for T
where
    NanBoxTag: From<T>,
    T: Copy + NanBoxPayload,
{
    fn into_nanboxed(self) -> *const () {
        let pointer = self.payload();
        let addr = u64::try_from(pointer.addr().get()).unwrap();
        let addr = addr & (2_u64.pow(u32::try_from(NANBOX_TAG_OFFSET).unwrap()) - 1);
        let data = NAN_BITS | NanBoxTag::from(self).tag() | addr;
        pointer.as_ptr().with_addr(usize::try_from(data).unwrap())
    }
}

trait NanBoxPayload {
    fn payload(self) -> NonNull<()>;
}

impl NanBoxPayload for () {
    fn payload(self) -> NonNull<()> {
        NonNull::dangling()
    }
}

impl NanBoxPayload for NaN {
    fn payload(self) -> NonNull<()> {
        NonNull::dangling()
    }
}

impl NanBoxPayload for bool {
    fn payload(self) -> NonNull<()> {
        NonNull::dangling()
    }
}

impl NanBoxPayload for GcStr<'_> {
    fn payload(self) -> NonNull<()> {
        GcStr::as_inner(self)
    }
}

impl<T> NanBoxPayload for GcRef<'_, T> {
    fn payload(self) -> NonNull<()> {
        GcRef::as_inner(self)
    }
}

impl NanBoxPayload for NativeFnPtr {
    fn payload(self) -> NonNull<()> {
        #[expect(
            clippy::as_conversions,
            reason = "`as` is the only possible way to cast a `fn` pointer to a `*` pointer that preserves provenance"
        )]
        NonNull::new(self as *mut ()).unwrap()
    }
}
