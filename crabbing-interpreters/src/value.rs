// TODO: apply this only to the actual comparison of `NativeFunction`
#![expect(unpredictable_function_pointer_comparisons)]

use std::cell::Cell;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::from_fn;

use variant_types::IntoVariant as _;

use crate::closure_compiler::Execute;
use crate::environment::Environment;
use crate::eval::Error;
use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::gc::Trace;
use crate::hash_map::HashMap;
use crate::interner::InternedString;
use crate::scope::Expression;
use crate::scope::Statement;
use crate::scope::Variable;
use crate::value::nanboxed::AsNanBoxed as _;

pub(crate) mod nanboxed;

pub(crate) type Cells<'a> = GcRef<'a, [Cell<GcRef<'a, Cell<nanboxed::Value<'a>>>>]>;
pub(crate) type NativeFnPtr =
    for<'a> fn(&Environment<'a>, Vec<Value<'a>>) -> Result<Value<'a>, NativeError<'a>>;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Value<'a> {
    Number(f64),
    String(GcStr<'a>),
    Bool(bool),
    Nil,
    Function(Function<'a>),
    NativeFunction(NativeFnPtr),
    Class(Class<'a>),
    Instance(Instance<'a>),
    BoundMethod(BoundMethod<'a>),
}

impl<'a> Value<'a> {
    pub fn into_nanboxed(self) -> nanboxed::Value<'a> {
        let data = match self {
            Self::Number(number) =>
                if number.is_nan() {
                    nanboxed::NaN.into_nanboxed()
                }
                else {
                    return unsafe { nanboxed::Value::from_f64_unchecked(number) };
                },
            Self::String(s) => s.into_nanboxed(),
            Self::Bool(b) => b.into_nanboxed(),
            Self::Nil => ().into_nanboxed(),
            Self::Function(function) => function.into_nanboxed(),
            Self::NativeFunction(native_function) => native_function.into_nanboxed(),
            Self::Class(class) => class.into_nanboxed(),
            Self::Instance(instance) => instance.into_nanboxed(),
            Self::BoundMethod(bmi) => bmi.into_nanboxed(),
        };
        nanboxed::Value::new(data)
    }

    pub fn typ(&self) -> &'static str {
        match self {
            Self::Number(_) => "Number",
            Self::String(_) => "String",
            Self::Bool(_) => "Bool",
            Self::Nil => "Nil",
            Self::Function(_) => "Function",
            Self::NativeFunction(_) => "NativeFunction",
            Self::Class(_) => "Class",
            Self::Instance(_) => "Instance",
            Self::BoundMethod(_) => "BoundMethod",
        }
    }

    pub(crate) fn is_truthy(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::Nil => false,
            _ => true,
        }
    }

    pub(crate) fn lox_debug(&self) -> String {
        match self {
            Self::String(s) => format!("{s:?}"),
            _ => self.to_string(),
        }
    }
}

unsafe impl Trace for Value<'_> {
    fn trace(&self) {
        match self {
            Self::Number(_) => (),
            Self::String(s) => s.trace(),
            Self::Bool(_) => (),
            Self::Nil => (),
            Self::Function(function) => function.trace(),
            Self::NativeFunction(_) => (),
            Self::Class(class) => class.trace(),
            Self::Instance(instance) => instance.trace(),
            Self::BoundMethod(bound_method) => bound_method.trace(),
        }
    }
}

impl Display for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(x) => write!(f, "{x}"),
            Self::String(s) => write!(f, "{s}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Nil => write!(f, "nil"),
            Self::Function(func) => write!(f, "{func:?}"),
            Self::NativeFunction(_) => write!(f, "<native fn>"),
            Self::Class(class) => write!(f, "{class:?}"),
            Self::Instance(instance) => write!(f, "{instance:?}"),
            Self::BoundMethod(bound_method) => write!(f, "{bound_method:?}"),
        }
    }
}

pub type Function<'a> = GcRef<'a, FunctionInner<'a>>;

pub struct FunctionInner<'a> {
    pub(crate) name: &'a str,
    pub(crate) parameters: &'a [Variable<'a>],
    pub(crate) code: &'a [Statement<'a>],
    pub(crate) cells: Cells<'a>,
    pub(crate) compiled_body: &'a Execute<'a>,
    pub(crate) code_ptr: usize,
}

unsafe impl Trace for FunctionInner<'_> {
    fn trace(&self) {
        self.cells.trace();
    }
}

impl Debug for Function<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<function {} at {:p}>", self.name, Self::as_ptr(self))
    }
}

pub type Class<'a> = GcRef<'a, ClassInner<'a>>;

pub struct ClassInner<'a> {
    pub(crate) name: &'a str,
    pub(crate) base: Option<Class<'a>>,
    pub(crate) methods: HashMap<InternedString, nanboxed::Value<'a>>,
}

impl<'a> ClassInner<'a> {
    fn mro(&self) -> impl Iterator<Item = &Self> {
        let mut class = Some(self);
        from_fn(move || {
            let class = &mut class;
            std::mem::replace(class, class.and_then(|class| class.base.as_deref()))
        })
    }

    pub(crate) fn lookup_method(&self, name: InternedString) -> Option<nanboxed::Value<'a>> {
        self.mro()
            .find_map(|class| class.methods.get(&name))
            .copied()
    }
}

unsafe impl Trace for ClassInner<'_> {
    fn trace(&self) {
        if let Some(base) = self.base {
            base.trace();
        }
        for method in self.methods.values() {
            method.trace();
        }
    }
}

impl Debug for Class<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<class {} at {:p}>", self.name, Self::as_ptr(self))
    }
}

pub mod instance {
    use std::cell::RefCell;

    use crate::gc::Trace;
    use crate::hash_map::HashMap;
    use crate::interner::InternedString;
    use crate::value::nanboxed;
    use crate::value::Class;

    pub struct InstanceInner<'a> {
        pub(crate) class: Class<'a>,
        attributes: RefCell<HashMap<InternedString, nanboxed::Value<'a>>>,
    }

    impl<'a> InstanceInner<'a> {
        pub(crate) fn new(class: Class<'a>) -> Self {
            Self { class, attributes: RefCell::default() }
        }

        pub(crate) fn setattr(&self, name: InternedString, value: nanboxed::Value<'a>) {
            // SAFETY: there are no references to/into `self.attributes` because `getattr`
            // returns a copied value, so we can mutate here
            unsafe { &mut *self.attributes.as_ptr() }.insert(name, value);
        }

        pub(crate) fn getattr(&self, name: InternedString) -> Option<nanboxed::Value<'a>> {
            // SAFETY: there are no mutable references to/into `self.attributes`, so we can create
            // a shared reference here
            unsafe { &*self.attributes.as_ptr() }.get(&name).copied()
        }
    }

    unsafe impl Trace for InstanceInner<'_> {
        fn trace(&self) {
            self.class.trace();
            self.attributes
                .borrow()
                .values()
                .for_each(|value| value.trace());
        }
    }
}

pub type Instance<'a> = GcRef<'a, instance::InstanceInner<'a>>;

impl Debug for Instance<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{} instance at {:p}>",
            self.class.name,
            Self::as_ptr(self),
        )
    }
}

pub type BoundMethod<'a> = GcRef<'a, BoundMethodInner<'a>>;

pub struct BoundMethodInner<'a> {
    pub(crate) method: Function<'a>,
    pub(crate) instance: Instance<'a>,
}

unsafe impl Trace for BoundMethodInner<'_> {
    fn trace(&self) {
        self.method.trace();
        self.instance.trace();
    }
}

impl Debug for BoundMethod<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<bound method {name} of {instance:?}>",
            name = self.method.name,
            instance = self.instance,
        )
    }
}

pub enum NativeError<'a> {
    Error(Error<'a>),
    ArityMismatch {
        expected: usize,
    },
    TypeError {
        name: String,
        expected: String,
        tys: String,
    },
    IoError {
        name: String,
        error: std::io::Error,
        filename: String,
    },
}

impl<'a> NativeError<'a> {
    pub(crate) fn at_expr(self, callee: Value<'a>, expr: &Expression<'a>) -> Error<'a> where {
        match self {
            Self::Error(err) => err,
            Self::ArityMismatch { expected } => Error::ArityMismatch {
                callee,
                expected,
                at: expr.into_variant(),
            },
            Self::TypeError { name, expected, tys } => Error::NativeFnCallArgTypeMismatch {
                name,
                at: expr.into_variant(),
                expected,
                tys,
            },
            Self::IoError { name, error, filename } => Error::NativeFnCallIoError {
                name,
                at: expr.into_variant(),
                error,
                filename,
            },
        }
    }
}
