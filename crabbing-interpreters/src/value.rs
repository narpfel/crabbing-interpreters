use std::cell::Cell;
use std::cell::RefCell;
use std::fmt::Debug;
use std::fmt::Display;

use rustc_hash::FxHashMap as HashMap;

use crate::closure_compiler::Execute;
use crate::eval::Error;
use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::gc::Trace;
use crate::interner::InternedString;
use crate::scope::Statement;
use crate::scope::Variable;

pub(crate) type Cells<'a> = GcRef<'a, [Cell<GcRef<'a, Cell<Value<'a>>>>]>;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Value<'a> {
    Number(f64),
    String(GcStr<'a>),
    Bool(bool),
    Nil,
    Function(Function<'a>),
    NativeFunction(for<'b> fn(Vec<Value<'b>>) -> Result<Value<'b>, NativeError<'b>>),
    Class(Class<'a>),
    Instance(Instance<'a>),
    BoundMethod(BoundMethod<'a>),
}

impl Value<'_> {
    pub fn typ(&self) -> &'static str {
        match self {
            Value::Number(_) => "Number",
            Value::String(_) => "String",
            Value::Bool(_) => "Bool",
            Value::Nil => "Nil",
            Value::Function(_) => "Function",
            Value::NativeFunction(_) => "NativeFunction",
            Value::Class(_) => "Class",
            Value::Instance(_) => "Instance",
            Value::BoundMethod(_) => "BoundMethod",
        }
    }

    pub(crate) fn is_truthy(&self) -> bool {
        use Value::*;
        match self {
            Bool(b) => *b,
            Nil => false,
            _ => true,
        }
    }

    pub(crate) fn lox_debug(&self) -> String {
        match self {
            Value::String(s) => format!("{s:?}"),
            _ => self.to_string(),
        }
    }
}

unsafe impl Trace for Value<'_> {
    fn trace(&self) {
        match self {
            Value::Number(_) => (),
            Value::String(s) => s.trace(),
            Value::Bool(_) => (),
            Value::Nil => (),
            Value::Function(function) => function.trace(),
            Value::NativeFunction(_) => (),
            Value::Class(class) => class.trace(),
            Value::Instance(instance) => instance.trace(),
            Value::BoundMethod(bound_method) => bound_method.trace(),
        }
    }
}

impl Display for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(x) => write!(f, "{x}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Nil => write!(f, "nil"),
            Value::Function(func) => write!(f, "{func:?}"),
            Value::NativeFunction(_) => write!(f, "<native fn>"),
            Value::Class(class) => write!(f, "{class:?}"),
            Value::Instance(instance) => write!(f, "{instance:?}"),
            Value::BoundMethod(bound_method) => write!(f, "{bound_method:?}"),
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
    pub(crate) methods: HashMap<InternedString, Value<'a>>,
}

impl<'a> ClassInner<'a> {
    fn mro(&self) -> impl Iterator<Item = &Self> {
        itertools::unfold(Some(self), |class| {
            std::mem::replace(class, class.and_then(|class| class.base.as_deref()))
        })
    }

    pub(crate) fn lookup_method(&self, name: InternedString) -> Option<Value<'a>> {
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

pub type Instance<'a> = GcRef<'a, InstanceInner<'a>>;

pub struct InstanceInner<'a> {
    pub(crate) class: Class<'a>,
    pub(crate) attributes: RefCell<HashMap<InternedString, Value<'a>>>,
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
    ArityMismatch { expected: usize },
}
