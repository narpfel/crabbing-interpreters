use core::slice;
use std::cell::Cell;
use std::cell::RefCell;
use std::fmt::Debug;
use std::fmt::Display;
use std::iter::zip;
use std::iter::Skip;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;

use ariadne::Color::Blue;
use ariadne::Color::Green;
use ariadne::Color::Magenta;
use ariadne::Color::Red;
use crabbing_interpreters_derive_report::Report;
use rustc_hash::FxHashMap as HashMap;
use variant_types::IntoVariant;

use crate::clone_from_cell::CloneInCellSafe;
use crate::clone_from_cell::GetClone;
use crate::closure_compiler::Execute;
use crate::interner::interned;
use crate::interner::InternedString;
use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::LiteralKind;
use crate::parse::Name;
use crate::parse::UnaryOp;
use crate::parse::UnaryOpKind;
use crate::rc_str::RcStr;
use crate::rc_value::RcValue;
use crate::scope::AssignmentTarget;
use crate::scope::AssignmentTargetTypes;
use crate::scope::Expression;
use crate::scope::ExpressionTypes;
use crate::scope::Slot;
use crate::scope::Statement;
use crate::scope::Target;
use crate::scope::Variable;
use crate::Sliced;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value<'a> {
    Number(f64),
    String(RcStr<'a>),
    Bool(bool),
    Nil,
    Function(Function<'a>),
    NativeFunction(for<'b> fn(Vec<Value<'b>>) -> Result<Value<'b>, NativeError<'b>>),
    Class(Class<'a>),
    Instance(Instance<'a>),
    BoundMethod(Function<'a>, Instance<'a>),
}

unsafe impl CloneInCellSafe for Value<'_> {}

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
            Value::BoundMethod(_, _) => "BoundMethod",
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

    fn lox_debug(&self) -> String {
        match self {
            Value::String(s) => format!("{:?}", s.deref()),
            _ => self.to_string(),
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
            Value::BoundMethod(method, instance) => write!(
                f,
                "<bound method {name} of {instance:?}>",
                name = method.name,
            ),
        }
    }
}

pub type Function<'a> = RcValue<FunctionInner<'a>>;

pub struct FunctionInner<'a> {
    name: &'a str,
    pub(crate) parameters: &'a [Variable<'a>],
    code: &'a [Statement<'a>],
    pub(crate) cells: Vec<Cell<Rc<Cell<Value<'a>>>>>,
    pub(crate) compiled_body: &'a Execute<'a>,
}

impl Debug for Function<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<function {} at {:p}>", self.name, Self::as_ptr(self))
    }
}

pub type Class<'a> = RcValue<ClassInner<'a>>;

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

    pub(crate) fn lookup_method(&self, name: InternedString) -> Option<&Value<'a>> {
        self.mro().find_map(|class| class.methods.get(&name))
    }
}

impl Debug for Class<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<class {} at {:p}>", self.name, Self::as_ptr(self))
    }
}

pub type Instance<'a> = RcValue<InstanceInner<'a>>;

pub struct InstanceInner<'a> {
    pub(crate) class: Class<'a>,
    pub(crate) attributes: RefCell<HashMap<InternedString, Value<'a>>>,
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

#[derive(Debug)]
pub enum ControlFlow<T, E> {
    Return(T),
    Error(E),
}

impl<T, E> From<E> for ControlFlow<T, E> {
    fn from(value: E) -> Self {
        ControlFlow::Error(value)
    }
}

#[derive(Debug, Report)]
#[exit_code(70)]
pub enum Error<'a> {
    #[error("type error in unary operator `{op}`: operand has type `{actual}` but `{op}` requires type `{expected}`")]
    #[with(
        actual = value.typ(),
        #[colour(Green)]
        expected = "Number",
    )]
    #[help("operator `{op}` can only be applied to numbers")]
    InvalidUnaryOp {
        op: UnaryOp<'a>,
        value: Value<'a>,
        #[diagnostics(
            0(label = "the operator in question", colour = Red),
            1(label = "this is of type `{actual}`", colour = Blue),
        )]
        at: ExpressionTypes::Unary<'a>,
    },

    #[error("type error in binary operator `{op}`: lhs has type `{lhs_type}`, but rhs has type `{rhs_type}`")]
    #[with(
        lhs_type = lhs.typ(),
        rhs_type = rhs.typ(),
        possible_types = match op.kind {
            BinOpKind::Plus => "two numbers or two strings",
            _ => "numbers",
        },
    )]
    #[help("operator `{op}` can only be applied to {possible_types}")]
    InvalidBinaryOp {
        lhs: Value<'a>,
        op: BinOp<'a>,
        rhs: Value<'a>,
        #[diagnostics(
            lhs(label = "this is of type `{lhs_type}`", colour = Blue),
            op(label = "the operator in question", colour = Magenta),
            rhs(label = "this is of type `{rhs_type}`", colour = Green),
        )]
        at: ExpressionTypes::Binary<'a>,
    },

    #[error("Undefined variable `{at}`.")]
    UndefinedName {
        #[diagnostics(loc(colour = Magenta))]
        at: Name<'a>,
    },

    #[error("not callable: `{callee_repr}` is of type `{callee_type}`")]
    #[with(
        callee_repr = callee.lox_debug(),
        callee_type = callee.typ(),
    )]
    Uncallable {
        callee: Value<'a>,
        #[diagnostics(
            callee(label = "this expression is of type `{callee_type}`", colour = Red),
        )]
        at: ExpressionTypes::Call<'a>,
    },

    #[error("arity error: `{callee}` expects {expected} args but {actual} args were passed")]
    #[with(
        actual = at.arguments.len(),
    )]
    ArityMismatch {
        callee: Value<'a>,
        expected: usize,
        #[diagnostics(
            callee(label = "expects {expected} arguments", colour = Red),
        )]
        at: ExpressionTypes::Call<'a>,
    },

    #[error("only instances have properties, but `{lhs}` is of type `{lhs_typ}`")]
    #[with(lhs_typ = lhs.typ())]
    NoProperty {
        lhs: Value<'a>,
        #[diagnostics(lhs(colour = Red))]
        at: ExpressionTypes::Attribute<'a>,
    },

    #[error("undefined property `{name}` of value `{lhs}`")]
    #[with(name = attribute.slice())]
    UndefinedProperty {
        lhs: Value<'a>,
        attribute: Name<'a>,
        #[diagnostics(lhs(colour = Red), attribute(colour = Magenta))]
        at: ExpressionTypes::Attribute<'a>,
    },

    #[error("undefined super property `{name}` of value `{super_}`")]
    #[with(name = attribute.slice())]
    UndefinedSuperProperty {
        super_: Value<'a>,
        attribute: Name<'a>,
        #[diagnostics(super_(colour = Red), attribute(colour = Magenta))]
        at: ExpressionTypes::Super<'a>,
    },

    #[error("only instances have fields, but `{lhs}` is of type `{lhs_typ}`")]
    #[with(lhs_typ = lhs.typ())]
    NoFields {
        lhs: Value<'a>,
        #[diagnostics(lhs(colour = Red), attribute(colour = Magenta))]
        at: AssignmentTargetTypes::Attribute<'a>,
    },

    #[error("only classes can be inherited from, but `{base}` is of type `{ty}`")]
    #[with(ty = base.typ())]
    InvalidBase {
        base: Value<'a>,
        #[diagnostics(0(colour = Red))]
        at: ExpressionTypes::Name<'a>,
    },
}

pub enum NativeError<'a> {
    Error(Error<'a>),
    ArityMismatch { expected: usize },
}

const ENV_SIZE: usize = 100_000;

pub struct Environment<'a> {
    stack: Box<[Value<'a>; ENV_SIZE]>,
    globals: HashMap<InternedString, usize>,
    is_global_defined: Box<[bool]>,
}

impl<'a> Environment<'a> {
    pub fn new(globals: HashMap<InternedString, usize>) -> Self {
        let mut stack: Box<[Value<'a>; ENV_SIZE]> = vec![Value::Nil; ENV_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let mut is_global_defined = vec![false; globals.len()].into_boxed_slice();
        if let Some(&slot) = globals.get(&interned::CLOCK) {
            stack[slot] = Value::NativeFunction(|arguments| {
                if !arguments.is_empty() {
                    return Err(NativeError::ArityMismatch { expected: 0 });
                }
                static START_TIME: OnceLock<Instant> = OnceLock::new();
                Ok(Value::Number(
                    START_TIME.get_or_init(Instant::now).elapsed().as_secs_f64(),
                ))
            });
            is_global_defined[slot] = true;
        }
        Self { stack, globals, is_global_defined }
    }

    pub(crate) fn get(
        &self,
        cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
        offset: usize,
        slot: Slot,
    ) -> Result<Value<'a>, Box<Error<'a>>> {
        let index = match slot {
            Slot::Local(slot) => offset + slot,
            Slot::Global(slot) => slot,
            Slot::Cell(slot) => return Ok(cell_vars[slot].get_clone().get_clone()),
        };
        Ok(self.stack[index].clone())
    }

    pub(crate) fn get_global_slot_by_name(
        &self,
        name: &'a Name<'a>,
    ) -> Result<usize, Box<Error<'a>>> {
        match self.globals.get(&name.id()).copied() {
            Some(slot) if self.is_global_defined[slot] => Ok(slot),
            _ => Err(Box::new(Error::UndefinedName { at: *name })),
        }
    }

    pub(crate) fn get_global_by_name(
        &self,
        name: &'a Name<'a>,
    ) -> Result<(usize, Value<'a>), Box<Error<'a>>> {
        let slot = self.get_global_slot_by_name(name)?;
        Ok((slot, self.stack[slot].clone()))
    }

    fn define_impl(
        &mut self,
        cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
        offset: usize,
        target: Target,
        value: Value<'a>,
        set_cell: impl FnOnce(&Cell<Rc<Cell<Value<'a>>>>, Value<'a>),
    ) {
        let index = match target {
            Target::Local(slot) => offset + slot,
            Target::GlobalByName => unreachable!(),
            Target::GlobalBySlot(slot) => {
                // FIXME: this is only necessary when called from `define`, not `set`.
                if let Some(is_defined) = self.is_global_defined.get_mut(slot) {
                    *is_defined = true;
                }
                slot
            }
            Target::Cell(slot) => {
                set_cell(&cell_vars[slot], value);
                return;
            }
        };
        self.stack[index] = value;
    }

    pub(crate) fn define(
        &mut self,
        cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
        offset: usize,
        target: Target,
        value: Value<'a>,
    ) {
        self.define_impl(cell_vars, offset, target, value, |cell, value| {
            cell.set(Rc::new(Cell::new(value)))
        })
    }

    pub(crate) fn set(
        &mut self,
        cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
        offset: usize,
        target: Target,
        value: Value<'a>,
    ) {
        self.define_impl(cell_vars, offset, target, value, |cell, value| {
            cell.get_clone().set(value)
        })
    }
}

pub fn eval<'a>(
    env: &mut Environment<'a>,
    cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
    offset: usize,
    expr: &Expression<'a>,
) -> Result<Value<'a>, Box<Error<'a>>> {
    use Value::*;
    Ok(match expr {
        Expression::Literal(lit) => match lit.kind {
            LiteralKind::Number(n) => Number(n),
            LiteralKind::String(s) => String(RcStr::Borrowed(s)),
            LiteralKind::True => Bool(true),
            LiteralKind::False => Bool(false),
            LiteralKind::Nil => Nil,
        },
        Expression::Unary(op, inner_expr) => {
            use UnaryOpKind::*;
            let value = eval(env, cell_vars, offset, inner_expr)?;
            match op.kind {
                Minus => match value {
                    Number(n) => Number(-n),
                    _ => Err(Error::InvalidUnaryOp { op: *op, value, at: expr.into_variant() })?,
                },
                Not => Bool(!value.is_truthy()),
            }
        }
        Expression::Assign { target, value, .. } => {
            let value = eval(env, cell_vars, offset, value)?;
            match target {
                AssignmentTarget::Variable(variable) => {
                    if let Target::GlobalByName = variable.target() {
                        let slot = env.get_global_slot_by_name(variable.name)?;
                        variable.set_target(Target::GlobalBySlot(slot));
                    }
                    env.set(cell_vars, offset, variable.target(), value.clone());
                }
                AssignmentTarget::Attribute { lhs, attribute } => {
                    let target_value = eval(env, cell_vars, offset, lhs)?;
                    match target_value {
                        Value::Instance(instance) => {
                            instance
                                .attributes
                                .borrow_mut()
                                .insert(attribute.id(), value.clone());
                        }
                        _ => Err(Error::NoFields {
                            lhs: target_value,
                            at: target.into_variant(),
                        })?,
                    }
                }
            }
            value
        }
        Expression::Binary { lhs, op, rhs } => {
            use BinOpKind::*;
            match op.kind {
                And => {
                    let lhs = eval(env, cell_vars, offset, lhs)?;
                    if lhs.is_truthy() {
                        eval(env, cell_vars, offset, rhs)?
                    }
                    else {
                        lhs
                    }
                }
                Or => {
                    let lhs = eval(env, cell_vars, offset, lhs)?;
                    if lhs.is_truthy() {
                        lhs
                    }
                    else {
                        eval(env, cell_vars, offset, rhs)?
                    }
                }
                _ => {
                    let lhs = eval(env, cell_vars, offset, lhs)?;
                    let rhs = eval(env, cell_vars, offset, rhs)?;
                    match (&lhs, op.kind, &rhs) {
                        (_, And | Or, _) => unreachable!(),
                        (_, EqualEqual, _) => Bool(lhs == rhs),
                        (_, NotEqual, _) => Bool(lhs != rhs),
                        (String(lhs), Plus, String(rhs)) =>
                            String(RcStr::Owned(Rc::from(format!("{}{}", lhs, rhs)))),
                        (Number(lhs), _, Number(rhs)) => match op.kind {
                            Plus => Number(lhs + rhs),
                            Less => Bool(lhs < rhs),
                            LessEqual => Bool(lhs <= rhs),
                            Greater => Bool(lhs > rhs),
                            GreaterEqual => Bool(lhs >= rhs),
                            Minus => Number(lhs - rhs),
                            Times => Number(lhs * rhs),
                            Divide => Number(lhs / rhs),
                            Power => Number(lhs.powf(*rhs)),
                            EqualEqual | NotEqual | Assign | And | Or => unreachable!(),
                        },
                        _ => Err(Error::InvalidBinaryOp {
                            lhs,
                            op: *op,
                            rhs,
                            at: expr.into_variant(),
                        })?,
                    }
                }
            }
        }
        Expression::Call {
            callee,
            arguments,
            stack_size_at_callsite,
            ..
        } => {
            let callee = eval(env, cell_vars, offset, callee)?;

            let eval_call = #[inline(always)]
            |env: &mut Environment<'a>,
                             function: &self::Function<'a>,
                             parameters: Skip<slice::Iter<Variable<'a>>>|
             -> Result<Value<'a>, Box<Error<'a>>> {
                if arguments.len() != parameters.len() {
                    Err(Error::ArityMismatch {
                        callee: callee.clone(),
                        expected: parameters.len(),
                        at: expr.into_variant(),
                    })?;
                }
                zip(*arguments, parameters).try_for_each(|(arg, param)| -> Result<(), Box<_>> {
                    let arg = eval(env, cell_vars, offset, arg)?;
                    env.define(
                        &function.cells,
                        offset + stack_size_at_callsite,
                        param.target(),
                        arg,
                    );
                    Ok(())
                })?;
                match execute(
                    env,
                    offset + stack_size_at_callsite,
                    function.code,
                    &function.cells,
                ) {
                    Ok(_) => Ok(Value::Nil),
                    Err(ControlFlow::Return(value)) => Ok(value),
                    Err(ControlFlow::Error(err)) => Err(err),
                }
                // FIXME: truncate env here to drop the callee’s locals
            };

            let eval_method_call = #[inline(always)]
            |env: &mut Environment<'a>,
                                    method: &self::Function<'a>,
                                    instance: Value<'a>|
             -> Result<Value<'a>, Box<Error<'a>>> {
                env.define(
                    &method.cells,
                    offset + stack_size_at_callsite,
                    method.parameters[0].target(),
                    instance,
                );
                let parameters = method.parameters.iter().skip(1);
                eval_call(env, method, parameters)
            };

            match callee {
                Value::Function(ref func) => eval_call(
                    env,
                    func,
                    // FIXME: clippy issue 11761
                    #[expect(clippy::iter_skip_zero)]
                    func.parameters.iter().skip(0),
                )?,
                Value::NativeFunction(func) => {
                    let arguments = arguments
                        .iter()
                        .map(|arg| eval(env, cell_vars, offset, arg))
                        .collect::<Result<_, _>>()?;
                    func(arguments).map_err(|err| match err {
                        NativeError::Error(err) => err,
                        NativeError::ArityMismatch { expected } => Error::ArityMismatch {
                            callee,
                            expected,
                            at: expr.into_variant(),
                        },
                    })?
                }
                Value::Class(ref class) => {
                    let instance = Value::Instance(RcValue::new(InstanceInner {
                        class: class.clone(),
                        attributes: RefCell::new(HashMap::default()),
                    }));
                    match class.lookup_method(interned::INIT) {
                        Some(Value::Function(ref init)) => {
                            eval_method_call(env, init, instance.clone())?;
                        }
                        Some(_) => unreachable!(),
                        None if arguments.is_empty() => (),
                        None => Err(Error::ArityMismatch {
                            callee: callee.clone(),
                            expected: 0,
                            at: expr.into_variant(),
                        })?,
                    }
                    instance
                }
                // FIXME: When explicitly calling `init`, the instance should be returned
                Value::BoundMethod(ref method, ref instance) =>
                    eval_method_call(env, method, Value::Instance(instance.clone()))?,
                _ => Err(Error::Uncallable { callee, at: expr.into_variant() })?,
            }
        }
        Expression::Grouping { expr, .. } => eval(env, cell_vars, offset, expr)?,
        Expression::Name(variable) => match variable.target() {
            Target::Local(slot) => env.get(cell_vars, offset, Slot::Local(slot))?,
            Target::GlobalByName => {
                let (slot, value) = env.get_global_by_name(variable.name)?;
                variable.set_target(Target::GlobalBySlot(slot));
                value
            }
            Target::GlobalBySlot(slot) => env.get(cell_vars, offset, Slot::Global(slot))?,
            Target::Cell(slot) => env.get(cell_vars, offset, Slot::Cell(slot))?,
        },
        Expression::Attribute { lhs, attribute } => {
            let lhs = eval(env, cell_vars, offset, lhs)?;
            match lhs {
                Value::Instance(ref instance) => instance
                    .attributes
                    .borrow()
                    .get(&attribute.id())
                    .cloned()
                    .or_else(|| {
                        instance
                            .class
                            .lookup_method(attribute.id())
                            .map(|method| match method {
                                Value::Function(method) =>
                                    Value::BoundMethod(method.clone(), instance.clone()),
                                _ => unreachable!(),
                            })
                    })
                    .ok_or_else(|| Error::UndefinedProperty {
                        lhs: lhs.clone(),
                        attribute: *attribute,
                        at: expr.into_variant(),
                    })?,
                _ => Err(Error::NoProperty { lhs, at: expr.into_variant() })?,
            }
        }
        Expression::Super { super_, this, attribute } => {
            let this = eval(env, cell_vars, offset, &Expression::Name(*this))?;
            let super_ = match eval(env, cell_vars, offset, &Expression::Name(*super_))? {
                Value::Class(super_) => super_,
                value => unreachable!("invalid base class value: {value}"),
            };
            match this {
                Value::Instance(ref instance) => super_
                    .lookup_method(attribute.id())
                    .map(|method| match method {
                        Value::Function(method) =>
                            Value::BoundMethod(method.clone(), instance.clone()),
                        _ => unreachable!(),
                    })
                    .ok_or_else(|| Error::UndefinedSuperProperty {
                        super_: this.clone(),
                        attribute: *attribute,
                        at: expr.into_variant(),
                    })?,
                _ => unreachable!(),
            }
        }
    })
}

pub fn execute<'a>(
    env: &mut Environment<'a>,
    offset: usize,
    program: &[Statement<'a>],
    cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
) -> Result<Value<'a>, ControlFlow<Value<'a>, Box<Error<'a>>>> {
    let mut last_value = Value::Nil;
    for statement in program {
        last_value = match statement {
            Statement::Expression(expr) => eval(env, cell_vars, offset, expr)?,
            Statement::Print(expr) => {
                println!("{}", eval(env, cell_vars, offset, expr)?);
                Value::Nil
            }
            Statement::Var(variable, initialiser) => {
                let value = if let Some(initialiser) = initialiser {
                    eval(env, cell_vars, offset, initialiser)?
                }
                else {
                    Value::Nil
                };
                env.define(cell_vars, offset, variable.target(), value);
                Value::Nil
            }
            Statement::Block(block) => execute(env, offset, block, cell_vars)?,
            Statement::If { condition, then, or_else } =>
                if eval(env, cell_vars, offset, condition)?.is_truthy() {
                    execute(env, offset, std::slice::from_ref(then), cell_vars)?
                }
                else {
                    match or_else {
                        Some(stmt) => execute(env, offset, std::slice::from_ref(stmt), cell_vars)?,
                        None => Value::Nil,
                    }
                },
            Statement::While { condition, body } => {
                while eval(env, cell_vars, offset, condition)?.is_truthy() {
                    execute(env, offset, std::slice::from_ref(body), cell_vars)?;
                }
                Value::Nil
            }
            Statement::For { init, condition, update, body } => {
                if let Some(init) = init {
                    execute(env, offset, std::slice::from_ref(init), cell_vars)?;
                }
                while condition
                    .as_ref()
                    .map_or(Ok(Value::Bool(true)), |cond| {
                        eval(env, cell_vars, offset, cond)
                    })?
                    .is_truthy()
                {
                    execute(env, offset, std::slice::from_ref(body), cell_vars)?;
                    if let Some(update) = update {
                        eval(env, cell_vars, offset, update)?;
                    }
                }
                Value::Nil
            }
            Statement::Function { target, function } => {
                // we need to define the function variable before evaluating the cells as the
                // function itself could be captured
                env.define(cell_vars, offset, target.target(), Value::Nil);
                let function = eval_function(cell_vars, function);
                env.set(cell_vars, offset, target.target(), function);
                Value::Nil
            }
            Statement::Return(expr) => {
                let return_value = expr
                    .as_ref()
                    .map_or(Ok(Value::Nil), |expr| eval(env, cell_vars, offset, expr))?;
                Err(ControlFlow::Return(return_value))?
            }
            Statement::Class { target, base, methods } => {
                let base = base
                    .as_ref()
                    .map(|base| match eval(env, cell_vars, offset, base)? {
                        Value::Class(class) => Ok::<Class, Box<Error<'a>>>(class),
                        value => Err(Error::InvalidBase { base: value, at: base.into_variant() })?,
                    })
                    .transpose()?;
                env.define(cell_vars, offset, target.target(), Value::Nil);
                let methods: HashMap<_, _> = methods
                    .iter()
                    .map(|method| (method.name.id(), eval_function(cell_vars, method)))
                    .collect();
                let class = RcValue::new(ClassInner { name: target.name.slice(), base, methods });
                env.set(
                    cell_vars,
                    offset,
                    target.target(),
                    Value::Class(class.clone()),
                );
                if let Some(ref base) = class.base {
                    let base = Value::Class(base.clone());
                    class.methods.values().for_each(|method| {
                        let Value::Function(method) = method
                        else {
                            unreachable!()
                        };
                        method.cells[0].set(Rc::new(Cell::new(base.clone())));
                    });
                }
                Value::Nil
            }
        }
    }
    Ok(last_value)
}

pub(crate) fn eval_function<'a>(
    cell_vars: &[Cell<Rc<Cell<Value<'a>>>>],
    function: &crate::scope::Function<'a>,
) -> Value<'a> {
    let crate::scope::Function {
        name,
        parameters,
        body,
        cells,
        compiled_body,
    } = function;
    let cells = cells
        .iter()
        .map(|cell| match cell {
            Some(idx) => Cell::new(cell_vars[*idx].get_clone()),
            None => Cell::new(Rc::new(Cell::new(Value::Nil))),
        })
        .collect();
    Value::Function(RcValue::new(FunctionInner {
        name: name.slice(),
        parameters,
        code: body,
        cells,
        compiled_body: *compiled_body,
    }))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bumpalo::Bump;
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::parse;
    use crate::scope;
    use crate::scope::Program;

    #[fixture]
    fn bump() -> Bump {
        Bump::new()
    }

    fn eval_str<'a>(bump: &'a Bump, src: &'a str) -> Result<Value<'a>, Error<'a>> {
        let (tokens, _) = crate::lex(bump, Path::new("<src>"), ";");
        let Ok(&[semi]) = tokens.collect::<Result<Vec<_>, _>>().as_deref()
        else {
            unreachable!()
        };
        let ast = crate::parse::tests::parse_str(bump, src).unwrap();
        let program =
            std::slice::from_ref(bump.alloc(parse::Statement::Expression { expr: ast, semi }));
        let Ok(Program {
            stmts: [scope::Statement::Expression(scoped_ast)],
            global_name_offsets,
            global_cell_count: 0,
        }) = scope::resolve_names(bump, &[], program)
        else {
            unreachable!()
        };
        let global_name_offsets = global_name_offsets
            .iter()
            .map(|(&name, v)| match v.target() {
                scope::Target::GlobalBySlot(slot) => (name, slot),
                _ => unreachable!(),
            })
            .collect();
        let global_cells = &[];
        eval(
            &mut Environment::new(global_name_offsets),
            global_cells,
            0,
            scoped_ast,
        )
        .map_err(|e| *e)
    }

    #[rstest]
    #[case::bool("true", Value::Bool(true))]
    #[case::bool("false", Value::Bool(false))]
    #[case::divide_numbers("4 / 2", Value::Number(2.0))]
    #[case::concat_strings(r#""a" + "b""#, Value::String(RcStr::Borrowed("ab")))]
    #[case::exponentiation("2 ** 2", Value::Number(4.0))]
    #[case::exponentiation("2 ** 2 ** 3", Value::Number(2.0_f64.powi(8)))]
    #[case::associativity("2 + 2 * 3", Value::Number(8.0))]
    #[case::associativity("2 + 2 - 3", Value::Number(1.0))]
    #[case::grouping("2 * (3 + 4)", Value::Number(14.0))]
    #[case::negation(r#"!"""#, Value::Bool(false))]
    #[case::negation(r#"!"abc""#, Value::Bool(false))]
    #[case::negation("!nil", Value::Bool(true))]
    #[case::negation("!0", Value::Bool(false))]
    #[case::negation("!27", Value::Bool(false))]
    #[case::negation("!true", Value::Bool(false))]
    #[case::negation("!false", Value::Bool(true))]
    #[case::precedence_of_negation_and_addition("-3 + 5", Value::Number(2.0))]
    #[case::precedence_of_addition_and_negation("3 + -5", Value::Number(-2.0))]
    #[case::precedence_of_negation_and_power("-2 ** 2", Value::Number(-4.0))]
    #[case::precedence_of_power_and_negation("2 ** -2", Value::Number(0.25))]
    fn test_eval<'a>(
        #[by_ref] bump: &'a Bump,
        #[case] src: &'static str,
        #[case] expected: Value<'a>,
    ) {
        pretty_assertions::assert_eq!(eval_str(bump, src).unwrap(), expected);
    }

    macro_rules! check {
        ($body:expr) => {
            for<'a> |result: Result<Value<'a>, Error<'a>>| -> () {
                #[allow(clippy::redundant_closure_call)]
                let () = $body(result);
            }
        };
    }

    macro_rules! check_err {
        ($pattern:pat) => {
            check!(|result| pretty_assertions::assert_matches!(result, Err($pattern)))
        };
    }

    #[rstest]
    #[case::add_nil(
        "42 + nil",
        check_err!(Error::InvalidBinaryOp {
            lhs: Value::Number(42.0),
            op: BinOp { kind: BinOpKind::Plus, token: _ },
            rhs: Value::Nil,
            at: _,
        }),
    )]
    #[case::type_error_in_grouping(
        "(nil - 2) + 27",
        check_err!(Error::InvalidBinaryOp {
            lhs: Value::Nil,
            op: BinOp { kind: BinOpKind::Minus, token: _ },
            rhs: Value::Number(2.0),
            at: _,
        }),
    )]
    #[case::type_error_in_grouping(
        "42 + (nil * 2)",
        check_err!(Error::InvalidBinaryOp {
            lhs: Value::Nil,
            op: BinOp { kind: BinOpKind::Times, token: _ },
            rhs: Value::Number(2.0),
            at: _,
        }),
    )]
    fn test_type_error(
        #[case] src: &str,
        #[case] expected: impl for<'a> FnOnce(Result<Value<'a>, Error<'a>>),
    ) {
        let bump = &Bump::new();
        expected(eval_str(bump, src));
    }
}
