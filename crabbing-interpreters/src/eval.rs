use std::cell::Cell;
use std::cell::RefCell;
use std::fmt::Debug;
use std::iter::zip;
use std::iter::Skip;
use std::slice;

use ariadne::Color::Blue;
use ariadne::Color::Green;
use ariadne::Color::Magenta;
use ariadne::Color::Red;
use crabbing_interpreters_derive_report::Report;
use variant_types::IntoVariant;

use crate::environment::Environment;
use crate::gc::Gc;
use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::gc::Trace as _;
use crate::hash_map::HashMap;
use crate::interner::interned;
use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::LiteralKind;
use crate::parse::Name;
use crate::parse::UnaryOp;
use crate::parse::UnaryOpKind;
use crate::scope::AssignmentTarget;
use crate::scope::AssignmentTargetTypes;
use crate::scope::Expression;
use crate::scope::ExpressionTypes;
use crate::scope::Slot;
use crate::scope::Statement;
use crate::scope::Target;
use crate::scope::Variable;
use crate::value::nanboxed;
use crate::value::BoundMethodInner;
use crate::value::Cells;
use crate::value::Class;
use crate::value::ClassInner;
use crate::value::FunctionInner;
use crate::value::InstanceInner;
use crate::value::NativeError;
use crate::value::Value;
use crate::Report;
use crate::Sliced;

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

impl<'a, T, E> From<E> for ControlFlow<T, Box<dyn Report + 'a>>
where
    E: Report + 'a,
{
    fn from(value: E) -> Self {
        ControlFlow::Error(Box::new(value))
    }
}

impl<'a, T, E> From<ControlFlow<T, E>> for ControlFlow<T, Box<dyn Report + 'a>>
where
    E: Report + 'a,
{
    fn from(value: ControlFlow<T, E>) -> Self {
        match value {
            ControlFlow::Return(value) => ControlFlow::Return(value),
            ControlFlow::Error(error) => ControlFlow::Error(error.into()),
        }
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

pub fn eval<'a>(
    env: &mut Environment<'a>,
    cell_vars: Cells<'a>,
    offset: usize,
    expr: &Expression<'a>,
    trace_call_stack: &dyn Fn(),
) -> Result<Value<'a>, Box<Error<'a>>> {
    use Value::*;
    Ok(match expr {
        Expression::Literal(lit) => match lit.kind {
            LiteralKind::Number(n) => Number(n),
            LiteralKind::String(s) => String(GcStr::new_in(env.gc, s)),
            LiteralKind::True => Bool(true),
            LiteralKind::False => Bool(false),
            LiteralKind::Nil => Nil,
        },
        Expression::Unary(op, inner_expr) => {
            use UnaryOpKind::*;
            let value = eval(env, cell_vars, offset, inner_expr, trace_call_stack)?;
            match op.kind {
                Minus => match value {
                    Number(n) => Number(-n),
                    _ => Err(Error::InvalidUnaryOp { op: *op, value, at: expr.into_variant() })?,
                },
                Not => Bool(!value.is_truthy()),
            }
        }
        Expression::Assign { target, value, .. } => {
            let value = eval(env, cell_vars, offset, value, trace_call_stack)?;
            match target {
                AssignmentTarget::Variable(variable) => {
                    if let Target::GlobalByName = variable.target() {
                        let slot = env.get_global_slot_by_name(variable.name)?;
                        variable.set_target(Target::GlobalBySlot(slot));
                    }
                    env.set(cell_vars, offset, variable.target(), value.into_nanboxed());
                }
                AssignmentTarget::Attribute { lhs, attribute } => {
                    let target_value = eval(env, cell_vars, offset, lhs, trace_call_stack)?;
                    match target_value {
                        Value::Instance(instance) => {
                            instance
                                .attributes
                                .borrow_mut()
                                .insert(attribute.id(), value.into_nanboxed());
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
                    let lhs = eval(env, cell_vars, offset, lhs, trace_call_stack)?;
                    if lhs.is_truthy() {
                        eval(env, cell_vars, offset, rhs, trace_call_stack)?
                    }
                    else {
                        lhs
                    }
                }
                Or => {
                    let lhs = eval(env, cell_vars, offset, lhs, trace_call_stack)?;
                    if lhs.is_truthy() {
                        lhs
                    }
                    else {
                        eval(env, cell_vars, offset, rhs, trace_call_stack)?
                    }
                }
                _ => {
                    let lhs = eval(env, cell_vars, offset, lhs, trace_call_stack)?;
                    let rhs = eval(env, cell_vars, offset, rhs, trace_call_stack)?;
                    match (&lhs, op.kind, &rhs) {
                        (_, And | Or, _) => unreachable!(),
                        (_, EqualEqual, _) => Bool(lhs == rhs),
                        (_, NotEqual, _) => Bool(lhs != rhs),
                        (String(lhs), Plus, String(rhs)) =>
                            String(GcStr::new_in(env.gc, &format!("{lhs}{rhs}"))),
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
            let callee = eval(env, cell_vars, offset, callee, trace_call_stack)?;

            let eval_call = #[inline(always)]
            |env: &mut Environment<'a>,
                             function: &crate::value::Function<'a>,
                             parameters: Skip<slice::Iter<Variable<'a>>>|
             -> Result<Value<'a>, Box<Error<'a>>> {
                if arguments.len() != parameters.len() {
                    Err(Error::ArityMismatch {
                        callee,
                        expected: parameters.len(),
                        at: expr.into_variant(),
                    })?;
                }
                zip(*arguments, parameters).try_for_each(|(arg, param)| -> Result<(), Box<_>> {
                    let arg = eval(env, cell_vars, offset, arg, trace_call_stack)?;
                    env.define(
                        function.cells,
                        offset + stack_size_at_callsite,
                        param.target(),
                        arg.into_nanboxed(),
                    );
                    Ok(())
                })?;
                let trace_call_stack = &move || {
                    callee.trace();
                    trace_call_stack();
                };
                match execute(
                    env,
                    offset + stack_size_at_callsite,
                    function.code,
                    function.cells,
                    trace_call_stack,
                ) {
                    Ok(_) => Ok(Value::Nil),
                    Err(ControlFlow::Return(value)) => Ok(value),
                    Err(ControlFlow::Error(err)) => Err(err),
                }
                // FIXME: truncate env here to drop the callee’s locals
            };

            let eval_method_call = #[inline(always)]
            |env: &mut Environment<'a>,
                                    method: &crate::value::Function<'a>,
                                    instance: Value<'a>|
             -> Result<Value<'a>, Box<Error<'a>>> {
                env.define(
                    method.cells,
                    offset + stack_size_at_callsite,
                    method.parameters[0].target(),
                    instance.into_nanboxed(),
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
                        .map(|arg| eval(env, cell_vars, offset, arg, trace_call_stack))
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
                Value::Class(class) => {
                    let instance = Value::Instance(GcRef::new_in(
                        env.gc,
                        InstanceInner {
                            class,
                            attributes: RefCell::new(HashMap::default()),
                        },
                    ));
                    match class
                        .lookup_method(interned::INIT)
                        .map(nanboxed::Value::parse)
                    {
                        Some(Value::Function(init)) => {
                            eval_method_call(env, &init, instance)?;
                        }
                        Some(_) => unreachable!(),
                        None if arguments.is_empty() => (),
                        None => Err(Error::ArityMismatch {
                            callee,
                            expected: 0,
                            at: expr.into_variant(),
                        })?,
                    }
                    instance
                }
                Value::BoundMethod(bound_method) => eval_method_call(
                    env,
                    &bound_method.method,
                    Value::Instance(bound_method.instance),
                )?,
                _ => Err(Error::Uncallable { callee, at: expr.into_variant() })?,
            }
        }
        Expression::Grouping { expr, .. } => eval(env, cell_vars, offset, expr, trace_call_stack)?,
        Expression::Name(variable) => match variable.target() {
            Target::Local(slot) => env.get(cell_vars, offset, Slot::Local(slot)).parse(),
            Target::GlobalByName => {
                let (slot, value) = env.get_global_by_name(variable.name)?;
                variable.set_target(Target::GlobalBySlot(slot));
                value.parse()
            }
            Target::GlobalBySlot(slot) => env.get(cell_vars, offset, Slot::Global(slot)).parse(),
            Target::Cell(slot) => env.get(cell_vars, offset, Slot::Cell(slot)).parse(),
        },
        Expression::Attribute { lhs, attribute } => {
            let lhs = eval(env, cell_vars, offset, lhs, trace_call_stack)?;
            match lhs {
                Value::Instance(instance) => instance
                    .attributes
                    .borrow()
                    .get(&attribute.id())
                    .copied()
                    .map(nanboxed::Value::parse)
                    .or_else(|| {
                        instance.class.lookup_method(attribute.id()).map(|method| {
                            match method.parse() {
                                Value::Function(method) => Value::BoundMethod(GcRef::new_in(
                                    env.gc,
                                    BoundMethodInner { method, instance },
                                )),
                                _ => unreachable!(),
                            }
                        })
                    })
                    .ok_or_else(|| Error::UndefinedProperty {
                        lhs,
                        attribute: *attribute,
                        at: expr.into_variant(),
                    })?,
                _ => Err(Error::NoProperty { lhs, at: expr.into_variant() })?,
            }
        }
        Expression::Super { super_, this, attribute } => {
            let this = eval(
                env,
                cell_vars,
                offset,
                &Expression::Name(*this),
                trace_call_stack,
            )?;
            let super_ = match eval(
                env,
                cell_vars,
                offset,
                &Expression::Name(*super_),
                trace_call_stack,
            )? {
                Value::Class(super_) => super_,
                value => unreachable!("invalid base class value: {value}"),
            };
            match this {
                Value::Instance(instance) => super_
                    .lookup_method(attribute.id())
                    .map(|method| match method.parse() {
                        Value::Function(method) => Value::BoundMethod(GcRef::new_in(
                            env.gc,
                            BoundMethodInner { method, instance },
                        )),
                        _ => unreachable!(),
                    })
                    .ok_or_else(|| Error::UndefinedSuperProperty {
                        super_: this,
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
    cell_vars: Cells<'a>,
    trace_call_stack: &dyn Fn(),
) -> Result<Value<'a>, ControlFlow<Value<'a>, Box<Error<'a>>>> {
    let mut last_value = Value::Nil;
    for statement in program {
        last_value = match statement {
            Statement::Expression(expr) => eval(env, cell_vars, offset, expr, trace_call_stack)?,
            Statement::Print(expr) => {
                println!("{}", eval(env, cell_vars, offset, expr, trace_call_stack)?);
                Value::Nil
            }
            Statement::Var(variable, initialiser) => {
                let value = if let Some(initialiser) = initialiser {
                    eval(env, cell_vars, offset, initialiser, trace_call_stack)?
                }
                else {
                    Value::Nil
                };
                env.define(cell_vars, offset, variable.target(), value.into_nanboxed());
                Value::Nil
            }
            Statement::Block(block) => execute(env, offset, block, cell_vars, trace_call_stack)?,
            Statement::If { condition, then, or_else } =>
                if eval(env, cell_vars, offset, condition, trace_call_stack)?.is_truthy() {
                    execute(
                        env,
                        offset,
                        slice::from_ref(then),
                        cell_vars,
                        trace_call_stack,
                    )?
                }
                else {
                    match or_else {
                        Some(stmt) => execute(
                            env,
                            offset,
                            slice::from_ref(stmt),
                            cell_vars,
                            trace_call_stack,
                        )?,
                        None => Value::Nil,
                    }
                },
            Statement::While { condition, body } => {
                while eval(env, cell_vars, offset, condition, trace_call_stack)?.is_truthy() {
                    execute(
                        env,
                        offset,
                        slice::from_ref(body),
                        cell_vars,
                        trace_call_stack,
                    )?;
                }
                Value::Nil
            }
            Statement::For { init, condition, update, body } => {
                if let Some(init) = init {
                    execute(
                        env,
                        offset,
                        slice::from_ref(init),
                        cell_vars,
                        trace_call_stack,
                    )?;
                }
                while condition
                    .as_ref()
                    .map_or(Ok(Value::Bool(true)), |cond| {
                        eval(env, cell_vars, offset, cond, trace_call_stack)
                    })?
                    .is_truthy()
                {
                    execute(
                        env,
                        offset,
                        slice::from_ref(body),
                        cell_vars,
                        trace_call_stack,
                    )?;
                    if let Some(update) = update {
                        eval(env, cell_vars, offset, update, trace_call_stack)?;
                    }
                }
                Value::Nil
            }
            Statement::Function { target, function } => {
                // we need to define the function variable before evaluating the cells as the
                // function itself could be captured
                env.define(
                    cell_vars,
                    offset,
                    target.target(),
                    Value::Nil.into_nanboxed(),
                );
                let function = eval_function(env.gc, cell_vars, function);
                env.set(cell_vars, offset, target.target(), function);
                Value::Nil
            }
            Statement::Return(expr) => {
                let return_value = expr.as_ref().map_or(Ok(Value::Nil), |expr| {
                    eval(env, cell_vars, offset, expr, trace_call_stack)
                })?;
                Err(ControlFlow::Return(return_value))?
            }
            Statement::InitReturn(this) => Err(ControlFlow::Return(eval(
                env,
                cell_vars,
                offset,
                this,
                trace_call_stack,
            )?))?,
            Statement::Class { target, base, methods } => {
                let base = base
                    .as_ref()
                    .map(
                        |base| match eval(env, cell_vars, offset, base, trace_call_stack)? {
                            Value::Class(class) => Ok::<Class, Box<Error<'a>>>(class),
                            value =>
                                Err(Error::InvalidBase { base: value, at: base.into_variant() })?,
                        },
                    )
                    .transpose()?;
                env.define(
                    cell_vars,
                    offset,
                    target.target(),
                    Value::Nil.into_nanboxed(),
                );
                let methods = methods
                    .iter()
                    .map(|method| (method.name.id(), eval_function(env.gc, cell_vars, method)))
                    .collect();
                let class = GcRef::new_in(
                    env.gc,
                    ClassInner { name: target.name.slice(), base, methods },
                );
                env.set(
                    cell_vars,
                    offset,
                    target.target(),
                    Value::Class(class).into_nanboxed(),
                );
                if let Some(base) = class.base {
                    let base = Value::Class(base).into_nanboxed();
                    class.methods.values().for_each(|method| {
                        let Value::Function(method) = method.parse()
                        else {
                            unreachable!()
                        };
                        method.cells[0].set(GcRef::new_in(env.gc, Cell::new(base)));
                    });
                }
                Value::Nil
            }
        };

        env.collect_if_necessary(last_value.into_nanboxed(), cell_vars, trace_call_stack);
    }
    Ok(last_value)
}

pub(crate) fn eval_function<'a>(
    gc: &'a Gc,
    cell_vars: Cells<'a>,
    function: &crate::scope::Function<'a>,
) -> nanboxed::Value<'a> {
    let crate::scope::Function {
        name,
        parameters,
        body,
        cells,
        compiled_body,
    } = function;

    let cells = GcRef::from_iter_in(
        gc,
        cells.iter().map(|cell| match cell {
            Some(idx) => Cell::new(cell_vars[*idx].get()),
            None => Cell::new(GcRef::new_in(gc, Cell::new(Value::Nil.into_nanboxed()))),
        }),
    );

    Value::Function(GcRef::new_in(
        gc,
        FunctionInner {
            name: name.slice(),
            parameters,
            code: body,
            cells,
            compiled_body: *compiled_body,
            code_ptr: 0,
        },
    ))
    .into_nanboxed()
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

    #[fixture]
    fn gc() -> Gc {
        Gc::default()
    }

    fn eval_str<'a>(gc: &'a Gc, bump: &'a Bump, src: &'a str) -> Result<Value<'a>, Error<'a>> {
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
            scopes: _,
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
        let global_cells = GcRef::from_iter_in(gc, [].into_iter());
        eval(
            &mut Environment::new(gc, global_name_offsets, global_cells),
            global_cells,
            0,
            scoped_ast,
            &|| (),
        )
        .map_err(|e| *e)
    }

    #[rstest]
    #[case::bool("true", Value::Bool(true))]
    #[case::bool("false", Value::Bool(false))]
    #[case::divide_numbers("4 / 2", Value::Number(2.0))]
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
        #[by_ref] gc: &'a Gc,
        #[case] src: &'static str,
        #[case] expected: Value<'a>,
    ) {
        pretty_assertions::assert_eq!(eval_str(gc, bump, src).unwrap(), expected);
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
        let gc = &Gc::default();
        expected(eval_str(gc, bump, src));
    }
}
