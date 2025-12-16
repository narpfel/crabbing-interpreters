use std::cell::Cell;
use std::iter::zip;
use std::iter::Skip;
use std::slice;

use bumpalo::Bump;
use variant_types::IntoVariant as _;

use crate::environment::Environment;
use crate::eval::eval_function;
use crate::eval::ControlFlow;
use crate::eval::Error;
use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::gc::Trace as _;
use crate::interner::interned;
use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::LiteralKind;
use crate::parse::UnaryOpKind;
use crate::scope::AssignmentTarget;
use crate::scope::Expression;
use crate::scope::Slot;
use crate::scope::Statement;
use crate::scope::Target;
use crate::scope::Variable;
use crate::value::instance::InstanceInner;
use crate::value::nanboxed;
use crate::value::BoundMethodInner;
use crate::value::Cells;
use crate::value::ClassInner;
use crate::value::Function;
use crate::value::Value;

pub(crate) struct State<'a, 'b> {
    pub(crate) env: &'b mut Environment<'a>,
    pub(crate) offset: usize,
    pub(crate) cell_vars: Cells<'a>,
    pub(crate) trace_call_stack: &'b dyn Fn(),
}

type ExecResult<'a> = Result<Value<'a>, ControlFlow<Value<'a>, Box<Error<'a>>>>;
type EvalResult<'a> = Result<Value<'a>, Box<Error<'a>>>;

pub(crate) type Execute<'a> = dyn for<'b, 'c> Fn(&'c mut State<'a, 'b>) -> ExecResult<'a> + 'a;
pub(crate) type Evaluate<'a> = dyn for<'b, 'c> Fn(&'c mut State<'a, 'b>) -> EvalResult<'a> + 'a;

pub(crate) fn compile_block<'a>(bump: &'a Bump, block: &'a [Statement<'a>]) -> &'a Execute<'a> {
    let stmts = &*bump.alloc_slice_fill_iter(block.iter().map(|stmt| compile_stmt(bump, stmt)));
    bump.alloc(
        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
            for stmt in stmts {
                stmt(state)?;
                state.env.collect_if_necessary(
                    Value::Nil.into_nanboxed(),
                    state.cell_vars,
                    state.trace_call_stack,
                );
            }
            Ok(Value::Nil)
        },
    )
}

fn compile_stmt<'a>(bump: &'a Bump, stmt: &'a Statement<'a>) -> &'a Execute<'a> {
    match stmt {
        Statement::Expression(expr) => {
            let expr = compile_expr(bump, expr);
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    expr(state).map_err(ControlFlow::Error)
                },
            )
        }
        Statement::Print(expr) => {
            let expr = compile_expr(bump, expr);
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    println!("{}", expr(state)?);
                    Ok(Value::Nil)
                },
            )
        }
        Statement::Var(variable, initialiser) => {
            let variable = *variable;
            let initialiser = initialiser
                .as_ref()
                .map(|initialiser| compile_expr(bump, initialiser))
                .unwrap_or(&|_| Ok(Value::Nil));
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    let initialiser = initialiser(state)?;
                    state.env.define(
                        state.cell_vars,
                        state.offset,
                        variable.target(),
                        initialiser.into_nanboxed(),
                    );
                    Ok(Value::Nil)
                },
            )
        }
        Statement::Block(block) => compile_block(bump, block),
        Statement::If { condition, then, or_else } => {
            let condition = compile_expr(bump, condition);
            let then = compile_stmt(bump, then);
            let or_else = or_else
                .map(|or_else| compile_stmt(bump, or_else))
                .unwrap_or(&|_| Ok(Value::Nil));
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    if condition(state)?.is_truthy() {
                        then(state)?;
                    }
                    else {
                        or_else(state)?;
                    }
                    Ok(Value::Nil)
                },
            )
        }
        Statement::While { condition, body } => {
            let condition = compile_expr(bump, condition);
            let body = compile_stmt(bump, body);
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    while condition(state)?.is_truthy() {
                        body(state)?;
                    }
                    Ok(Value::Nil)
                },
            )
        }
        Statement::For { init, condition, update, body } => {
            let init = init
                .as_ref()
                .map(|init| compile_stmt(bump, init))
                .unwrap_or(&|_| Ok(Value::Nil));
            let condition = condition
                .as_ref()
                .map(|condition| compile_expr(bump, condition))
                .unwrap_or(&|_| Ok(Value::Bool(true)));
            let update = update
                .as_ref()
                .map(|update| compile_expr(bump, update))
                .unwrap_or(&|_| Ok(Value::Nil));
            let body = compile_stmt(bump, body);

            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    init(state)?;
                    while condition(state)?.is_truthy() {
                        body(state)?;
                        update(state)?;
                    }
                    Ok(Value::Nil)
                },
            )
        }
        Statement::Function { target, function } => {
            let target = *target;
            let function = *function;
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    state.env.define(
                        state.cell_vars,
                        state.offset,
                        target.target(),
                        Value::Nil.into_nanboxed(),
                    );
                    let function = eval_function(state.env.gc, state.cell_vars, &function);
                    state
                        .env
                        .set(state.cell_vars, state.offset, target.target(), function);
                    Ok(Value::Nil)
                },
            )
        }
        Statement::Return(expr) => {
            let expr = expr
                .as_ref()
                .map(|expr| compile_expr(bump, expr))
                .unwrap_or(&|_| Ok(Value::Nil));
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    Err(ControlFlow::Return(expr(state)?))
                },
            )
        }
        Statement::InitReturn(this) => {
            let this = compile_expr(bump, this);
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    Err(ControlFlow::Return(this(state)?))
                },
            )
        }
        Statement::Class { target, base, methods } => {
            let target = *target;
            let base_expr = base;
            let base = base.as_ref().map(|base| compile_expr(bump, base));
            let methods = *methods;
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> ExecResult<'a> {
                    let base = match base {
                        Some(base) => Some(base(state)?),
                        None => None,
                    };
                    let base = match base {
                        Some(Value::Class(class)) => Some(class),
                        Some(value) => Err(Box::new(Error::InvalidBase {
                            base: value,
                            at: base_expr.unwrap().into_variant(),
                        }))?,
                        None => None,
                    };
                    state.env.define(
                        state.cell_vars,
                        state.offset,
                        target.target(),
                        Value::Nil.into_nanboxed(),
                    );
                    let methods = methods
                        .iter()
                        .map(|method| {
                            (
                                method.name.id(),
                                eval_function(state.env.gc, state.cell_vars, method),
                            )
                        })
                        .collect();
                    let class = GcRef::new_in(
                        state.env.gc,
                        ClassInner { name: target.name.slice(), base, methods },
                    );
                    state.env.set(
                        state.cell_vars,
                        state.offset,
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
                            method.cells[0].set(GcRef::new_in(state.env.gc, Cell::new(base)));
                        });
                    }
                    Ok(Value::Nil)
                },
            )
        }
    }
}

fn compile_expr<'a>(bump: &'a Bump, expr: &'a Expression<'a>) -> &'a Evaluate<'a> {
    match expr {
        Expression::Literal(lit) => match lit.kind {
            LiteralKind::Number(n) => bump.alloc(
                for<'b, 'c> move |_: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    Ok(Value::Number(n))
                },
            ),
            LiteralKind::String(s) => bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    Ok(Value::String(GcStr::new_in(state.env.gc, s)))
                },
            ),
            LiteralKind::True => &|_| Ok(Value::Bool(true)),
            LiteralKind::False => &|_| Ok(Value::Bool(false)),
            LiteralKind::Nil => &|_| Ok(Value::Nil),
        },
        Expression::Unary(operator, operand) => {
            let operand = compile_expr(bump, operand);
            match operator.kind {
                UnaryOpKind::Minus => bump.alloc(
                    for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                        let operand = operand(state)?;
                        match operand {
                            Value::Number(n) => Ok(Value::Number(-n)),
                            _ => Err(Error::InvalidUnaryOp {
                                op: *operator,
                                value: operand,
                                at: expr.into_variant(),
                            })?,
                        }
                    },
                ),
                UnaryOpKind::Not => bump.alloc(
                    for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                        let operand = operand(state)?;
                        Ok(Value::Bool(!operand.is_truthy()))
                    },
                ),
            }
        }
        Expression::Binary { lhs, op, rhs } => {
            let lhs = *lhs;
            let rhs = *rhs;
            match op.kind {
                BinOpKind::EqualEqual =>
                    any_binop(bump, lhs, rhs, |_, lhs, rhs| Ok(Value::Bool(lhs == rhs))),
                BinOpKind::NotEqual =>
                    any_binop(bump, lhs, rhs, |_, lhs, rhs| Ok(Value::Bool(lhs != rhs))),
                BinOpKind::Less =>
                    number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| Value::Bool(lhs < rhs)),
                BinOpKind::LessEqual =>
                    number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| Value::Bool(lhs <= rhs)),
                BinOpKind::Greater =>
                    number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| Value::Bool(lhs > rhs)),
                BinOpKind::GreaterEqual =>
                    number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| Value::Bool(lhs >= rhs)),
                BinOpKind::Plus =>
                    any_binop(bump, lhs, rhs, |state, lhs, rhs| match (&lhs, &rhs) {
                        (Value::Number(lhs), Value::Number(rhs)) => Ok(Value::Number(lhs + rhs)),
                        (Value::String(lhs), Value::String(rhs)) => Ok(Value::String(
                            GcStr::new_in(state.env.gc, &format!("{lhs}{rhs}")),
                        )),
                        _ => Err(Error::InvalidBinaryOp {
                            lhs,
                            op: *op,
                            rhs,
                            at: expr.into_variant(),
                        })?,
                    }),
                BinOpKind::Minus => number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| {
                    Value::Number(lhs - rhs)
                }),
                BinOpKind::Times => number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| {
                    Value::Number(lhs * rhs)
                }),
                BinOpKind::Divide => number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| {
                    Value::Number(lhs / rhs)
                }),
                BinOpKind::Power => number_binop(bump, expr, op, lhs, rhs, |lhs, rhs| {
                    Value::Number(lhs.powf(rhs))
                }),
                BinOpKind::Assign => unreachable!(),
                BinOpKind::And => {
                    let lhs = compile_expr(bump, lhs);
                    let rhs = compile_expr(bump, rhs);
                    bump.alloc(
                        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                            let lhs = lhs(state)?;
                            if lhs.is_truthy() {
                                rhs(state)
                            }
                            else {
                                Ok(lhs)
                            }
                        },
                    )
                }
                BinOpKind::Or => {
                    let lhs = compile_expr(bump, lhs);
                    let rhs = compile_expr(bump, rhs);
                    bump.alloc(
                        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                            let lhs = lhs(state)?;
                            if lhs.is_truthy() {
                                Ok(lhs)
                            }
                            else {
                                rhs(state)
                            }
                        },
                    )
                }
            }
        }
        Expression::Grouping { l_paren: _, expr, r_paren: _ } => compile_expr(bump, expr),
        Expression::Name(variable) => match variable.target() {
            Target::Local(slot) => bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    Ok(state
                        .env
                        .get(state.cell_vars, state.offset, Slot::Local(slot))
                        .parse())
                },
            ),
            Target::GlobalByName => bump.alloc(
                // FIXME: optimisation missing
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    let (_, value) = state.env.get_global_by_name(variable.name)?;
                    Ok(value.parse())
                },
            ),
            Target::GlobalBySlot(slot) => bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    Ok(state
                        .env
                        .get(state.cell_vars, state.offset, Slot::Global(slot))
                        .parse())
                },
            ),
            Target::Cell(slot) => bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    Ok(state
                        .env
                        .get(state.cell_vars, state.offset, Slot::Cell(slot))
                        .parse())
                },
            ),
        },
        Expression::Assign { target, equal: _, value } => {
            let value = compile_expr(bump, value);
            match target {
                AssignmentTarget::Variable(variable) => {
                    let variable = *variable;
                    bump.alloc(
                        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                            let value = value(state)?;
                            if let Target::GlobalByName = variable.target() {
                                let slot = state.env.get_global_slot_by_name(variable.name)?;
                                variable.set_target(Target::GlobalBySlot(slot));
                            }
                            state.env.set(
                                state.cell_vars,
                                state.offset,
                                variable.target(),
                                value.into_nanboxed(),
                            );
                            Ok(value)
                        },
                    )
                }
                AssignmentTarget::Attribute { lhs, attribute } => {
                    let lhs = compile_expr(bump, lhs);
                    bump.alloc(
                        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                            let value = value(state)?;
                            let target_value = lhs(state)?;
                            match target_value {
                                Value::Instance(instance) =>
                                    instance.setattr(attribute.id(), value.into_nanboxed()),
                                _ => Err(Error::NoFields {
                                    lhs: target_value,
                                    at: target.into_variant(),
                                })?,
                            }
                            Ok(value)
                        },
                    )
                }
            }
        }
        Expression::Call {
            callee,
            l_paren: _,
            arguments,
            r_paren: _,
            stack_size_at_callsite,
        } => {
            let callee = compile_expr(bump, callee);
            let arguments =
                &*bump.alloc_slice_fill_iter(arguments.iter().map(|arg| compile_expr(bump, arg)));
            let stack_size_at_callsite = stack_size_at_callsite.get();
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    let callee = callee(state)?;

                    let eval_call = #[inline(always)]
                    |state: &mut State<'a, '_>,
                                     function: &Function<'a>,
                                     parameters: Skip<slice::Iter<Variable<'a>>>|
                     -> EvalResult<'a> {
                        if arguments.len() != parameters.len() {
                            Err(Error::ArityMismatch {
                                callee,
                                expected: parameters.len(),
                                at: expr.into_variant(),
                            })?;
                        }
                        zip(arguments, parameters).try_for_each(
                            |(arg, param)| -> Result<(), Box<_>> {
                                let arg = arg(state)?;
                                state.env.define(
                                    function.cells,
                                    state.offset + stack_size_at_callsite,
                                    param.target(),
                                    arg.into_nanboxed(),
                                );
                                Ok(())
                            },
                        )?;
                        let trace_call_stack = state.trace_call_stack;
                        let trace_call_stack = &move || {
                            trace_call_stack();
                            callee.trace();
                        };
                        match (function.compiled_body)(&mut State {
                            env: state.env,
                            offset: state.offset + stack_size_at_callsite,
                            cell_vars: function.cells,
                            trace_call_stack,
                        }) {
                            Ok(_) => Ok(Value::Nil),
                            Err(ControlFlow::Return(value)) => Ok(value),
                            Err(ControlFlow::Error(err)) => Err(err),
                        }
                        // FIXME: truncate env here to drop the callee’s locals
                    };

                    let eval_method_call =
                        #[inline(always)]
                        |state: &mut State<'a, '_>,
                         method: &Function<'a>,
                         instance: Value<'a>|
                         -> Result<Value<'a>, Box<Error<'a>>> {
                            state.env.define(
                                method.cells,
                                state.offset + stack_size_at_callsite,
                                method.parameters[0].target(),
                                instance.into_nanboxed(),
                            );
                            let parameters = method.parameters.iter().skip(1);
                            eval_call(state, method, parameters)
                        };

                    match callee {
                        Value::Function(ref func) => eval_call(
                            state,
                            func,
                            // FIXME: clippy issue 11761
                            #[expect(clippy::iter_skip_zero)]
                            func.parameters.iter().skip(0),
                        ),
                        Value::NativeFunction(func) => {
                            let arguments = arguments
                                .iter()
                                .map(|arg| arg(state))
                                .collect::<Result<_, _>>()?;
                            func(state.env, arguments).map_err(|err| {
                                Box::new(err.at_expr(&state.env.interner, callee, expr))
                            })
                        }
                        Value::Class(class) => {
                            let instance = Value::Instance(GcRef::new_in(
                                state.env.gc,
                                InstanceInner::new(class),
                            ));
                            match class
                                .lookup_method(interned::INIT)
                                .map(nanboxed::Value::parse)
                            {
                                Some(Value::Function(init)) => {
                                    eval_method_call(state, &init, instance)?;
                                }
                                Some(_) => unreachable!(),
                                None if arguments.is_empty() => (),
                                None => Err(Error::ArityMismatch {
                                    callee,
                                    expected: 0,
                                    at: expr.into_variant(),
                                })?,
                            }
                            Ok(instance)
                        }

                        Value::BoundMethod(bound_method) => eval_method_call(
                            state,
                            &bound_method.method,
                            Value::Instance(bound_method.instance),
                        ),
                        _ => Err(Error::Uncallable { callee, at: expr.into_variant() })?,
                    }
                },
            )
        }
        Expression::Attribute { lhs, attribute } => {
            let lhs = compile_expr(bump, lhs);
            let attr_id = attribute.id();
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    let lhs = lhs(state)?;
                    match lhs {
                        Value::Instance(instance) => instance
                            .getattr(attr_id)
                            .ok()
                            .map(nanboxed::Value::parse)
                            .or_else(|| {
                                instance.class.lookup_method(attr_id).map(|method| {
                                    match method.parse() {
                                        Value::Function(method) =>
                                            Value::BoundMethod(GcRef::new_in(
                                                state.env.gc,
                                                BoundMethodInner { method, instance },
                                            )),
                                        _ => unreachable!(),
                                    }
                                })
                            })
                            .ok_or_else(|| {
                                Box::new(Error::UndefinedProperty {
                                    lhs,
                                    attribute: *attribute,
                                    at: expr.into_variant(),
                                })
                            }),
                        _ => Err(Error::NoProperty { lhs, at: expr.into_variant() })?,
                    }
                },
            )
        }
        Expression::Super { super_, this, attribute } => {
            let this = compile_expr(bump, bump.alloc(Expression::Name(*this)));
            let super_ = compile_expr(bump, bump.alloc(Expression::Name(*super_)));
            let attr_id = attribute.id();
            bump.alloc(
                for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
                    let super_class = match super_(state)? {
                        Value::Class(super_class) => super_class,
                        value => unreachable!("invalid base class value: {value}"),
                    };
                    let this = this(state)?;
                    match this {
                        Value::Instance(instance) => super_class
                            .lookup_method(attr_id)
                            .map(|method| match method.parse() {
                                Value::Function(method) => Value::BoundMethod(GcRef::new_in(
                                    state.env.gc,
                                    BoundMethodInner { method, instance },
                                )),
                                _ => unreachable!(),
                            })
                            .ok_or_else(|| {
                                Box::new(Error::UndefinedSuperProperty {
                                    super_: this,
                                    attribute: *attribute,
                                    at: expr.into_variant(),
                                })
                            }),
                        _ => unreachable!(),
                    }
                },
            )
        }
    }
}

fn any_binop<'a>(
    bump: &'a Bump,
    lhs: &'a Expression<'a>,
    rhs: &'a Expression<'a>,
    op: impl for<'b, 'c> Fn(&'c mut State<'a, 'b>, Value<'a>, Value<'a>) -> EvalResult<'a> + 'a,
) -> &'a Evaluate<'a> {
    let lhs = compile_expr(bump, lhs);
    let rhs = compile_expr(bump, rhs);
    bump.alloc(
        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
            let lhs = lhs(state)?;
            let rhs = rhs(state)?;
            op(state, lhs, rhs)
        },
    )
}

fn number_binop<'a>(
    bump: &'a Bump,
    expr: &'a Expression<'a>,
    operator: &'a BinOp<'a>,
    lhs: &'a Expression<'a>,
    rhs: &'a Expression<'a>,
    op: impl Fn(f64, f64) -> Value<'a> + 'a,
) -> &'a Evaluate<'a> {
    let lhs = compile_expr(bump, lhs);
    let rhs = compile_expr(bump, rhs);
    bump.alloc(
        for<'b, 'c> move |state: &'c mut State<'a, 'b>| -> EvalResult<'a> {
            let lhs = lhs(state)?;
            let rhs = rhs(state)?;
            match (&lhs, &rhs) {
                (Value::Number(lhs), Value::Number(rhs)) => Ok(op(*lhs, *rhs)),
                _ => Err(Box::new(Error::InvalidBinaryOp {
                    lhs,
                    op: *operator,
                    rhs,
                    at: expr.into_variant(),
                })),
            }
        },
    )
}
