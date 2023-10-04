use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;

use ariadne::Color::Blue;
use ariadne::Color::Green;
use ariadne::Color::Magenta;
use ariadne::Color::Red;
use crabbing_interpreters_derive_report::Report;
use variant_types::IntoVariant;

use crate::clone_from_cell::CloneInCellSafe;
use crate::clone_from_cell::GetClone;
use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::LiteralKind;
use crate::parse::Name;
use crate::parse::UnaryOp;
use crate::parse::UnaryOpKind;
use crate::rc_str::RcStr;
use crate::scope::Expression;
use crate::scope::ExpressionTypes;
use crate::scope::Slot;
use crate::scope::Statement;
use crate::scope::Target;
use crate::Sliced;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value<'a> {
    Number(f64),
    String(RcStr<'a>),
    Bool(bool),
    Nil,
    Function(Function<'a>),
    NativeFunction(for<'b> fn(Vec<Value<'b>>) -> Result<Value<'b>, NativeError<'b>>),
}

unsafe impl CloneInCellSafe for Value<'_> {}

#[derive(Clone)]
pub struct Function<'a>(Rc<FunctionInner<'a>>);

struct FunctionInner<'a> {
    name: &'a str,
    parameters: &'a [&'a str],
    code: &'a [Statement<'a>],
    cells: Vec<Rc<Cell<Value<'a>>>>,
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
        }
    }

    fn is_truthy(&self) -> bool {
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
        }
    }
}

impl Debug for Function<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<function {} at {:p}>", self.0.name, Rc::as_ptr(&self.0))
    }
}

impl PartialEq for Function<'_> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialOrd for Function<'_> {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        None
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
        #[diagnostics(0(colour = Magenta))]
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
}

pub enum NativeError<'a> {
    Error(Error<'a>),
    ArityMismatch { expected: usize },
}

pub struct Environment<'a> {
    stack: Box<[Value<'a>; 100_000]>,
    globals: HashMap<&'a str, usize>,
}

impl<'a> Environment<'a> {
    pub fn new(global_names: HashMap<&'a str, usize>) -> Self {
        let mut globals: Box<[Value<'a>; 100_000]> = vec![Value::Nil; 100_000]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        globals[0] = Value::NativeFunction(|arguments| {
            if !arguments.is_empty() {
                return Err(NativeError::ArityMismatch { expected: 0 });
            }
            static START_TIME: OnceLock<Instant> = OnceLock::new();
            Ok(Value::Number(
                START_TIME.get_or_init(Instant::now).elapsed().as_secs_f64(),
            ))
        });
        Self { stack: globals, globals: global_names }
    }

    fn get(
        &self,
        cell_vars: &[Rc<Cell<Value<'a>>>],
        offset: usize,
        slot: Slot,
    ) -> Result<Value<'a>, Box<Error<'a>>> {
        let index = match slot {
            Slot::Local(slot) => offset + slot,
            Slot::Global(slot) => slot,
            Slot::Cell(slot) => return Ok(cell_vars[slot].get_clone()),
        };
        Ok(self.stack[index].clone())
    }

    fn get_global_slot_by_name(&self, name: &'a Name<'a>) -> Result<usize, Box<Error<'a>>> {
        self.globals
            .get(name.slice())
            .copied()
            .ok_or_else(|| Box::new(Error::UndefinedName { at: *name }))
    }

    fn get_global_by_name(&self, name: &'a Name<'a>) -> Result<(usize, Value<'a>), Box<Error<'a>>> {
        let slot = self.get_global_slot_by_name(name)?;
        Ok((slot, self.stack[slot].clone()))
    }

    fn set(
        &mut self,
        cell_vars: &[Rc<Cell<Value<'a>>>],
        offset: usize,
        target: Target,
        value: Value<'a>,
    ) -> Result<(), Box<Error<'a>>> {
        let index = match target {
            Target::Local(slot) => offset + slot,
            Target::GlobalByName => unreachable!(),
            Target::GlobalBySlot(slot) => slot,
            Target::Cell(slot) => {
                cell_vars[slot].set(value);
                return Ok(());
            }
        };
        self.stack[index] = value;
        Ok(())
    }
}

pub fn eval<'a>(
    env: &mut Environment<'a>,
    cell_vars: &[Rc<Cell<Value<'a>>>],
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
        Expression::Assign { target: variable, value, .. } => {
            let value = eval(env, cell_vars, offset, value)?;
            let target = variable.target();
            if let Target::GlobalByName = target {
                let slot = env.get_global_slot_by_name(variable.name)?;
                variable.set_target(Target::GlobalBySlot(slot));
            }
            env.set(cell_vars, offset, target, value.clone())?;
            return Ok(value);
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
            match callee {
                Value::Function(ref func) => {
                    if arguments.len() != func.0.parameters.len() {
                        Err(Error::ArityMismatch {
                            callee: callee.clone(),
                            expected: func.0.parameters.len(),
                            at: expr.into_variant(),
                        })?;
                    }
                    let arguments = *arguments;
                    arguments.iter().enumerate().try_for_each(|(i, arg)| {
                        let arg = eval(env, cell_vars, offset, arg)?;
                        env.set(
                            cell_vars,
                            offset + stack_size_at_callsite,
                            Target::Local(i),
                            arg,
                        )?;
                        Ok::<(), Box<Error<'a>>>(())
                    })?;
                    match execute(
                        env,
                        offset + stack_size_at_callsite,
                        func.0.code,
                        &func.0.cells,
                    ) {
                        Ok(value) => Ok(value),
                        Err(ControlFlow::Return(value)) => Ok(value),
                        Err(ControlFlow::Error(err)) => Err(err),
                    }?
                    // FIXME: truncate env here to drop the callee’s locals
                }
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
    })
}

pub fn execute<'a>(
    env: &mut Environment<'a>,
    offset: usize,
    program: &[Statement<'a>],
    cell_vars: &[Rc<Cell<Value<'a>>>],
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
                env.set(cell_vars, offset, variable.target(), value)?;
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
                Ok::<_, Box<Error<'a>>>(Value::Nil)
            }?,
            Statement::Function {
                target,
                parameters: _,
                parameter_names,
                body,
                cells,
            } => {
                let cells = cells
                    .iter()
                    .map(|cell| match cell {
                        Some(idx) => Rc::clone(&cell_vars[*idx]),
                        None => Rc::new(Cell::new(Value::Nil)),
                    })
                    .collect();
                env.set(
                    cell_vars,
                    offset,
                    target.target(),
                    Value::Function(Function(Rc::new(FunctionInner {
                        name: target.name.slice(),
                        parameters: parameter_names,
                        code: body,
                        cells,
                    }))),
                )?;
                Value::Nil
            }
            Statement::Return(expr) => {
                let return_value = expr
                    .as_ref()
                    .map_or(Ok(Value::Nil), |expr| eval(env, cell_vars, offset, expr))?;
                Err(ControlFlow::Return(return_value))?
            }
        }
    }
    Ok(last_value)
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::parse;
    use crate::scope;

    #[fixture]
    fn bump() -> Bump {
        Bump::new()
    }

    fn eval_str<'a>(bump: &'a Bump, src: &'a str) -> Result<Value<'a>, Error<'a>> {
        let Ok((&[semi], _)) = crate::lex(bump, "<src>", ";")
        else {
            unreachable!()
        };
        let ast = crate::parse::tests::parse_str(bump, src).unwrap();
        let program =
            std::slice::from_ref(bump.alloc(parse::Statement::Expression { expr: ast, semi }));
        let Ok((
            [scope::Statement::Expression(scoped_ast)],
            global_name_offsets,
            _global_cell_count @ 0,
        )) = scope::resolve_names(bump, &[], program)
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
    fn test_eval<'a>(bump: &'a Bump, #[case] src: &'static str, #[case] expected: Value<'a>) {
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
