use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;

use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::Expression;
use crate::parse::LiteralKind;
use crate::parse::Name;
use crate::parse::Statement;
use crate::parse::UnaryOp;
use crate::parse::UnaryOpKind;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value<'a> {
    Number(f64),
    String(Rc<str>),
    Bool(bool),
    Nil,
    Function(Function<'a>),
    NativeFunction(fn(Vec<Value<'a>>) -> Result<Value<'a>, TypeError<'a>>),
}

#[derive(Clone)]
pub struct Function<'a>(Rc<FunctionInner<'a>>);

struct FunctionInner<'a> {
    name: &'a str,
    parameters: &'a [&'a str],
    code: &'a [Statement<'a>],
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

#[derive(Debug, thiserror::Error)]
pub enum TypeError<'a> {
    #[error("invalid operand to unary operator")]
    InvalidUnaryOp {
        op: UnaryOp<'a>,
        value: Value<'a>,
        at: Expression<'a>,
    },
    #[error("invalid operands to binary operator")]
    InvalidBinaryOp {
        lhs: Value<'a>,
        op: BinOp<'a>,
        rhs: Value<'a>,
        at: Expression<'a>,
    },
    #[error("unbound name")]
    NameError(Name<'a>),
}

pub struct Environment<'a> {
    scopes: Vec<std::collections::HashMap<&'a str, Value<'a>>>,
}

impl<'a> Environment<'a> {
    pub fn new() -> Self {
        let mut globals = HashMap::default();
        globals.insert(
            "clock",
            Value::NativeFunction(|arguments| {
                if !arguments.is_empty() {
                    todo!("type (arity) error");
                }
                static START_TIME: OnceLock<Instant> = OnceLock::new();
                Ok(Value::Number(
                    START_TIME.get_or_init(Instant::now).elapsed().as_secs_f64(),
                ))
            }),
        );
        Self { scopes: vec![globals] }
    }

    fn with_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.scopes.push(HashMap::default());
        let result = f(self);
        self.scopes.pop();
        result
    }

    fn with_stackframe<T>(
        &mut self,
        new_scope: HashMap<&'a str, Value<'a>>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let globals = std::mem::take(&mut self.scopes[0]);
        let mut scope = Self { scopes: vec![globals] };
        scope.scopes.push(new_scope);
        let result = scope.with_scope(f);
        std::mem::swap(&mut self.scopes[0], &mut scope.scopes[0]);
        result
    }

    fn get(&self, name: &Name<'a>) -> Result<Value<'a>, TypeError<'a>> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name.name()) {
                return Ok(value.clone());
            }
        }

        Err(TypeError::NameError(*name))
    }

    fn set(&mut self, name: &Name<'a>, value: Value<'a>) -> Result<(), TypeError<'a>> {
        for scope in self.scopes.iter_mut().rev() {
            if let Entry::Occupied(mut entry) = scope.entry(name.name()) {
                entry.insert(value);
                return Ok(());
            }
        }
        Err(TypeError::NameError(*name))
    }

    fn insert(&mut self, name: &'a str, value: Value<'a>) {
        self.scopes.last_mut().unwrap().insert(name, value);
    }
}

pub fn eval<'a>(
    env: &mut Environment<'a>,
    expr: &Expression<'a>,
) -> Result<Value<'a>, TypeError<'a>> {
    use Value::*;
    Ok(match expr {
        Expression::Literal(lit) => match lit.kind {
            LiteralKind::Number(n) => Number(n),
            LiteralKind::String(s) => String(Rc::from(s)),
            LiteralKind::True => Bool(true),
            LiteralKind::False => Bool(false),
            LiteralKind::Nil => Nil,
        },
        Expression::Unary(op, inner_expr) => {
            use UnaryOpKind::*;
            let value = eval(env, inner_expr)?;
            match op.kind {
                Minus => match value {
                    Number(n) => Number(-n),
                    _ => return Err(TypeError::InvalidUnaryOp { op: *op, value, at: *expr }),
                },
                Not => Bool(!value.is_truthy()),
            }
        }
        Expression::Assign { target, value, .. } => {
            let value = eval(env, value)?;
            env.set(target, value.clone())?;
            return Ok(value);
        }
        Expression::Binary { lhs, op, rhs } => {
            use BinOpKind::*;
            match op.kind {
                And => {
                    let lhs = eval(env, lhs)?;
                    if lhs.is_truthy() {
                        eval(env, rhs)?
                    }
                    else {
                        lhs
                    }
                }
                Or => {
                    let lhs = eval(env, lhs)?;
                    if lhs.is_truthy() {
                        lhs
                    }
                    else {
                        eval(env, rhs)?
                    }
                }
                _ => {
                    let lhs = eval(env, lhs)?;
                    let rhs = eval(env, rhs)?;
                    match (&lhs, op.kind, &rhs) {
                        (_, And | Or, _) => unreachable!(),
                        (_, EqualEqual, _) => Bool(lhs == rhs),
                        (_, NotEqual, _) => Bool(lhs != rhs),
                        (String(lhs), Plus, String(rhs)) =>
                            String(Rc::from(format!("{}{}", lhs, rhs))),
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
                        _ =>
                            return Err(TypeError::InvalidBinaryOp { lhs, op: *op, rhs, at: *expr }),
                    }
                }
            }
        }
        Expression::Call { callee, arguments, .. } => {
            let callee = eval(env, callee)?;
            match callee {
                Value::Function(func) => {
                    if arguments.len() != func.0.parameters.len() {
                        todo!("type error: parameter count mismatch");
                    }
                    let arguments = *arguments;
                    let params = func.0.parameters;
                    let new_scope = params
                        .iter()
                        .zip(arguments.iter())
                        .map(|(&param, arg)| {
                            let arg = eval(env, arg)?;
                            Ok((param, arg))
                        })
                        .collect::<Result<_, _>>()?;
                    env.with_stackframe(new_scope, |env| execute(env, func.0.code))?
                }
                Value::NativeFunction(func) => {
                    let arguments = arguments
                        .iter()
                        .map(|arg| eval(env, arg))
                        .collect::<Result<_, _>>()?;
                    func(arguments)?
                }
                _ => todo!("type error: not callable"),
            }
        }
        Expression::Grouping { expr, .. } => eval(env, expr)?,
        Expression::Ident(name) => env.get(name)?,
    })
}

pub fn execute<'a>(
    env: &mut Environment<'a>,
    program: &[Statement<'a>],
) -> Result<Value<'a>, TypeError<'a>> {
    let mut last_value = Value::Nil;
    for statement in program {
        last_value = match statement {
            Statement::Expression(expr) => eval(env, expr)?,
            Statement::Print(expr) => {
                println!("{}", eval(env, expr)?);
                Value::Nil
            }
            Statement::Var(name, initialiser) => {
                let value = if let Some(initialiser) = initialiser {
                    eval(env, initialiser)?
                }
                else {
                    Value::Nil
                };
                env.insert(name.name(), value);
                Value::Nil
            }
            Statement::Block(block) => env.with_scope(|env| execute(env, block))?,
            Statement::If { condition, then, or_else } =>
                if eval(env, condition)?.is_truthy() {
                    execute(env, std::slice::from_ref(then))?
                }
                else {
                    match or_else {
                        Some(stmt) => execute(env, std::slice::from_ref(stmt))?,
                        None => Value::Nil,
                    }
                },
            Statement::While { condition, body } => {
                while eval(env, condition)?.is_truthy() {
                    execute(env, std::slice::from_ref(body))?;
                }
                Value::Nil
            }
            Statement::For { init, condition, update, body } => env.with_scope(|env| {
                if let Some(init) = init {
                    execute(env, std::slice::from_ref(init))?;
                }
                while condition
                    .map_or(Ok(Value::Bool(true)), |cond| eval(env, &cond))?
                    .is_truthy()
                {
                    execute(env, std::slice::from_ref(body))?;
                    if let Some(update) = update {
                        eval(env, update)?;
                    }
                }
                Ok(Value::Nil)
            })?,
            Statement::Function {
                name,
                parameters: _,
                parameter_names,
                body,
            } => {
                env.insert(
                    name.name(),
                    Value::Function(Function(Rc::new(FunctionInner {
                        name: name.name(),
                        parameters: parameter_names,
                        code: body,
                    }))),
                );
                Value::Nil
            }
        }
    }
    Ok(last_value)
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;
    use rstest::rstest;

    use super::*;

    fn eval_str<'a>(bump: &'a Bump, src: &'a str) -> Result<Value<'a>, TypeError<'a>> {
        let ast = crate::parse::tests::parse_str(bump, src).unwrap();
        let mut env = Environment::new();
        eval(&mut env, &ast)
    }

    #[rstest]
    #[case::bool("true", Value::Bool(true))]
    #[case::bool("false", Value::Bool(false))]
    #[case::divide_numbers("4 / 2", Value::Number(2.0))]
    #[case::concat_strings(r#""a" + "b""#, Value::String(Rc::from("ab")))]
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
    fn test_eval(#[case] src: &'static str, #[case] expected: Value) {
        // FIXME: remove the leak
        let bump = Box::leak(Box::new(Bump::new()));
        pretty_assertions::assert_eq!(eval_str(bump, src).unwrap(), expected);
    }

    macro_rules! check {
        ($body:expr) => {
            for<'a> |result: Result<Value<'a>, TypeError<'a>>| -> () {
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
        check_err!(TypeError::InvalidBinaryOp {
            lhs: Value::Number(42.0),
            op: BinOp { kind: BinOpKind::Plus, token: _ },
            rhs: Value::Nil,
            at: _,
        }),
    )]
    #[case::type_error_in_grouping(
        "(nil - 2) + 27",
        check_err!(TypeError::InvalidBinaryOp {
            lhs: Value::Nil,
            op: BinOp { kind: BinOpKind::Minus, token: _ },
            rhs: Value::Number(2.0),
            at: _,
        }),
    )]
    #[case::type_error_in_grouping(
        "42 + (nil * 2)",
        check_err!(TypeError::InvalidBinaryOp {
            lhs: Value::Nil,
            op: BinOp { kind: BinOpKind::Times, token: _ },
            rhs: Value::Number(2.0),
            at: _,
        }),
    )]
    fn test_type_error(
        #[case] src: &str,
        #[case] expected: impl for<'a> FnOnce(Result<Value<'a>, TypeError<'a>>),
    ) {
        let bump = &Bump::new();
        expected(eval_str(bump, src));
    }
}
