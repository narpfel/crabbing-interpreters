use std::borrow::Cow;
use std::fmt::Display;

use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::Expression;
use crate::parse::LiteralKind;
use crate::parse::UnaryOp;
use crate::parse::UnaryOpKind;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Value<'a> {
    Number(f64),
    String(Cow<'a, str>),
    Bool(bool),
    Nil,
}

impl Value<'_> {
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
        }
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
}

pub fn eval<'a>(expr: &Expression<'a>) -> Result<Value<'a>, TypeError<'a>> {
    use Value::*;
    Ok(match expr {
        Expression::Literal(lit) => match lit.kind {
            LiteralKind::Number(n) => Number(n),
            LiteralKind::String(s) => String(Cow::Borrowed(s)),
            LiteralKind::True => Bool(true),
            LiteralKind::False => Bool(false),
            LiteralKind::Nil => Nil,
        },
        Expression::Unary(op, inner_expr) => {
            use UnaryOpKind::*;
            let value = eval(inner_expr)?;
            match op.kind {
                Minus => match value {
                    Number(n) => Number(-n),
                    _ => return Err(TypeError::InvalidUnaryOp { op: *op, value, at: *expr }),
                },
                Not => Bool(!value.is_truthy()),
            }
        }
        Expression::Binary { lhs, op, rhs } => {
            use BinOpKind::*;
            let lhs = eval(lhs)?;
            let rhs = eval(rhs)?;
            match (&lhs, op.kind, &rhs) {
                (_, EqualEqual, _) => Bool(lhs == rhs),
                (_, NotEqual, _) => Bool(lhs != rhs),
                (String(lhs), Plus, String(rhs)) => String(Cow::Owned(format!("{}{}", lhs, rhs))),
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
                    EqualEqual | NotEqual => unreachable!(),
                },
                _ => return Err(TypeError::InvalidBinaryOp { lhs, op: *op, rhs, at: *expr }),
            }
        }
        Expression::Grouping(expr) => eval(expr)?,
    })
}

#[cfg(test)]
mod tests {
    use bumpalo::Bump;
    use rstest::rstest;

    use super::*;

    fn eval_str<'a>(bump: &'a Bump, src: &'a str) -> Result<Value<'a>, TypeError<'a>> {
        let ast = crate::parse::tests::parse_str(bump, src).unwrap();
        eval(&ast)
    }

    #[rstest]
    #[case::bool("true", Value::Bool(true))]
    #[case::bool("false", Value::Bool(false))]
    #[case::divide_numbers("4 / 2", Value::Number(2.0))]
    #[case::concat_strings(r#""a" + "b""#, Value::String(Cow::Borrowed("ab")))]
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
    fn test_eval(#[case] src: &str, #[case] expected: Value) {
        let bump = &Bump::new();
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
