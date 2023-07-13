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
