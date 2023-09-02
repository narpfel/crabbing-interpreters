use std::iter::Copied;
use std::iter::Peekable;

use bumpalo::Bump;

use crate::lex::Loc;
use crate::lex::Token;
use crate::lex::TokenKind;
use crate::Report;

#[derive(Debug)]
pub enum Error<'a> {
    Eof(Eof),
    ExpectedEof {
        actual: Token<'a>,
    },
    UnexpectedToken {
        expected: &'static [TokenKind],
        actual: Token<'a>,
    },
    UnexpectedTokenWithKind {
        expected: TokenKind,
        actual: Token<'a>,
    },
    UnexpectedTokenMsg {
        expected: &'static str,
        actual: Token<'a>,
    },
    AmbiguousPrecedences {
        left_op_token: Token<'a>,
        right_op: BinOp<'a>,
    },
    InvalidAssignmentTarget {
        lhs: Expression<'a>,
        equal: Token<'a>,
        rhs: Expression<'a>,
    },
    InvalidForLoopInitialiser(Statement<'a>),
}

impl From<Eof> for Error<'_> {
    fn from(value: Eof) -> Self {
        Self::Eof(value)
    }
}

impl Report for Error<'_> {
    fn print(&self) {
        match self {
            Error::Eof(_) => eprintln!("Unexpected end of file"),
            Error::ExpectedEof { actual } => {
                eprintln!(
                    "[line {}] Error at '{}': Expected EOF",
                    actual.loc().line(),
                    actual.slice(),
                );
            }
            Error::UnexpectedToken { expected, actual } => {
                eprintln!(
                    "[line {}] Error at '{}': Expect one of the following tokens: {:?}",
                    actual.loc().line(),
                    actual.slice(),
                    expected,
                );
            }
            Error::UnexpectedTokenWithKind { expected, actual } => {
                let message = format!(
                    "[line {}] Error at '{}': Expect {}.",
                    actual.loc().line(),
                    actual.slice(),
                    match expected {
                        TokenKind::Identifier => "variable name".to_owned(),
                        _ => format!("{expected:?}"),
                    },
                );
                eprintln!("{message}");
            }
            Error::UnexpectedTokenMsg { expected, actual } => {
                eprintln!(
                    "[line {}] Error at '{}': Expect {}.",
                    actual.loc().line(),
                    actual.slice(),
                    expected,
                );
            }
            Error::AmbiguousPrecedences { left_op_token, right_op } => {
                eprintln!(
                    "[line {}] Error at '{}': ambiguous precedences for operators `{}` and `{}`",
                    left_op_token.loc().line(),
                    left_op_token.slice(),
                    left_op_token.slice(),
                    right_op.token.slice(),
                );
            }
            Error::InvalidAssignmentTarget { equal, .. } => {
                eprintln!(
                    "[line {}] Error at '{}': Invalid assignment target.",
                    equal.loc().line(),
                    equal.slice(),
                );
            }
            Error::InvalidForLoopInitialiser(_) => {
                eprintln!("[line ??] Error at '??': Invalid for loop initialiser");
            }
        }
    }

    fn exit_code(&self) -> i32 {
        65
    }
}

#[derive(Debug)]
pub struct Eof;

#[derive(Debug, Clone, Copy)]
pub enum Statement<'a> {
    Expression(Expression<'a>),
    Print(Expression<'a>),
    Var(Name<'a>, Option<Expression<'a>>),
    Block(&'a [Statement<'a>]),
    If {
        condition: Expression<'a>,
        then: &'a Statement<'a>,
        or_else: Option<&'a Statement<'a>>,
    },
    While {
        condition: Expression<'a>,
        body: &'a Statement<'a>,
    },
    For {
        init: Option<&'a Statement<'a>>,
        condition: Option<Expression<'a>>,
        update: Option<Expression<'a>>,
        body: &'a Statement<'a>,
    },
    // FIXME: functions should know their stack frame’s size to make it possible to reserve enough
    // space before calling them and to drop the frame after the call returns
    Function {
        name: Name<'a>,
        parameters: &'a [Name<'a>],
        parameter_names: &'a [&'a str],
        body: &'a [Statement<'a>],
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Expression<'a> {
    Literal(Literal<'a>),
    Unary(UnaryOp<'a>, &'a Expression<'a>),
    Binary {
        lhs: &'a Expression<'a>,
        op: BinOp<'a>,
        rhs: &'a Expression<'a>,
    },
    Grouping {
        l_paren: Token<'a>,
        expr: &'a Expression<'a>,
        r_paren: Token<'a>,
    },
    Ident(Name<'a>),
    Assign {
        target: Name<'a>,
        equal: Token<'a>,
        value: &'a Expression<'a>,
    },
    Call {
        callee: &'a Expression<'a>,
        l_paren: Token<'a>,
        arguments: &'a [Expression<'a>],
        r_paren: Token<'a>,
    },
}

impl<'a> Expression<'a> {
    #[cfg(test)]
    pub fn as_sexpr(&self) -> String {
        match self {
            Expression::Literal(lit) => lit.kind.value_string(),
            Expression::Unary(operator, operand) =>
                format!("({} {})", operator.token.slice(), operand.as_sexpr()),

            Expression::Binary { lhs, op, rhs } => format!(
                "({} {} {})",
                op.token.slice(),
                lhs.as_sexpr(),
                rhs.as_sexpr(),
            ),
            Expression::Grouping { expr, .. } => format!("(group {})", expr.as_sexpr()),
            Expression::Ident(name) => format!("(name {})", name.slice()),
            Expression::Assign { target, value, .. } =>
                format!("(= {} {})", target.slice(), value.as_sexpr()),
            Expression::Call { callee, arguments, .. } => format!(
                "(call {}{}{})",
                callee.as_sexpr(),
                if arguments.is_empty() { "" } else { " " },
                arguments
                    .iter()
                    .map(Expression::as_sexpr)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Literal<'a> {
    pub(crate) kind: LiteralKind<'a>,
    token: Token<'a>,
}

impl<'a> Literal<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        self.token.loc()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LiteralKind<'a> {
    Number(f64),
    String(&'a str),
    True,
    False,
    Nil,
}

impl LiteralKind<'_> {
    pub(crate) fn value_string(self) -> String {
        match self {
            LiteralKind::Number(x) => format!("{x:?}"),
            LiteralKind::String(s) => format!("{s:?}"),
            LiteralKind::True => "true".to_owned(),
            LiteralKind::False => "false".to_owned(),
            LiteralKind::Nil => "nil".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnaryOp<'a> {
    pub(crate) kind: UnaryOpKind,
    pub(crate) token: Token<'a>,
}

impl<'a> UnaryOp<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        self.token.loc()
    }

    pub(crate) fn slice(&self) -> &'a str {
        self.token.slice()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOpKind {
    Minus,
    Not,
}

#[derive(Debug, Clone, Copy)]
pub struct BinOp<'a> {
    pub(crate) kind: BinOpKind,
    pub(crate) token: Token<'a>,
}

impl<'a> BinOp<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        self.token.loc()
    }

    pub(crate) fn slice(&self) -> &'a str {
        self.token.slice()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Plus,
    Minus,
    Times,
    Divide,
    Power,
    Assign,
    And,
    Or,
}

impl<'a> BinOp<'a> {
    fn new(token: Token<'a>) -> Result<Self, Error<'a>> {
        use TokenKind::*;
        let kind = match token.kind {
            EqualEqual => BinOpKind::EqualEqual,
            BangEqual => BinOpKind::NotEqual,
            Less => BinOpKind::Less,
            LessEqual => BinOpKind::LessEqual,
            Greater => BinOpKind::Greater,
            GreaterEqual => BinOpKind::GreaterEqual,
            Plus => BinOpKind::Plus,
            Minus => BinOpKind::Minus,
            Star => BinOpKind::Times,
            Slash => BinOpKind::Divide,
            StarStar => BinOpKind::Power,
            Equal => BinOpKind::Assign,
            And => BinOpKind::And,
            Or => BinOpKind::Or,
            _ => unexpected_token_with_message("a binary operator", token)?,
        };
        Ok(Self { kind, token })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Name<'a>(pub(crate) Token<'a>);

impl<'a> Name<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        self.0.loc()
    }

    pub(crate) fn slice(&self) -> &'a str {
        self.0.slice()
    }
}

pub struct Tokens<'a>(Peekable<Copied<std::slice::Iter<'a, Token<'a>>>>);

impl<'a> Tokens<'a> {
    fn next(&mut self) -> Result<Token<'a>, Eof> {
        self.0.next().not_eof()
    }

    fn peek(&mut self) -> Result<Token<'a>, Eof> {
        self.0.peek().copied().not_eof()
    }

    fn consume(&mut self, kind: TokenKind) -> Result<Token<'a>, Error<'a>> {
        let token = self.next()?;
        if token.kind == kind {
            Ok(token)
        }
        else {
            Err(Error::UnexpectedTokenWithKind { expected: kind, actual: token })
        }
    }

    fn consume_one_of(&mut self, kinds: &'static [TokenKind]) -> Result<Token<'a>, Error<'a>> {
        let token = self.next()?;
        if kinds.contains(&token.kind) {
            Ok(token)
        }
        else {
            unexpected_token(kinds, token)?
        }
    }

    fn eof(&mut self) -> Result<(), Error<'a>> {
        let token = self.0.peek();
        if let Some(&token) = token {
            Err(Error::ExpectedEof { actual: token })
        }
        else {
            Ok(())
        }
    }
}

pub fn parse<'a, T>(
    parser: impl FnOnce(&'a Bump, &mut Tokens<'a>) -> Result<T, Error<'a>>,
    bump: &'a Bump,
    tokens: &'a [Token<'a>],
) -> Result<T, Error<'a>> {
    let tokens = &mut Tokens(tokens.iter().copied().peekable());
    let result = parser(bump, tokens)?;
    tokens.eof()?;
    Ok(result)
}

pub fn program<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a>,
) -> Result<&'a [Statement<'a>], Error<'a>> {
    let mut statements = Vec::new();

    while tokens.peek().is_ok() {
        statements.push(declaration(bump, tokens)?);
    }

    tokens.eof()?;
    Ok(bump.alloc_slice_copy(&statements))
}

fn declaration<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    let token = tokens.peek()?;
    Ok(match token.kind {
        TokenKind::Var => vardecl(bump, tokens)?,
        TokenKind::Fun => function(bump, tokens)?,
        _ => statement(bump, tokens)?,
    })
}

fn vardecl<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    tokens.consume(TokenKind::Var)?;
    let name = Name(tokens.consume(TokenKind::Identifier)?);
    let maybe_equal = tokens.peek()?;
    let initialiser = if matches!(maybe_equal.kind, TokenKind::Equal) {
        tokens.consume(TokenKind::Equal)?;
        let initialiser = expression(bump, tokens)?;
        Some(initialiser)
    }
    else {
        None
    };
    tokens.consume(TokenKind::Semicolon)?;
    Ok(Statement::Var(name, initialiser))
}

fn function<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    tokens.consume(TokenKind::Fun)?;
    let name = Name(tokens.consume(TokenKind::Identifier)?);
    tokens.consume(TokenKind::LParen)?;
    let mut parameters = Vec::new();
    loop {
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::RParen) {
            break;
        }
        if matches!(token.kind, TokenKind::Identifier) {
            let token = tokens.consume(TokenKind::Identifier)?;
            parameters.push(Name(token));
        }
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::Comma) {
            tokens.consume(TokenKind::Comma)?;
        }
        else {
            break;
        }
    }
    tokens.consume(TokenKind::RParen)?;
    let body = block(bump, tokens)?;
    Ok(Statement::Function {
        name,
        parameters: bump.alloc_slice_copy(&parameters),
        parameter_names: bump.alloc_slice_fill_iter(parameters.iter().map(Name::slice)),
        body,
    })
}

fn statement<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    let token = tokens.peek()?;
    Ok(match token.kind {
        TokenKind::Print => print(bump, tokens)?,
        TokenKind::LBrace => Statement::Block(block(bump, tokens)?),
        TokenKind::If => if_statement(bump, tokens)?,
        TokenKind::While => while_loop(bump, tokens)?,
        TokenKind::For => for_loop(bump, tokens)?,
        _ => expression_statement(bump, tokens)?,
    })
}

fn print<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    tokens.consume(TokenKind::Print)?;
    let expr = expression(bump, tokens)?;
    tokens.consume(TokenKind::Semicolon)?;
    Ok(Statement::Print(expr))
}

fn block<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<&'a [Statement<'a>], Error<'a>> {
    tokens.consume(TokenKind::LBrace)?;
    let mut statements = Vec::new();
    loop {
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::RBrace) {
            tokens.consume(TokenKind::RBrace)?;
            break;
        }
        statements.push(declaration(bump, tokens)?);
    }
    Ok(bump.alloc_slice_copy(&statements))
}

fn if_statement<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    tokens.consume(TokenKind::If)?;
    tokens.consume(TokenKind::LParen)?;
    let condition = expression(bump, tokens)?;
    tokens.consume(TokenKind::RParen)?;
    let then = bump.alloc(statement(bump, tokens)?);
    let token = tokens.peek();
    let or_else = if matches!(token, Ok(Token { kind: TokenKind::Else, .. })) {
        tokens.consume(TokenKind::Else)?;
        Some(&*bump.alloc(statement(bump, tokens)?))
    }
    else {
        None
    };
    Ok(Statement::If { condition, then, or_else })
}

fn while_loop<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    tokens.consume(TokenKind::While)?;
    tokens.consume(TokenKind::LParen)?;
    let condition = expression(bump, tokens)?;
    tokens.consume(TokenKind::RParen)?;
    let body = statement(bump, tokens)?;
    Ok(Statement::While { condition, body: bump.alloc(body) })
}

fn for_loop<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Statement<'a>, Error<'a>> {
    tokens.consume(TokenKind::For)?;
    tokens.consume(TokenKind::LParen)?;
    let token = tokens.peek()?;
    let init = if matches!(token.kind, TokenKind::Semicolon) {
        tokens.consume(TokenKind::Semicolon)?;
        None
    }
    else {
        let init = declaration(bump, tokens)?;
        match init {
            Statement::Expression(_) => (),
            Statement::Var(_, _) => (),
            _ => return Err(Error::InvalidForLoopInitialiser(init)),
        }
        Some(&*bump.alloc(init))
    };
    let token = tokens.peek()?;
    let condition = if matches!(token.kind, TokenKind::Semicolon) {
        None
    }
    else {
        Some(expression(bump, tokens)?)
    };
    tokens.consume(TokenKind::Semicolon)?;
    let token = tokens.peek()?;
    let update = if matches!(token.kind, TokenKind::RParen) {
        None
    }
    else {
        Some(expression(bump, tokens)?)
    };
    tokens.consume(TokenKind::RParen)?;
    let body = bump.alloc(statement(bump, tokens)?);
    Ok(Statement::For { init, condition, update, body })
}

fn expression_statement<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a>,
) -> Result<Statement<'a>, Error<'a>> {
    let result = Statement::Expression(expression(bump, tokens)?);
    tokens.consume(TokenKind::Semicolon)?;
    Ok(result)
}

pub fn expression<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a>,
) -> Result<Expression<'a>, Error<'a>> {
    expr_impl(bump, tokens, None)
}

fn expr_impl<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a>,
    left_op: Option<(Token<'a>, Operator)>,
) -> Result<Expression<'a>, Error<'a>> {
    let mut lhs = primary(bump, tokens)?;

    while let Ok(token) = tokens.peek() {
        let op = match token.kind {
            TokenKind::RParen => break,
            TokenKind::Semicolon => break,
            TokenKind::Comma => break,
            _ => BinOp::new(token)?,
        };

        match precedence(left_op.unzip().1, Operator::Infix(op.kind)) {
            Precedence::Left => break,
            Precedence::Right => (),
            Precedence::Ambiguous => Err(Error::AmbiguousPrecedences {
                left_op_token: left_op
                    .expect("ambiguous precedence is only possible when there is a left operand")
                    .0,
                right_op: op,
            })?,
        }
        // eat the `op` token
        let _ = tokens.next();
        let rhs = expr_impl(bump, tokens, Some((op.token, Operator::Infix(op.kind))))?;
        lhs =
            if matches!(op.kind, BinOpKind::Assign) {
                Expression::Assign {
                    target: as_assignment_target(lhs).map_err(|()| {
                        Error::InvalidAssignmentTarget { lhs, equal: op.token, rhs }
                    })?,
                    equal: op.token,
                    value: bump.alloc(rhs),
                }
            }
            else {
                Expression::Binary {
                    lhs: bump.alloc(lhs),
                    op,
                    rhs: bump.alloc(rhs),
                }
            };
    }

    Ok(lhs)
}

fn primary<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Expression<'a>, Error<'a>> {
    use TokenKind::*;
    let token = tokens.peek()?;
    let mut expr = match token.kind {
        LParen => grouping(bump, tokens)?,
        String | Number | True | False | Nil => literal(tokens)?,
        Minus | Bang => unary_op(bump, tokens)?,
        Identifier => ident(tokens)?,
        // TODO: could error with “expected expression” error here
        _ => unexpected_token_with_message("expression", token)?,
    };

    while let Ok(Token { kind: TokenKind::LParen, .. }) = tokens.peek() {
        let l_paren = tokens.consume(TokenKind::LParen)?;
        let arguments = call_arguments(bump, tokens)?;
        let r_paren = tokens.consume(TokenKind::RParen)?;
        expr = Expression::Call {
            callee: bump.alloc(expr),
            l_paren,
            arguments,
            r_paren,
        };
    }

    Ok(expr)
}

fn call_arguments<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a>,
) -> Result<&'a [Expression<'a>], Error<'a>> {
    let mut arguments = Vec::new();

    loop {
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::RParen) {
            break;
        }
        arguments.push(expression(bump, tokens)?);
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::Comma) {
            tokens.consume(TokenKind::Comma)?;
        }
        else {
            break;
        }
    }

    Ok(bump.alloc_slice_copy(&arguments))
}

fn grouping<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Expression<'a>, Error<'a>> {
    let l_paren = tokens.consume(TokenKind::LParen)?;
    let expr = expression(bump, tokens)?;
    let r_paren = tokens.consume(TokenKind::RParen)?;
    Ok(Expression::Grouping { l_paren, r_paren, expr: bump.alloc(expr) })
}

fn unary_op<'a>(bump: &'a Bump, tokens: &mut Tokens<'a>) -> Result<Expression<'a>, Error<'a>> {
    let token = tokens.consume_one_of(&[TokenKind::Minus, TokenKind::Bang])?;
    let kind = match token.kind {
        TokenKind::Minus => UnaryOpKind::Minus,
        TokenKind::Bang => UnaryOpKind::Not,
        _ => unreachable!(),
    };
    let operand = expr_impl(bump, tokens, Some((token, Operator::Prefix(kind))))?;
    Ok(Expression::Unary(
        UnaryOp { kind, token },
        bump.alloc(operand),
    ))
}

fn literal<'a>(tokens: &mut Tokens<'a>) -> Result<Expression<'a>, Error<'a>> {
    use TokenKind::*;
    let token = tokens.next()?;
    let kind = match token.kind {
        Number => LiteralKind::Number(
            token
                .slice()
                .parse()
                .expect("lexer makes sure this is a valid f64 literal"),
        ),
        String => LiteralKind::String(&token.slice()[1..token.slice().len() - 1]),
        True => LiteralKind::True,
        False => LiteralKind::False,
        Nil => LiteralKind::Nil,
        _ => unexpected_token(&[Number, String, True, False, Nil], token)?,
    };
    Ok(Expression::Literal(Literal { kind, token }))
}

fn ident<'a>(tokens: &mut Tokens<'a>) -> Result<Expression<'a>, Error<'a>> {
    Ok(Expression::Ident(Name(
        tokens.consume(TokenKind::Identifier)?,
    )))
}

fn unexpected_token<'a>(expected: &'static [TokenKind], actual: Token<'a>) -> Result<!, Error<'a>> {
    Err(Error::UnexpectedToken { expected, actual })
}

fn unexpected_token_with_message<'a>(
    expected: &'static str,
    actual: Token<'a>,
) -> Result<!, Error<'a>> {
    Err(Error::UnexpectedTokenMsg { expected, actual })
}

fn as_assignment_target(lhs: Expression) -> Result<Name, ()> {
    if let Expression::Ident(name) = lhs {
        Ok(name)
    }
    else {
        Err(())
    }
}

trait NotEof<'a>
where
    Self: 'a,
{
    fn not_eof(self) -> Result<Token<'a>, Eof>;
}

impl<'a> NotEof<'a> for Option<Token<'a>> {
    fn not_eof(self) -> Result<Token<'a>, Eof> {
        self.ok_or(Eof)
    }
}

#[derive(Debug, Clone, Copy)]
enum Operator {
    Prefix(#[expect(unused)] UnaryOpKind),
    Infix(BinOpKind),
}

#[derive(Debug, Clone, Copy)]
enum Precedence {
    Left,
    Right,
    Ambiguous,
}

fn precedence(left: Option<Operator>, right: Operator) -> Precedence {
    use BinOpKind::*;
    use Operator::*;
    use Precedence::*;
    match left {
        None => Right,
        Some(Prefix(_)) => match right {
            Prefix(_) => unreachable!(),
            Infix(Power) => Right,
            Infix(_) => Left,
        },
        Some(Infix(Plus | Minus)) => match right {
            Prefix(_) => Right,
            Infix(Or | And | Plus | Minus) => Left,
            Infix(Times | Divide | Power) => Right,
            Infix(EqualEqual | NotEqual | Less | LessEqual | Greater | GreaterEqual) => Left,
            Infix(Assign) => Ambiguous,
        },
        Some(Infix(Times | Divide)) => match right {
            Prefix(_) => Right,
            Infix(Or | And | Plus | Minus | Times | Divide) => Left,
            Infix(Power) => Right,
            Infix(EqualEqual | NotEqual | Less | LessEqual | Greater | GreaterEqual) => Left,
            Infix(Assign) => Ambiguous,
        },
        Some(Infix(Power)) => match right {
            Prefix(_) => Right,
            Infix(Power) => Right,
            Infix(Or | And | Plus | Minus | Times | Divide) => Left,
            Infix(EqualEqual | NotEqual | Less | LessEqual | Greater | GreaterEqual) => Left,
            Infix(Assign) => Ambiguous,
        },
        Some(Infix(EqualEqual | NotEqual | Less | LessEqual | Greater | GreaterEqual)) =>
            match right {
                Prefix(_) => Right,
                Infix(Or | And) => Left,
                Infix(Plus | Minus | Times | Divide | Power) => Right,
                Infix(EqualEqual | NotEqual | Less | LessEqual | Greater | GreaterEqual) =>
                    Ambiguous,
                Infix(Assign) => Ambiguous,
            },
        Some(Infix(Assign)) => Right,
        Some(Infix(Or)) => match right {
            Infix(Assign) => Left,
            _ => Right,
        },
        Some(Infix(And)) => match right {
            Infix(Assign | Or | And) => Left,
            _ => Right,
        },
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use rstest::rstest;

    use super::*;

    pub(crate) fn parse_str<'a>(bump: &'a Bump, src: &'a str) -> Result<Expression<'a>, Error<'a>> {
        let tokens = crate::lex(bump, "<test>", src).unwrap();
        parse(expression, bump, tokens)
    }

    #[rstest]
    #[case::unary_minus_minus("- - 5", "(- (- 5.0))")]
    #[case::unary_minus_not("- ! 5", "(- (! 5.0))")]
    #[case::minus_5_plus_3("-5 + 3", "(+ (- 5.0) 3.0)")]
    #[case::plus_plus("1 + 2 + 3", "(+ (+ 1.0 2.0) 3.0)")]
    #[case::plus_times("1 + 2 * 3", "(+ 1.0 (* 2.0 3.0))")]
    #[case::plus_times_minus("1 + 2 * 3 - 4", "(- (+ 1.0 (* 2.0 3.0)) 4.0)")]
    #[case::times_plus("1 * 2 + 3", "(+ (* 1.0 2.0) 3.0)")]
    #[case::three_plus_minus_five("3 + - 5", "(+ 3.0 (- 5.0))")]
    #[case::eq("1 == 2", "(== 1.0 2.0)")]
    #[case::power_power_power("1 ** 2 ** 3 ** 4", "(** 1.0 (** 2.0 (** 3.0 4.0)))")]
    #[case::group_power_power("(1 + 2) ** 3 ** 4", "(** (group (+ 1.0 2.0)) (** 3.0 4.0))")]
    #[case::times_power("1 * 2 ** 3", "(* 1.0 (** 2.0 3.0))")]
    #[case::power_times("1 ** 2 * 3", "(* (** 1.0 2.0) 3.0)")]
    #[case::unary_minus_power("-2 ** 3", "(- (** 2.0 3.0))")]
    #[case::power_unary_minus("2 ** -3", "(** 2.0 (- 3.0))")]
    #[case::unary_minus_times("-2 * 3", "(* (- 2.0) 3.0)")]
    #[case::unary_minus_eq("-2 == 3", "(== (- 2.0) 3.0)")]
    #[case::unary_minus_eq_unary_minus("-2 == -3", "(== (- 2.0) (- 3.0))")]
    #[case::eq_unary_minus("2 == -3", "(== 2.0 (- 3.0))")]
    #[case::div_div("1 / 2 / 3", "(/ (/ 1.0 2.0) 3.0)")]
    #[case::plus_eq("1 + 2 == 3", "(== (+ 1.0 2.0) 3.0)")]
    #[case::eq_plus("1 == 2 + 3", "(== 1.0 (+ 2.0 3.0))")]
    #[case::minus_unary_minus("1--1", "(- 1.0 (- 1.0))")]
    #[case::string_concat(r#""a" + "b""#, r#"(+ "a" "b")"#)]
    #[case::bool_conjunction("true < false", "(< true false)")]
    #[case::nil("nil", "nil")]
    #[case::comparison_of_parenthesised_gt(
        "(1 > 2) == (3 > 4)",
        "(== (group (> 1.0 2.0)) (group (> 3.0 4.0)))"
    )]
    #[case::assign_1_plus_2("a = 1 + 2", "(= a (+ 1.0 2.0))")]
    #[case::assign_assign("a = b = c", "(= a (= b (name c)))")]
    #[case::one_plus_two_and_three("1 + 2 and 3", "(and (+ 1.0 2.0) 3.0)")]
    #[case::and_or("1 and 2 or 3", "(or (and 1.0 2.0) 3.0)")]
    #[case::or_and("1 or 2 and 3", "(or 1.0 (and 2.0 3.0))")]
    #[case::call("f()", "(call (name f))")]
    #[case::call_call("f()()", "(call (call (name f)))")]
    #[case::call_call_with_args("f(1, 2)(3, 4)", "(call (call (name f) 1.0 2.0) 3.0 4.0)")]
    #[case::call_with_trailing_comma("f(1, 2,)", "(call (name f) 1.0 2.0)")]
    fn test_parser(#[case] src: &str, #[case] expected: &str) {
        let bump = &Bump::new();
        pretty_assertions::assert_eq!(parse_str(bump, src).unwrap().as_sexpr(), expected);
    }

    macro_rules! check {
        ($body:expr) => {
            for<'a> |result: Result<Expression<'a>, Error<'a>>| -> () {
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
    #[case::eq_eq(
        "1 == 2 == 3",
        check_err!(Error::AmbiguousPrecedences { .. }),
    )]
    #[case::eof_after_operator("1 -", check_err!(Error::Eof(_)))]
    #[case::three_adjecent_numbers(
        "1 2 3",
        check_err!(Error::UnexpectedTokenMsg {
            expected: "a binary operator",
            actual: Token { kind: TokenKind::Number, .. },
        }),
    )]
    #[case::expect_expr_in_parens(
        "()",
        check_err!(Error::UnexpectedTokenMsg { expected: "expression", .. }),
    )]
    #[case::unclosed_paren("(1 + 2", check_err!(Error::Eof(_)))]
    #[case::one_plus("1+", check_err!(Error::Eof(_)))]
    #[case::comparison_of_gt("1 > 2 == 3 > 4", check_err!(Error::AmbiguousPrecedences { .. }))]
    #[case::comparison_of_gt("1 == 3 > 4", check_err!(Error::AmbiguousPrecedences { .. }))]
    #[case::comparison_of_gt("1 > 2 == 3", check_err!(Error::AmbiguousPrecedences { .. }))]
    fn test_parse_error(
        #[case] src: &str,
        #[case] expected: impl for<'a> FnOnce(Result<Expression<'a>, Error<'a>>),
    ) {
        let bump = &Bump::new();
        expected(parse_str(bump, src));
    }
}
