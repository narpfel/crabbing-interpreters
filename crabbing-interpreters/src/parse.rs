// TODO: reduce size of `Error`
#![expect(clippy::result_large_err)]

use std::iter::Peekable;

use ariadne::Color::Blue;
use ariadne::Color::Red;
use bumpalo::Bump;
use crabbing_interpreters_derive_report::Report;

use crate::interner::InternedString;
use crate::interner::Interner;
use crate::lex::Loc;
use crate::lex::Token;
use crate::lex::TokenIter;
use crate::lex::TokenKind;
use crate::scope::ErrorAtToken;
use crate::Sliced;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FunctionKind {
    Function,
    Method,
}

#[derive(Debug, Report)]
#[exit_code(65)]
pub enum Error<'a> {
    #[error("Unexpected end of file")]
    Eof {
        #[diagnostics(0(colour = Red, label = "end of file here"))]
        at: Eof<'a>,
    },

    #[error("Expected end of file")]
    ExpectedEof { at: ErrorAtToken<'a> },

    #[error("Expected of of the following tokens: {expected:?}")]
    UnexpectedToken {
        expected: &'static [TokenKind],
        at: ErrorAtToken<'a>,
    },

    #[error("at `{token}`: expected {expected:?}")]
    #[with(token = at.0)]
    UnexpectedTokenWithKind {
        expected: TokenKind,
        #[diagnostics(
            0(colour = Red),
        )]
        at: ErrorAtToken<'a>,
    },

    #[error("Expect {expected}.")]
    UnexpectedTokenMsg {
        expected: &'static str,
        #[diagnostics(0(colour = Red))]
        at: ErrorAtToken<'a>,
    },

    #[error("ambiguous precedences between operators `{at}` and `{right_op}`")]
    #[with(right_op = at.1)]
    AmbiguousPrecedences {
        #[diagnostics(
            0(colour = Red),
            1(colour = Blue),
        )]
        at: ErrorAtToken<'a, BinOp<'a>>,
    },

    #[error("invalid assignment target: `{lhs}`")]
    #[with(
        lhs = at.1,
        expr_type = at.1.kind_name(),
    )]
    InvalidAssignmentTarget {
        #[diagnostics(
            0(colour = Red),
            1(colour = Blue, label = "only names can be assigned to, not {expr_type}s"),
        )]
        at: ErrorAtToken<'a, Expression<'a>>,
    },

    #[error("invalid `{at}` loop initialiser: {stmt_kind} statement")]
    #[with(stmt_kind = at.1.kind_name())]
    InvalidForLoopInitialiser {
        #[diagnostics(
            0(colour = Blue),
            1(colour = Red, label = "only expression statements and variable declarations are allowed here"),
        )]
        at: ErrorAtToken<'a, Statement<'a>>,
    },

    #[error("unterminated string literal")]
    UnterminatedStringLiteral {
        #[diagnostics(at(colour = Red))]
        at: crate::lex::Error<'a>,
    },
}

impl<'a> From<Eof<'a>> for Error<'a> {
    fn from(value: Eof<'a>) -> Self {
        Self::Eof { at: value }
    }
}

impl<'a> From<crate::lex::Error<'a>> for Error<'a> {
    fn from(value: crate::lex::Error<'a>) -> Self {
        Self::UnterminatedStringLiteral { at: value }
    }
}

#[derive(Debug)]
pub struct Eof<'a>(Loc<'a>);

impl<'a> Eof<'a> {
    fn loc(&self) -> Loc<'a> {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Statement<'a> {
    Expression {
        expr: Expression<'a>,
        semi: Token<'a>,
    },
    Print {
        print: Token<'a>,
        expr: Expression<'a>,
        semi: Token<'a>,
    },
    Var {
        var: Token<'a>,
        name: Name<'a>,
        init: Option<Expression<'a>>,
        semi: Token<'a>,
    },
    Block {
        open_brace: Token<'a>,
        stmts: &'a [Statement<'a>],
        close_brace: Token<'a>,
    },
    If {
        if_token: Token<'a>,
        condition: Expression<'a>,
        then: &'a Statement<'a>,
        or_else: Option<&'a Statement<'a>>,
    },
    While {
        while_token: Token<'a>,
        condition: Expression<'a>,
        body: &'a Statement<'a>,
    },
    For {
        for_token: Token<'a>,
        init: Option<&'a Statement<'a>>,
        condition: Option<Expression<'a>>,
        update: Option<Expression<'a>>,
        body: &'a Statement<'a>,
    },
    // FIXME: functions should know their stack frame’s size to make it possible to reserve enough
    // space before calling them and to drop the frame after the call returns
    Function {
        fun: Option<Token<'a>>,
        name: Name<'a>,
        parameters: &'a [Name<'a>],
        body: &'a [Statement<'a>],
        close_brace: Token<'a>,
    },
    Return {
        return_token: Token<'a>,
        expr: Option<Expression<'a>>,
        semi: Token<'a>,
    },
    Class {
        class: Token<'a>,
        name: Name<'a>,
        base: Option<Expression<'a>>,
        methods: &'a [Statement<'a>],
        close_brace: Token<'a>,
    },
}

impl<'a> Statement<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            Statement::Expression { expr, semi } => expr.loc().until(semi.loc()),
            Statement::Print { print, expr: _, semi } => print.loc().until(semi.loc()),
            Statement::Var { var, name: _, init: _, semi } => var.loc().until(semi.loc()),
            Statement::Block { open_brace, stmts: _, close_brace } =>
                open_brace.loc().until(close_brace.loc()),
            Statement::If { if_token, condition: _, then, or_else } =>
                if_token.loc().until(or_else.unwrap_or(then).loc()),
            Statement::While { while_token, condition: _, body } =>
                while_token.loc().until(body.loc()),
            Statement::For {
                for_token,
                init: _,
                condition: _,
                update: _,
                body,
            } => for_token.loc().until(body.loc()),
            Statement::Function {
                fun,
                name,
                parameters: _,
                body: _,
                close_brace,
            } => fun
                .map(|fun| fun.loc())
                .unwrap_or(name.loc())
                .until(close_brace.loc()),
            Statement::Return { return_token, expr: _, semi } =>
                return_token.loc().until(semi.loc()),
            Statement::Class {
                class,
                name: _,
                base: _,
                methods: _,
                close_brace,
            } => class.loc().until(close_brace.loc()),
        }
    }

    fn kind_name(&self) -> &'static str {
        match self {
            Statement::Expression { .. } => "expression",
            Statement::Print { .. } => "print",
            Statement::Var { .. } => "variable declaration",
            Statement::Block { .. } => "block",
            Statement::If { .. } => "if",
            Statement::While { .. } => "while",
            Statement::For { .. } => "for",
            Statement::Function { .. } => "function definition",
            Statement::Return { .. } => "return",
            Statement::Class { .. } => "class",
        }
    }
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
        target: AssignmentTarget<'a>,
        equal: Token<'a>,
        value: &'a Expression<'a>,
    },
    Call {
        callee: &'a Expression<'a>,
        l_paren: Token<'a>,
        arguments: &'a [Expression<'a>],
        r_paren: Token<'a>,
    },
    Attribute {
        lhs: &'a Expression<'a>,
        attribute: Name<'a>,
    },
    This(Name<'a>),
    Super {
        super_: Name<'a>,
        attribute: Name<'a>,
    },
}

impl<'a> Expression<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        match self {
            Expression::Literal(lit) => lit.loc(),
            Expression::Unary(op, expr) => op.token.loc().until(expr.loc()),
            Expression::Binary { lhs, rhs, .. } => lhs.loc().until(rhs.loc()),
            Expression::Grouping { l_paren, r_paren, .. } => l_paren.loc().until(r_paren.loc()),
            Expression::Ident(name) => name.loc(),
            Expression::Assign { target, value, .. } => target.loc().until(value.loc()),
            Expression::Call { callee, r_paren, .. } => callee.loc().until(r_paren.loc()),
            Expression::Attribute { lhs, attribute } => lhs.loc().until(attribute.loc()),
            Expression::This(this) => this.loc(),
            Expression::Super { super_, attribute } => super_.loc().until(attribute.loc()),
        }
    }

    pub(crate) fn slice(&self) -> &'a str {
        self.loc().slice()
    }

    fn kind_name(&self) -> &'static str {
        match self {
            Expression::Literal(_) => "literal",
            Expression::Unary(_, _) => "unary operation",
            Expression::Binary { .. } => "binary operation",
            Expression::Grouping { .. } => "parenthesised expression",
            Expression::Ident(_) => "name",
            Expression::Assign { .. } => "assignment",
            Expression::Call { .. } => "call expression",
            Expression::Attribute { .. } => "attribute access",
            Expression::This(_) => "this expression",
            Expression::Super { .. } => "super expression",
        }
    }

    #[cfg(test)]
    pub fn as_sexpr(&self) -> String {
        use itertools::Itertools;

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
                    .collect_vec()
                    .join(" ")
            ),
            Expression::Attribute { lhs, attribute } =>
                format!("(attr {} {})", lhs.as_sexpr(), attribute.slice()),
            Expression::This(_) => "(this)".to_string(),
            Expression::Super { super_: _, attribute } => format!("(super {})", attribute.slice()),
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
pub struct Name<'a> {
    id: InternedString,
    pub(crate) loc: Loc<'a>,
}

impl<'a> Name<'a> {
    pub(crate) fn new(id: InternedString, loc: &'a Loc<'a>) -> Self {
        Self { id, loc: *loc }
    }

    pub(crate) fn id(&self) -> InternedString {
        self.id
    }

    pub(crate) fn loc(&self) -> Loc<'a> {
        self.loc
    }

    pub(crate) fn loc_ref(&'a self) -> &'a Loc<'a> {
        &self.loc
    }

    pub(crate) fn slice(&self) -> &'a str {
        self.loc.slice()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AssignmentTarget<'a> {
    Name(Name<'a>),
    Attribute {
        lhs: &'a Expression<'a>,
        attribute: Name<'a>,
    },
}

impl<'a> AssignmentTarget<'a> {
    fn loc(&self) -> Loc<'a> {
        match self {
            AssignmentTarget::Name(name) => name.loc(),
            AssignmentTarget::Attribute { lhs, attribute } => lhs.loc().until(attribute.loc()),
        }
    }

    #[cfg(test)]
    fn slice(&self) -> &'a str {
        self.loc().slice()
    }
}

pub struct Tokens<'a, 'b> {
    iter: Peekable<TokenIter<'a>>,
    eof_loc: Loc<'a>,
    interner: &'b mut Interner<'a>,
}

impl<'a> Tokens<'a, '_> {
    fn next(&mut self) -> Result<Token<'a>, Error<'a>> {
        Ok(self.iter.next().transpose()?.not_eof(&self.eof_loc)?)
    }

    fn peek(&mut self) -> Result<Token<'a>, Error<'a>> {
        Ok(self
            .iter
            .peek()
            .copied()
            .transpose()?
            .not_eof(&self.eof_loc)?)
    }

    fn consume(&mut self, kind: TokenKind) -> Result<Token<'a>, Error<'a>> {
        let token = self.next()?;
        if token.kind == kind {
            Ok(token)
        }
        else {
            Err(Error::UnexpectedTokenWithKind {
                expected: kind,
                at: ErrorAtToken::at(token),
            })
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
        let token = self.iter.peek();
        if let Some(&token) = token {
            Err(Error::ExpectedEof { at: ErrorAtToken::at(token?) })
        }
        else {
            Ok(())
        }
    }
}

pub fn parse<'a, 'b, T>(
    parser: impl FnOnce(&'a Bump, &mut Tokens<'a, 'b>) -> Result<T, Error<'a>>,
    bump: &'a Bump,
    tokens: TokenIter<'a>,
    eof_loc: Loc<'a>,
    interner: &'b mut Interner<'a>,
) -> Result<T, Error<'a>> {
    let mut tokens = Tokens {
        iter: tokens.peekable(),
        eof_loc,
        interner,
    };
    let result = parser(bump, &mut tokens)?;
    tokens.eof()?;
    Ok(result)
}

pub fn program<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<&'a [Statement<'a>], Error<'a>> {
    let mut statements = Vec::new();

    while tokens.peek().is_ok() {
        statements.push(declaration(bump, tokens)?);
    }

    tokens.eof()?;
    Ok(bump.alloc_slice_copy(&statements))
}

fn declaration<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<Statement<'a>, Error<'a>> {
    let token = tokens.peek()?;
    Ok(match token.kind {
        TokenKind::Class => class(bump, tokens)?,
        TokenKind::Var => vardecl(bump, tokens)?,
        TokenKind::Fun => function(bump, tokens, FunctionKind::Function)?,
        _ => statement(bump, tokens)?,
    })
}

fn class<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Statement<'a>, Error<'a>> {
    let class = tokens.consume(TokenKind::Class)?;
    let name = name(tokens)?;
    let token = tokens.peek()?;
    let base = if matches!(token.kind, TokenKind::Less) {
        tokens.consume(TokenKind::Less)?;
        Some(ident(tokens)?)
    }
    else {
        None
    };
    tokens.consume(TokenKind::LBrace)?;
    let mut methods = Vec::new();
    loop {
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::RBrace) {
            break;
        }
        methods.push(function(bump, tokens, FunctionKind::Method)?);
    }
    let close_brace = tokens.consume(TokenKind::RBrace)?;
    Ok(Statement::Class {
        class,
        name,
        base,
        methods: bump.alloc_slice_copy(&methods),
        close_brace,
    })
}

fn vardecl<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Statement<'a>, Error<'a>> {
    let var = tokens.consume(TokenKind::Var)?;
    let name = name(tokens)?;
    let maybe_equal = tokens.peek()?;
    let initialiser = if matches!(maybe_equal.kind, TokenKind::Equal) {
        tokens.consume(TokenKind::Equal)?;
        let initialiser = expression(bump, tokens)?;
        Some(initialiser)
    }
    else {
        None
    };
    let semi = tokens.consume(TokenKind::Semicolon)?;
    Ok(Statement::Var { var, name, init: initialiser, semi })
}

fn function<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
    kind: FunctionKind,
) -> Result<Statement<'a>, Error<'a>> {
    let fun = match kind {
        FunctionKind::Function => Some(tokens.consume(TokenKind::Fun)?),
        FunctionKind::Method => None,
    };
    let function_name = name(tokens)?;
    tokens.consume(TokenKind::LParen)?;
    let mut parameters = Vec::new();
    loop {
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::RParen) {
            break;
        }
        if matches!(token.kind, TokenKind::Identifier) {
            parameters.push(name(tokens)?);
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
    let (_, body, close_brace) = block(bump, tokens)?;
    Ok(Statement::Function {
        fun,
        name: function_name,
        parameters: bump.alloc_slice_copy(&parameters),
        body,
        close_brace,
    })
}

fn statement<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Statement<'a>, Error<'a>> {
    let token = tokens.peek()?;
    Ok(match token.kind {
        TokenKind::Print => print(bump, tokens)?,
        TokenKind::LBrace => {
            let (open_brace, stmts, close_brace) = block(bump, tokens)?;
            Statement::Block { open_brace, stmts, close_brace }
        }
        TokenKind::If => if_statement(bump, tokens)?,
        TokenKind::While => while_loop(bump, tokens)?,
        TokenKind::For => for_loop(bump, tokens)?,
        TokenKind::Return => return_stmt(bump, tokens)?,
        _ => expression_statement(bump, tokens)?,
    })
}

fn print<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Statement<'a>, Error<'a>> {
    let print = tokens.consume(TokenKind::Print)?;
    let expr = expression(bump, tokens)?;
    let semi = tokens.consume(TokenKind::Semicolon)?;
    Ok(Statement::Print { print, expr, semi })
}

fn block<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<(Token<'a>, &'a [Statement<'a>], Token<'a>), Error<'a>> {
    let open_brace = tokens.consume(TokenKind::LBrace)?;
    let mut statements = Vec::new();
    let close_brace = loop {
        let token = tokens.peek()?;
        if matches!(token.kind, TokenKind::RBrace) {
            break tokens.consume(TokenKind::RBrace)?;
        }
        statements.push(declaration(bump, tokens)?);
    };
    Ok((open_brace, bump.alloc_slice_copy(&statements), close_brace))
}

fn if_statement<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<Statement<'a>, Error<'a>> {
    let if_token = tokens.consume(TokenKind::If)?;
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
    Ok(Statement::If { if_token, condition, then, or_else })
}

fn while_loop<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Statement<'a>, Error<'a>> {
    let while_token = tokens.consume(TokenKind::While)?;
    tokens.consume(TokenKind::LParen)?;
    let condition = expression(bump, tokens)?;
    tokens.consume(TokenKind::RParen)?;
    let body = statement(bump, tokens)?;
    Ok(Statement::While {
        while_token,
        condition,
        body: bump.alloc(body),
    })
}

fn for_loop<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Statement<'a>, Error<'a>> {
    let for_token = tokens.consume(TokenKind::For)?;
    tokens.consume(TokenKind::LParen)?;
    let token = tokens.peek()?;
    let init = if matches!(token.kind, TokenKind::Semicolon) {
        tokens.consume(TokenKind::Semicolon)?;
        None
    }
    else {
        let init = declaration(bump, tokens)?;
        match init {
            Statement::Expression { .. } => (),
            Statement::Var { .. } => (),
            _ =>
                return Err(Error::InvalidForLoopInitialiser { at: ErrorAtToken(for_token, init) }),
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
    Ok(Statement::For { for_token, init, condition, update, body })
}

fn return_stmt<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<Statement<'a>, Error<'a>> {
    let return_token = tokens.consume(TokenKind::Return)?;
    let token = tokens.peek()?;
    let expr = if matches!(token.kind, TokenKind::Semicolon) {
        None
    }
    else {
        Some(expression(bump, tokens)?)
    };
    let semi = tokens.consume(TokenKind::Semicolon)?;
    Ok(Statement::Return { return_token, expr, semi })
}

fn expression_statement<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<Statement<'a>, Error<'a>> {
    Ok(Statement::Expression {
        expr: expression(bump, tokens)?,
        semi: tokens.consume(TokenKind::Semicolon)?,
    })
}

pub fn expression<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
) -> Result<Expression<'a>, Error<'a>> {
    expr_impl(bump, tokens, None)
}

fn expr_impl<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
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
                at: ErrorAtToken(
                    left_op
                        .expect(
                            "ambiguous precedence is only possible when there is a left operand",
                        )
                        .0,
                    op,
                ),
            })?,
        }
        // eat the `op` token
        let _ = tokens.next();
        let rhs = expr_impl(bump, tokens, Some((op.token, Operator::Infix(op.kind))))?;
        lhs = if matches!(op.kind, BinOpKind::Assign) {
            Expression::Assign {
                target: as_assignment_target(lhs).map_err(|()| Error::InvalidAssignmentTarget {
                    at: ErrorAtToken(op.token, lhs),
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

fn primary<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Expression<'a>, Error<'a>> {
    use TokenKind::*;
    let token = tokens.peek()?;
    let mut expr = match token.kind {
        LParen => grouping(bump, tokens)?,
        String | Number | True | False | Nil => literal(tokens)?,
        Minus | Bang => unary_op(bump, tokens)?,
        Identifier => ident(tokens)?,
        This => {
            let this = tokens.consume(This)?;
            Expression::This(Name {
                id: tokens.interner.intern(this.slice()),
                loc: this.loc(),
            })
        }
        Super => {
            let super_ = tokens.consume(Super)?;
            let super_ = Name {
                id: tokens.interner.intern(super_.slice()),
                loc: super_.loc(),
            };
            tokens.consume(TokenKind::Dot)?;
            let attribute = name(tokens)?;
            Expression::Super { super_, attribute }
        }
        // TODO: could error with “expected expression” error here
        _ => unexpected_token_with_message("expression", token)?,
    };

    loop {
        expr = match tokens.peek() {
            Ok(Token { kind: TokenKind::LParen, .. }) => {
                let l_paren = tokens.consume(TokenKind::LParen)?;
                let arguments = call_arguments(bump, tokens)?;
                let r_paren = tokens.consume(TokenKind::RParen)?;
                Expression::Call {
                    callee: bump.alloc(expr),
                    l_paren,
                    arguments,
                    r_paren,
                }
            }
            Ok(Token { kind: TokenKind::Dot, .. }) => {
                tokens.consume(TokenKind::Dot)?;
                let attribute = name(tokens)?;
                Expression::Attribute { lhs: bump.alloc(expr), attribute }
            }
            _ => break,
        }
    }

    Ok(expr)
}

fn call_arguments<'a>(
    bump: &'a Bump,
    tokens: &mut Tokens<'a, '_>,
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

fn grouping<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Expression<'a>, Error<'a>> {
    let l_paren = tokens.consume(TokenKind::LParen)?;
    let expr = expression(bump, tokens)?;
    let r_paren = tokens.consume(TokenKind::RParen)?;
    Ok(Expression::Grouping { l_paren, r_paren, expr: bump.alloc(expr) })
}

fn unary_op<'a>(bump: &'a Bump, tokens: &mut Tokens<'a, '_>) -> Result<Expression<'a>, Error<'a>> {
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

fn literal<'a>(tokens: &mut Tokens<'a, '_>) -> Result<Expression<'a>, Error<'a>> {
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

fn ident<'a>(tokens: &mut Tokens<'a, '_>) -> Result<Expression<'a>, Error<'a>> {
    Ok(Expression::Ident(name(tokens)?))
}

fn name<'a>(tokens: &mut Tokens<'a, '_>) -> Result<Name<'a>, Error<'a>> {
    let name_token = tokens.consume(TokenKind::Identifier)?;
    Ok(Name {
        id: tokens.interner.intern(name_token.slice()),
        loc: name_token.loc(),
    })
}

fn unexpected_token<'a>(expected: &'static [TokenKind], actual: Token<'a>) -> Result<!, Error<'a>> {
    Err(Error::UnexpectedToken { expected, at: ErrorAtToken::at(actual) })
}

fn unexpected_token_with_message<'a>(
    expected: &'static str,
    actual: Token<'a>,
) -> Result<!, Error<'a>> {
    Err(Error::UnexpectedTokenMsg { expected, at: ErrorAtToken::at(actual) })
}

fn as_assignment_target(lhs: Expression) -> Result<AssignmentTarget, ()> {
    match lhs {
        Expression::Ident(name) => Ok(AssignmentTarget::Name(name)),
        Expression::Attribute { lhs, attribute } =>
            Ok(AssignmentTarget::Attribute { lhs, attribute }),
        _ => Err(()),
    }
}

trait NotEof<'a>
where
    Self: 'a,
{
    fn not_eof(self, eof_loc: &Loc<'a>) -> Result<Token<'a>, Eof<'a>>;
}

impl<'a> NotEof<'a> for Option<Token<'a>> {
    fn not_eof(self, eof_loc: &Loc<'a>) -> Result<Token<'a>, Eof<'a>> {
        self.ok_or(Eof(*eof_loc))
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
    use std::path::Path;

    use rstest::rstest;

    use super::*;

    pub(crate) fn parse_str<'a>(bump: &'a Bump, src: &'a str) -> Result<Expression<'a>, Error<'a>> {
        let (tokens, eof_loc) = crate::lex(bump, Path::new("<test>"), src);
        parse(expression, bump, tokens, eof_loc, &mut Interner::default())
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
    #[case::eof_after_operator("1 -", check_err!(Error::Eof { at: _ }))]
    #[case::three_adjecent_numbers(
        "1 2 3",
        check_err!(Error::UnexpectedTokenMsg {
            expected: "a binary operator",
            at: ErrorAtToken(Token { kind: TokenKind::Number, .. }, ()),
        }),
    )]
    #[case::expect_expr_in_parens(
        "()",
        check_err!(Error::UnexpectedTokenMsg { expected: "expression", .. }),
    )]
    #[case::unclosed_paren("(1 + 2", check_err!(Error::Eof { at: _ }))]
    #[case::one_plus("1+", check_err!(Error::Eof { at: _ }))]
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
