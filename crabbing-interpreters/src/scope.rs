use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Write;
use std::primitive::usize;

use ariadne::Color::Red;
use bumpalo::Bump;
use crabbing_interpreters_derive_report::Report;
use itertools::Itertools as _;
use variant_types_derive::derive_variant_types;

use crate::lex::Loc;
use crate::lex::Token;
use crate::nonempty;
use crate::parse;
use crate::parse::BinOp;
use crate::parse::Literal;
use crate::parse::Name;
use crate::parse::UnaryOp;

const EMPTY: &str = "";

#[derive(Debug, Default)]
struct Locals<'a>(nonempty::Vec<HashMap<&'a str, usize>>);

#[derive(Debug, Default)]
struct Scope<'a> {
    locals: Locals<'a>,
}

#[derive(Debug, Default)]
struct Scopes<'a> {
    scopes: nonempty::Vec<Scope<'a>>,
    offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum Slot {
    Local(usize),
    Global(usize),
}

#[derive(Debug, Report)]
#[exit_code(65)]
pub enum Error<'a> {
    #[error("Already a variable with this name in this scope: `{at}`")]
    DuplicateDefinition {
        #[diagnostics(0(colour = Red))]
        at: Name<'a>,
    },

    #[error("Can’t return from top-level code.")]
    TopLevelReturn {
        #[diagnostics(0(colour = Red))]
        at: ErrorAtToken<'a>,
    },
}

#[derive(Debug)]
pub struct ErrorAtToken<'a, T = ()>(pub(crate) Token<'a>, pub(crate) T);

impl<'a> ErrorAtToken<'a> {
    pub fn at(token: Token<'a>) -> Self {
        Self(token, ())
    }
}

impl<'a, T> ErrorAtToken<'a, T> {
    pub fn slice(&self) -> &'a str {
        self.0.slice()
    }

    pub fn loc(&self) -> Loc<'a> {
        self.0.loc()
    }
}

impl Locals<'_> {
    fn push(&mut self) {
        self.0.push(Default::default())
    }

    fn pop(&mut self) {
        self.0.pop();
    }

    fn lookup(&self, name: &Name) -> Option<usize> {
        self.0
            .iter()
            .rev()
            .find_map(|scope| scope.get(name.slice()))
            .cloned()
    }
}

impl Scope<'_> {
    fn push(&mut self) {
        self.locals.push()
    }

    fn pop(&mut self) {
        self.locals.pop()
    }
}

impl<'a> Scopes<'a> {
    fn new(global_names: &[&'a str]) -> Self {
        let mut scopes = Scopes::default();
        let duplicate_names = global_names.iter().duplicates().collect_vec();
        assert!(
            duplicate_names.is_empty(),
            "duplicates in builtin global names are not permitted: {duplicate_names:?}",
        );
        for name in global_names {
            scopes.add_str(name);
        }
        scopes
    }

    fn add(&mut self, name: &Name<'a>) -> Result<usize, Error<'a>> {
        if self.is_in_globals() {
            return Ok(self
                .lookup_local_innermost(name)
                .unwrap_or_else(|| self.add_str(name.slice())));
        }
        else if self.lookup_local_innermost(name).is_some() {
            return Err(Error::DuplicateDefinition { at: *name });
        }
        else {
            Ok(self.add_str(name.slice()))
        }
    }

    fn add_str(&mut self, name: &'a str) -> usize {
        let last = self.scopes.last_mut().locals.0.last_mut();
        let slot = self.offset;
        last.insert(name, slot);
        self.offset += 1;
        slot
    }

    fn lookup(&self, name: &Name) -> Option<Slot> {
        let last = self.scopes.last();
        match last.locals.lookup(name) {
            Some(slot) => Some(Slot::Local(slot)),
            None => self.lookup_global(name),
        }
    }

    fn lookup_local_innermost(&self, name: &Name) -> Option<usize> {
        let last = self.scopes.last().locals.0.last();
        last.get(name.slice()).cloned()
    }

    fn lookup_global(&self, name: &Name) -> Option<Slot> {
        let globals = self.scopes.first();
        globals.locals.lookup(name).map(Slot::Global)
    }

    fn with_block<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.scopes.last_mut().push();
        let old_offset = self.offset;
        let result = f(self);
        self.offset = old_offset;
        self.scopes.last_mut().pop();
        result
    }

    fn with_function<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.push();
        let old_offset = std::mem::take(&mut self.offset);
        let result = f(self);
        self.pop();
        self.offset = old_offset;
        result
    }

    fn current_stack_size(&self) -> usize {
        self.scopes
            .last()
            .locals
            .0
            .iter()
            .rev()
            .find_map(|map| map.values().max().map(|n| n + 1))
            .unwrap_or(0)
    }

    fn push(&mut self) {
        self.scopes.push(Scope::default())
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn is_in_globals(&self) -> bool {
        self.scopes.len() == 1 && self.scopes.last().locals.0.len() == 1
    }

    fn is_in_function(&self) -> bool {
        self.scopes.len() != 1
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Statement<'a> {
    Expression(Expression<'a>),
    Print(Expression<'a>),
    Var(Name<'a>, usize, Option<Expression<'a>>),
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
    Function {
        name: Name<'a>,
        slot: usize,
        parameters: &'a [Name<'a>],
        parameter_names: &'a [&'a str],
        body: &'a [Statement<'a>],
    },
    Return(Option<Expression<'a>>),
}

impl Statement<'_> {
    pub fn as_sexpr(&self, indent: usize) -> String {
        let result = match self {
            Statement::Expression(expr) => format!("(expr {})", expr.as_sexpr()),
            Statement::Print(expr) => format!("(print {})", expr.as_sexpr()),
            Statement::Var(name, slot, init) => format!(
                "(var {} @{} {})",
                name.slice(),
                slot,
                init.map(|init| init.as_sexpr())
                    .unwrap_or_else(|| "∅".to_string()),
            ),
            Statement::Block(stmts) => format!(
                "(block{}{})",
                if stmts.is_empty() { "" } else { "\n" },
                stmts
                    .iter()
                    .map(|stmt| stmt.as_sexpr(indent))
                    .collect::<String>()
                    .trim_end(),
            ),
            Statement::If { condition, then, or_else } => format!(
                "(if\n{EMPTY:indent$}{}\n{}{})",
                condition.as_sexpr(),
                then.as_sexpr(indent),
                or_else
                    .map(|or_else| or_else.as_sexpr(indent))
                    .unwrap_or_else(|| format!("{EMPTY:indent$}∅\n"))
                    .trim_end(),
            ),
            Statement::While { condition, body } => format!(
                "(while\n{EMPTY:indent$}{}\n{})",
                condition.as_sexpr(),
                body.as_sexpr(indent).trim_end(),
            ),
            Statement::For { init, condition, update, body } => format!(
                "(for\n{}{EMPTY:indent$}{}\n{EMPTY:indent$}{}\n{})",
                init.map(|init| init.as_sexpr(indent))
                    .unwrap_or_else(|| format!("{EMPTY:indent$}∅\n")),
                condition
                    .map(|condition| condition.as_sexpr())
                    .unwrap_or_else(|| "∅".to_string()),
                update
                    .map(|update| update.as_sexpr())
                    .unwrap_or_else(|| "∅".to_string()),
                body.as_sexpr(indent).trim_end(),
            ),
            Statement::Function { name, slot, parameter_names, body, .. } => format!(
                "(fun {name} @{slot} [{params}]\n{})",
                Statement::Block(body).as_sexpr(indent).trim_end(),
                name = name.slice(),
                params = parameter_names.join(" "),
            ),
            Statement::Return(expr) => format!(
                "(return {})",
                expr.map_or_else(|| "∅".to_string(), |expr| expr.as_sexpr()),
            ),
        };
        result.lines().fold(String::new(), |mut s, line| {
            writeln!(s, "{0:indent$}{line}", "").unwrap();
            s
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GlobalName<'a> {
    ByName(&'a Name<'a>),
    BySlot(&'a Name<'a>, usize),
}

#[derive(Debug, Clone, Copy)]
pub enum AssignTarget<'a> {
    Local(&'a Name<'a>, usize),
    Global(&'a Cell<crate::scope::GlobalName<'a>>),
}

impl<'a> AssignTarget<'a> {
    fn as_sexpr(&self) -> String {
        match self {
            AssignTarget::Local(name, slot) => format!("(local {} @{slot})", name.slice()),
            AssignTarget::Global(target) => match target.get() {
                GlobalName::ByName(name) => format!("(global-by-name {})", name.slice()),
                GlobalName::BySlot(name, slot) => format!("(global {} @{slot})", name.slice()),
            },
        }
    }

    fn loc(&self) -> Loc<'a> {
        match self {
            AssignTarget::Local(name, _) => name.loc(),
            AssignTarget::Global(target) => match target.get() {
                GlobalName::ByName(name) | GlobalName::BySlot(name, _) => name.loc(),
            },
        }
    }
}

#[derive_variant_types]
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
    Local(&'a Name<'a>, usize),
    Global(&'a Cell<crate::scope::GlobalName<'a>>),
    Assign {
        target: AssignTarget<'a>,
        equal: Token<'a>,
        value: &'a Expression<'a>,
    },
    Call {
        callee: &'a Expression<'a>,
        l_paren: Token<'a>,
        arguments: &'a [Expression<'a>],
        r_paren: Token<'a>,
        stack_size_at_callsite: usize,
    },
}

impl<'a> Expression<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        match self {
            Expression::Literal(lit) => lit.loc(),
            Expression::Unary(op, expr) => op.token.loc().until(expr.loc()),
            Expression::Binary { lhs, rhs, .. } => lhs.loc().until(rhs.loc()),
            Expression::Grouping { l_paren, r_paren, .. } => l_paren.loc().until(r_paren.loc()),
            Expression::Local(name, _) => name.loc(),
            Expression::Global(global) => {
                let (GlobalName::ByName(name) | GlobalName::BySlot(name, _)) = global.get();
                name.loc()
            }
            Expression::Assign { target, value, .. } => target.loc().until(value.loc()),
            Expression::Call { callee, r_paren, .. } => callee.loc().until(r_paren.loc()),
        }
    }

    pub(crate) fn slice(&self) -> &'a str {
        self.loc().slice()
    }

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
            Expression::Assign { target, value, .. } =>
                format!("(= {} {})", target.as_sexpr(), value.as_sexpr()),
            Expression::Call {
                callee,
                arguments,
                stack_size_at_callsite,
                ..
            } => format!(
                "(call +{stack_size_at_callsite} {}{}{})",
                callee.as_sexpr(),
                if arguments.is_empty() { "" } else { " " },
                arguments
                    .iter()
                    .map(Expression::as_sexpr)
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            Expression::Local(name, slot) => format!("(local {} @{slot})", name.slice()),
            Expression::Global(global) => match global.get() {
                GlobalName::ByName(name) => format!("(global-by-name {})", name.slice()),
                GlobalName::BySlot(name, slot) => format!("(global {} @{slot})", name.slice()),
            },
        }
    }
}

pub(crate) fn resolve_names<'a>(
    bump: &'a Bump,
    global_names: &[&'a str],
    program: &'a [crate::parse::Statement<'a>],
) -> Result<(&'a [Statement<'a>], HashMap<&'a str, usize>), Error<'a>> {
    let mut scopes = Scopes::new(global_names);
    let stmts = &*bump.alloc_slice_copy(
        &program
            .iter()
            .map(|stmt| resolve_stmt(bump, &mut scopes, stmt))
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok((stmts, scopes.scopes.first().locals.0.first().clone()))
}

fn resolve_stmt<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    stmt: &'a parse::Statement<'a>,
) -> Result<Statement<'a>, Error<'a>> {
    use parse::Statement::*;
    Ok(match stmt {
        Expression { expr, semi: _ } => Statement::Expression(resolve_expr(bump, scopes, expr)),
        Print { print: _, expr, semi: _ } => Statement::Print(resolve_expr(bump, scopes, expr)),
        Var { var: _, name, init, semi: _ } => {
            let slot = scopes.add(name)?;
            Statement::Var(
                *name,
                slot,
                init.as_ref().map(|init| resolve_expr(bump, scopes, init)),
            )
        }
        Block { open_brace: _, stmts, close_brace: _ } => scopes.with_block(|scopes| {
            Ok(Statement::Block(
                bump.alloc_slice_copy(
                    &stmts
                        .iter()
                        .map(|stmt| resolve_stmt(bump, scopes, stmt))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            ))
        })?,
        If { if_token: _, condition, then, or_else } => Statement::If {
            condition: resolve_expr(bump, scopes, condition),
            then: bump.alloc(resolve_stmt(bump, scopes, then)?),
            or_else: or_else
                .map(|or_else| Ok(&*bump.alloc(resolve_stmt(bump, scopes, or_else)?)))
                .transpose()?,
        },
        While { while_token: _, condition, body } => Statement::While {
            condition: resolve_expr(bump, scopes, condition),
            body: bump.alloc(resolve_stmt(bump, scopes, body)?),
        },
        For {
            for_token: _,
            init,
            condition,
            update,
            body,
        } => {
            let (init, condition, update, body) = scopes.with_block(|scopes| {
                let init = init
                    .map(|init| Ok(&*bump.alloc(resolve_stmt(bump, scopes, init)?)))
                    .transpose()?;
                let (condition, update, body) = scopes.with_block(|scopes| {
                    Ok((
                        condition
                            .as_ref()
                            .map(|condition| resolve_expr(bump, scopes, condition)),
                        update
                            .as_ref()
                            .map(|update| resolve_expr(bump, scopes, update)),
                        bump.alloc(resolve_stmt(bump, scopes, body)?),
                    ))
                })?;
                Ok((init, condition, update, body))
            })?;
            Statement::For { init, condition, update, body }
        }
        Function {
            fun: _,
            name,
            parameters,
            parameter_names,
            body,
            close_brace: _,
        } => {
            let slot = scopes.add(name)?;
            Statement::Function {
                name: *name,
                slot,
                parameters,
                parameter_names,
                body: scopes.with_function(|scopes| {
                    for name in *parameters {
                        scopes.add(name)?;
                    }
                    Ok(bump.alloc_slice_copy(
                        &body
                            .iter()
                            .map(|stmt| resolve_stmt(bump, scopes, stmt))
                            .collect::<Result<Vec<_>, _>>()?,
                    ))
                })?,
            }
        }
        Return { return_token, expr, semi: _ } =>
            if !scopes.is_in_function() {
                Err(Error::TopLevelReturn { at: ErrorAtToken::at(*return_token) })?
            }
            else {
                Statement::Return(expr.as_ref().map(|expr| resolve_expr(bump, scopes, expr)))
            },
    })
}

fn resolve_expr<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    expr: &'a parse::Expression<'a>,
) -> Expression<'a> {
    use parse::Expression::*;
    match expr {
        Literal(lit) => Expression::Literal(*lit),
        Unary(op, expr) => Expression::Unary(*op, bump.alloc(resolve_expr(bump, scopes, expr))),
        Binary { lhs, op, rhs } => Expression::Binary {
            lhs: bump.alloc(resolve_expr(bump, scopes, lhs)),
            op: *op,
            rhs: bump.alloc(resolve_expr(bump, scopes, rhs)),
        },
        Grouping { l_paren, expr, r_paren } => Expression::Grouping {
            l_paren: *l_paren,
            expr: bump.alloc(resolve_expr(bump, scopes, expr)),
            r_paren: *r_paren,
        },
        Ident(name) => match scopes.lookup(name) {
            Some(Slot::Local(slot)) => Expression::Local(name, slot),
            Some(Slot::Global(slot)) =>
                Expression::Global(bump.alloc(Cell::new(GlobalName::BySlot(name, slot)))),
            None => Expression::Global(bump.alloc(Cell::new(GlobalName::ByName(name)))),
        },
        Assign { target, equal, value } => {
            let target = match scopes.lookup(target) {
                Some(Slot::Global(slot)) =>
                    AssignTarget::Global(bump.alloc(Cell::new(GlobalName::BySlot(target, slot)))),
                Some(Slot::Local(slot)) => AssignTarget::Local(target, slot),
                None => AssignTarget::Global(bump.alloc(Cell::new(GlobalName::ByName(target)))),
            };
            Expression::Assign {
                target,
                equal: *equal,
                value: bump.alloc(resolve_expr(bump, scopes, value)),
            }
        }
        Call { callee, l_paren, arguments, r_paren } => Expression::Call {
            callee: bump.alloc(resolve_expr(bump, scopes, callee)),
            l_paren: *l_paren,
            arguments: bump
                .alloc_slice_fill_iter(arguments.iter().map(|arg| resolve_expr(bump, scopes, arg))),
            r_paren: *r_paren,
            stack_size_at_callsite: scopes.current_stack_size(),
        },
    }
}
