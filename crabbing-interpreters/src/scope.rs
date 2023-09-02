use std::collections::HashMap;
use std::fmt::Write;
use std::primitive::usize;

use bumpalo::Bump;
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

impl Slot {
    fn as_sexpr(&self) -> String {
        match self {
            Slot::Local(slot) => format!("(local {slot})"),
            Slot::Global(slot) => format!("(global {slot})"),
        }
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
        for name in global_names {
            scopes.add_str(name);
        }
        scopes
    }

    fn add(&mut self, name: &Name<'a>) -> usize {
        self.add_str(name.slice())
    }

    fn add_str(&mut self, name: &'a str) -> usize {
        // FIXME: return `DuplicateNameError` if `name` is already present in innermost
        // scope
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
        };
        result.lines().fold(String::new(), |mut s, line| {
            writeln!(s, "{0:indent$}{line}", "").unwrap();
            s
        })
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
    Global(&'a Name<'a>, usize),
    Assign {
        target_name: &'a Name<'a>,
        target: Slot,
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
    NameError(&'a Name<'a>),
    AssignNameError {
        target_name: ExpressionTypes::NameError<'a>,
        equal: Token<'a>,
        value: &'a Expression<'a>,
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
            Expression::Global(name, _) => name.loc(),
            Expression::Assign { target_name, value, .. } => target_name.loc().until(value.loc()),
            Expression::Call { callee, r_paren, .. } => callee.loc().until(r_paren.loc()),
            Expression::NameError(name) => name.loc(),
            Expression::AssignNameError { target_name, value, .. } =>
                target_name.loc().until(value.loc()),
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
            Expression::Assign { target_name, target, value, .. } => format!(
                "(= {} {} {})",
                target_name.slice(),
                target.as_sexpr(),
                value.as_sexpr(),
            ),
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
            Expression::Global(name, slot) => format!("(global {} @{slot})", name.slice()),
            Expression::NameError(name) => format!("(name-error {})", name.slice()),
            Expression::AssignNameError { target_name, value, .. } => format!(
                "(=-name-error {} {})",
                target_name.slice(),
                value.as_sexpr()
            ),
        }
    }
}

pub(crate) fn resolve_names<'a>(
    bump: &'a Bump,
    global_names: &[&'a str],
    program: &'a [crate::parse::Statement<'a>],
) -> &'a [Statement<'a>] {
    let mut scopes = Scopes::new(global_names);
    // FIXME: The global scope actually has a special case in that it allows
    // capturing names in closures before they have been defined (using these
    // captured names yields a name error as usual).
    bump.alloc_slice_fill_iter(
        program
            .iter()
            .map(|stmt| resolve_stmt(bump, &mut scopes, stmt)),
    )
}

fn resolve_stmt<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    stmt: &'a parse::Statement<'a>,
) -> Statement<'a> {
    use parse::Statement::*;
    match stmt {
        Expression(expr) => Statement::Expression(resolve_expr(bump, scopes, expr)),
        Print(expr) => Statement::Print(resolve_expr(bump, scopes, expr)),
        Var(name, init) => {
            // FIXME: multiple definitions of the same name should be an error if not in
            // global scope
            let slot = scopes
                .lookup_local_innermost(name)
                .unwrap_or_else(|| scopes.add(name));
            Statement::Var(
                *name,
                slot,
                init.as_ref().map(|init| resolve_expr(bump, scopes, init)),
            )
        }
        Block(stmts) =>
            scopes.with_block(|scopes| {
                Statement::Block(bump.alloc_slice_fill_iter(
                    stmts.iter().map(|stmt| resolve_stmt(bump, scopes, stmt)),
                ))
            }),
        If { condition, then, or_else } => Statement::If {
            condition: resolve_expr(bump, scopes, condition),
            then: bump.alloc(resolve_stmt(bump, scopes, then)),
            or_else: or_else.map(|or_else| &*bump.alloc(resolve_stmt(bump, scopes, or_else))),
        },
        While { condition, body } => Statement::While {
            condition: resolve_expr(bump, scopes, condition),
            body: bump.alloc(resolve_stmt(bump, scopes, body)),
        },
        For { init, condition, update, body } => {
            let (init, condition, update, body) = scopes.with_block(|scopes| {
                let init = init.map(|init| &*bump.alloc(resolve_stmt(bump, scopes, init)));
                let (condition, update, body) = scopes.with_block(|scopes| {
                    (
                        condition
                            .as_ref()
                            .map(|condition| resolve_expr(bump, scopes, condition)),
                        update
                            .as_ref()
                            .map(|update| resolve_expr(bump, scopes, update)),
                        bump.alloc(resolve_stmt(bump, scopes, body)),
                    )
                });
                (init, condition, update, body)
            });
            Statement::For { init, condition, update, body }
        }
        Function { name, parameters, parameter_names, body } => {
            // FIXME: multiple definitions of the same name should be an error if not in
            // global scope
            let slot = scopes
                .lookup_local_innermost(name)
                .unwrap_or_else(|| scopes.add(name));
            Statement::Function {
                name: *name,
                slot,
                parameters,
                parameter_names,
                body: scopes.with_function(|scopes| {
                    for name in *parameters {
                        scopes.add(name);
                    }
                    bump.alloc_slice_fill_iter(
                        body.iter().map(|stmt| resolve_stmt(bump, scopes, stmt)),
                    )
                }),
            }
        }
    }
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
            Some(Slot::Global(slot)) => Expression::Global(name, slot),
            None => Expression::NameError(name),
        },
        Assign { target, equal, value } => match scopes.lookup(target) {
            Some(target_slot) => Expression::Assign {
                target_name: target,
                target: target_slot,
                equal: *equal,
                value: bump.alloc(resolve_expr(bump, scopes, value)),
            },
            None => Expression::AssignNameError {
                target_name: ExpressionTypes::NameError(target),
                equal: *equal,
                value: bump.alloc(resolve_expr(bump, scopes, value)),
            },
        },
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
