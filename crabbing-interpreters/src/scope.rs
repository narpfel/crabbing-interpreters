use std::cell::Cell;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;
use std::iter::empty;
use std::iter::once;
use std::ptr;

use ariadne::Color::Blue;
use ariadne::Color::Red;
use bumpalo::Bump;
use crabbing_interpreters_derive_report::Report;
use indexmap::IndexMap;
use itertools::Either;
use itertools::Itertools as _;
use rustc_hash::FxHashMap as HashMap;
use variant_types_derive::derive_variant_types;

use crate::closure_compiler::compile_block;
use crate::closure_compiler::Execute;
use crate::interner::interned;
use crate::interner::InternedString;
use crate::lex::Loc;
use crate::lex::Token;
use crate::nonempty;
use crate::parse;
use crate::parse::BinOp;
use crate::parse::FunctionKind;
use crate::parse::Literal;
use crate::parse::Name;
use crate::parse::UnaryOp;
use crate::IndentLines;

const EMPTY: &str = "";

#[derive(Debug, Clone, Copy)]
pub(crate) enum HasBase {
    Yes,
    No,
}

impl From<HasBase> for bool {
    fn from(value: HasBase) -> Self {
        match value {
            HasBase::Yes => true,
            HasBase::No => false,
        }
    }
}

impl From<bool> for HasBase {
    fn from(value: bool) -> Self {
        if value {
            HasBase::Yes
        }
        else {
            HasBase::No
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FunctionKindWithBase {
    Function,
    Method(HasBase),
}

#[derive(Debug, Clone, Copy, Default)]
enum IsInit {
    #[default]
    No,
    Yes,
}

impl IsInit {
    fn then<T>(self, iterator: impl Iterator<Item = T>) -> impl Iterator<Item = T> {
        match self {
            IsInit::Yes => Either::Left(iterator),
            IsInit::No => Either::Right(iter::empty()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CellRef<'a> {
    Local(Variable<'a>),
    NonLocal(Variable<'a>),
}

impl<'a> CellRef<'a> {
    fn variable(&self) -> Variable<'a> {
        match self {
            Self::Local(var) => *var,
            Self::NonLocal(var) => *var,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct LocalScope<'a> {
    names: HashMap<InternedString, Variable<'a>>,
    layout: StackFrame<'a>,
}

#[derive(Debug, Default)]
struct Locals<'a>(nonempty::Vec<LocalScope<'a>>);

#[derive(Debug, Clone)]
enum FrameEntry<'a> {
    Local(Variable<'a>),
    Call(&'a Cell<usize>),
    ChildScope(StackFrame<'a>),
    FunctionScope(StackFrame<'a>),
}

impl FrameEntry<'_> {
    fn as_sexpr(&self, indent: usize) -> String {
        match self {
            FrameEntry::Local(variable) => variable.as_sexpr(),
            FrameEntry::Call(stack_size_at_callsite) =>
                format!("(call +{})", stack_size_at_callsite.get()),
            FrameEntry::ChildScope(scope) => scope.as_sexpr("block", indent),
            FrameEntry::FunctionScope(scope) => scope.as_sexpr("function", indent),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct StackFrame<'a>(Vec<FrameEntry<'a>>);

impl<'a> StackFrame<'a> {
    fn iter(&self) -> std::slice::Iter<'_, FrameEntry<'a>> {
        self.0.iter()
    }

    fn push(&mut self, entry: FrameEntry<'a>) {
        self.0.push(entry)
    }

    pub(crate) fn as_sexpr(&self, frame_type: &str, indent: usize) -> String {
        format!(
            "({frame_type}{}{})",
            if self.0.is_empty() { "" } else { "\n" },
            self.iter()
                .map(|entry| entry.as_sexpr(indent))
                .join("\n")
                .indent_lines(indent)
                .trim_end(),
        )
    }
}

#[derive(Debug, Default)]
struct Scope<'a> {
    is_init: IsInit,
    locals: Locals<'a>,
    cells: IndexMap<Variable<'a>, CellRef<'a>>,
}

#[derive(Debug, Clone, Copy)]
struct PointerCompare<'a, T>(&'a T);

impl<T> PartialEq for PointerCompare<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::from_ref(self.0) == ptr::from_ref(other.0)
    }
}

impl<T> Eq for PointerCompare<'_, T> {}

impl<T> Hash for PointerCompare<'_, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        ptr::from_ref(self.0).hash(state)
    }
}

#[derive(Debug)]
struct Scopes<'a> {
    bump: &'a Bump,
    scopes: nonempty::Vec<Scope<'a>>,
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Slot {
    Local(usize),
    Global(usize),
    Cell(usize),
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ty, slot) = match self {
            Slot::Local(slot) => ("local", slot),
            Slot::Global(slot) => ("global", slot),
            Slot::Cell(slot) => ("cell", slot),
        };
        write!(f, "({ty} @{slot})")
    }
}

#[derive(Debug, Report)]
#[exit_code(65)]
pub enum Error<'a> {
    #[error("Already a variable with this name in this scope: `{at}`")]
    DuplicateDefinition {
        #[diagnostics(loc(colour = Red))]
        at: Name<'a>,
    },

    #[error("Can’t return from top-level code.")]
    TopLevelReturn {
        #[diagnostics(0(colour = Red))]
        at: ErrorAtToken<'a>,
    },

    #[error("Initializer cannot return a value.")]
    ReturnValueInInit {
        #[diagnostics(
            0(colour = Blue, label = "cannot return a value"),
            1(colour = Red, label = "Help: remove this expression"),
        )]
        at: ErrorAtToken<'a, parse::Expression<'a>>,
    },

    #[error("Can’t use `{at}` outside of class.")]
    TopLevelThis {
        #[diagnostics(loc(colour = Red))]
        at: Name<'a>,
    },

    #[error("Can’t use `{at}` outside of class.")]
    TopLevelSuper {
        #[diagnostics(loc(colour = Red))]
        at: Name<'a>,
    },

    #[error("Can’t use `{at}` in a class with no bases.")]
    SuperInParentlessClass {
        #[diagnostics(loc(colour = Red))]
        at: Name<'a>,
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

impl<'a> Locals<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn first(&self) -> &LocalScope<'a> {
        self.0.first()
    }

    fn last(&self) -> &LocalScope<'a> {
        self.0.last()
    }

    fn last_mut(&mut self) -> &mut LocalScope<'a> {
        self.0.last_mut()
    }

    fn iter(&self) -> std::slice::Iter<'_, LocalScope<'a>> {
        self.0.iter()
    }

    fn push(&mut self) {
        self.0.push(Default::default())
    }

    fn pop(&mut self) {
        let LocalScope { names: _, layout } = self.0.pop().unwrap();
        self.last_mut().layout.push(FrameEntry::ChildScope(layout));
    }

    fn lookup(&self, name: &'a Name<'a>) -> Option<Variable<'a>> {
        self.iter()
            .rev()
            .find_map(|scope| scope.names.get(&name.id()))
            .cloned()
            .map(|variable| variable.at(name))
    }

    fn define(&mut self, variable: Variable<'a>) {
        let innermost_scope = self.last_mut();
        innermost_scope.layout.push(FrameEntry::Local(variable));
        let was_present = innermost_scope.names.insert(variable.name.id(), variable);
        debug_assert!(was_present.is_none());
    }
}

impl<'a> Scope<'a> {
    fn new(is_init: IsInit) -> Self {
        Self { is_init, ..Self::default() }
    }

    fn push(&mut self) {
        self.locals.push()
    }

    fn pop(&mut self) {
        self.locals.pop()
    }

    fn define(&mut self, variable: Variable<'a>) {
        self.locals.define(variable);
    }
}

impl<'a> Scopes<'a> {
    fn new(bump: &'a Bump, global_names: &'a [Name<'a>]) -> Self {
        let mut scopes = Scopes {
            bump,
            scopes: Default::default(),
            offset: Default::default(),
        };
        let duplicate_names = global_names
            .iter()
            .map(|name| name.slice())
            .duplicates()
            .collect_vec();
        assert!(
            duplicate_names.is_empty(),
            "duplicates in builtin global names are not permitted: {duplicate_names:?}",
        );
        for name in global_names {
            scopes.add_unconditionally(name);
        }
        scopes
    }

    fn add(&mut self, name: &'a Name<'a>) -> Result<Variable<'a>, Error<'a>> {
        if self.is_in_globals() {
            Ok(self
                .lookup_local_innermost(name.id())
                .unwrap_or_else(|| self.add_unconditionally(name)))
        }
        else if self.lookup_local_innermost(name.id()).is_some() {
            Err(Error::DuplicateDefinition { at: *name })
        }
        else {
            Ok(self.add_unconditionally(name))
        }
    }

    fn add_unconditionally(&mut self, name: &'a Name<'a>) -> Variable<'a> {
        let slot = self.offset;
        let variable = if self.is_in_globals() {
            Variable::global(self.bump, name, slot)
        }
        else {
            Variable::local(self.bump, name, slot)
        };
        self.scopes.last_mut().define(variable);
        self.offset += 1;
        variable
    }

    fn lookup(&mut self, name: &'a Name<'a>) -> Option<Variable<'a>> {
        let last = self.scopes.last();
        let cell_count = last.cells.len();
        match last.locals.lookup(name) {
            Some(slot) => Some(slot),
            None => {
                let mut nonlocal_scopes = self.scopes.iter().rev().skip(1);
                nonlocal_scopes
                    .find_map(|scope| scope.locals.lookup(name))
                    .map(|variable| {
                        if let Target::GlobalBySlot(_) | Target::GlobalByName = variable.target() {
                            variable
                        }
                        else {
                            self.scopes
                                .last_mut()
                                .cells
                                .entry(variable)
                                .or_insert_with(|| {
                                    CellRef::NonLocal(Variable::cell(self.bump, name, cell_count))
                                })
                                .variable()
                        }
                    })
            }
        }
    }

    fn local_to_cell(&mut self, variable: Variable<'a>) -> usize {
        if let Some(v) = self.lookup(variable.name) {
            match v.target() {
                Target::Local(_) => (),
                Target::GlobalByName => unreachable!(),
                Target::GlobalBySlot(_) => unreachable!(),
                Target::Cell(slot) => return slot,
            }
        }
        let last = self.scopes.last_mut();
        let cell_count = last.cells.len();
        last.cells.entry(variable).or_insert_with(|| {
            variable.make_cell(cell_count);
            CellRef::Local(variable)
        });
        cell_count
    }

    fn lookup_local_innermost(&self, name: InternedString) -> Option<Variable<'a>> {
        let last = &self.scopes.last().locals.last().names;
        last.get(&name).cloned()
    }

    fn with_block<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, Error<'a>>,
    ) -> Result<T, Error<'a>> {
        self.scopes.last_mut().push();
        let old_offset = self.offset;
        let result = f(self)?;
        self.offset = old_offset;
        self.scopes.last_mut().pop();
        Ok(result)
    }

    fn with_function(
        &mut self,
        is_init: IsInit,
        f: impl FnOnce(&mut Self) -> Result<(&'a [Variable<'a>], &'a [Statement<'a>]), Error<'a>>,
    ) -> Result<FunctionScope<'a>, Error<'a>> {
        self.push(is_init);
        let old_offset = std::mem::take(&mut self.offset);
        let (parameters, body) = f(self)?;
        let scope = self.pop();
        let locals = scope.locals;
        let layout = locals.last().layout.clone();
        self.scopes
            .last_mut()
            .locals
            .last_mut()
            .layout
            .push(FrameEntry::FunctionScope(layout));
        self.offset = old_offset;
        Ok(FunctionScope { parameters, body, cells: scope.cells })
    }

    fn current_stack_size(&self) -> usize {
        self.scopes
            .last()
            .locals
            .iter()
            .rev()
            .find_map(|scope| {
                scope
                    .names
                    .values()
                    .filter_map(|var| var.as_local())
                    .max()
                    .map(|n| n + 1)
            })
            .unwrap_or(0)
    }

    fn push(&mut self, is_init: IsInit) {
        self.scopes.push(Scope::new(is_init))
    }

    fn pop(&mut self) -> Scope<'a> {
        self.scopes.pop().unwrap()
    }

    fn is_in_globals(&self) -> bool {
        self.scopes.len() == 1 && self.scopes.last().locals.len() == 1
    }

    fn is_in_function(&self) -> bool {
        self.scopes.len() != 1
    }

    fn call(&mut self, stack_size_at_callsite: &'a Cell<usize>) {
        self.scopes
            .last_mut()
            .locals
            .last_mut()
            .layout
            .push(FrameEntry::Call(stack_size_at_callsite));
    }
}

struct FunctionScope<'a> {
    parameters: &'a [Variable<'a>],
    body: &'a [Statement<'a>],
    cells: IndexMap<Variable<'a>, CellRef<'a>>,
}

#[derive(Clone, Copy)]
pub struct Function<'a> {
    pub(crate) name: Name<'a>,
    pub(crate) parameters: &'a [Variable<'a>],
    pub(crate) body: &'a [Statement<'a>],
    pub(crate) cells: &'a [Option<usize>],
    pub(crate) compiled_body: &'a Cell<Either<&'a Bump, &'a Execute<'a>>>,
}

impl Debug for Function<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("name", &self.name)
            .field("parameters", &self.parameters)
            .field("body", &self.body)
            .field("cells", &self.cells)
            .finish()
    }
}

impl<'a> Function<'a> {
    pub(crate) fn compiled_body(&self) -> &'a Execute<'a> {
        match self.compiled_body.get() {
            Either::Right(compiled_body) => compiled_body,
            Either::Left(bump) => {
                let body = compile_block(bump, self.body);
                self.compiled_body.set(Either::Right(body));
                body
            }
        }
    }

    fn as_sexpr(&self, indent: usize, kind: FunctionKind, target: Option<Variable>) -> String {
        let Self {
            name,
            parameters,
            body,
            cells,
            compiled_body: _,
        } = self;
        let kind = match kind {
            FunctionKind::Function => "fun",
            FunctionKind::Method => "method",
        };
        format!(
            "({kind} {name} {}{}[{params}] {cells:?}\n{})",
            target.map_or(String::new(), |target| target.target().to_string()),
            if target.is_some() { " " } else { "" },
            Statement::Block(body).as_sexpr(indent).trim_end(),
            name = name.loc().slice(),
            params = parameters
                .iter()
                .map(|variable| variable.as_sexpr())
                .join(" "),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Statement<'a> {
    Expression(Expression<'a>),
    Print(Expression<'a>),
    Var(Variable<'a>, Option<Expression<'a>>),
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
        target: Variable<'a>,
        function: Function<'a>,
    },
    Return(Option<Expression<'a>>),
    // FIXME: This is not strictly necessary because it does the same as `Return(Some(this))`.
    InitReturn(Expression<'a>),
    Class {
        target: Variable<'a>,
        base: Option<Expression<'a>>,
        methods: &'a [Function<'a>],
    },
}

impl Statement<'_> {
    pub fn as_sexpr(&self, indent: usize) -> String {
        let result = match self {
            Statement::Expression(expr) => format!("(expr {})", expr.as_sexpr()),
            Statement::Print(expr) => format!("(print {})", expr.as_sexpr()),
            Statement::Var(target, init) => format!(
                "(var {} {} {})",
                target.name.slice(),
                target.target(),
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
            Statement::Function { target, function } =>
                function.as_sexpr(indent, FunctionKind::Function, Some(*target)),
            Statement::Return(expr) => format!(
                "(return {})",
                expr.map_or_else(|| "∅".to_string(), |expr| expr.as_sexpr()),
            ),
            Statement::InitReturn(_) => "(init-return)".to_owned(),
            Statement::Class { target, base, methods } => format!(
                "(class {} {} {}{}{})",
                target.name.slice(),
                target.target(),
                base.map(|base| base.as_sexpr())
                    .unwrap_or_else(|| "∅".to_string()),
                if methods.is_empty() { "" } else { "\n" },
                methods
                    .iter()
                    .map(|function| function.as_sexpr(indent, FunctionKind::Method, None))
                    .join("\n")
                    .indent_lines(indent)
                    .trim_end(),
            ),
        };
        result.indent_lines(indent)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Variable<'a> {
    pub(crate) name: &'a Name<'a>,
    target: TargetRef<'a>,
}

impl<'a> Variable<'a> {
    fn as_local(&self) -> Option<usize> {
        match self.target() {
            Target::Local(slot) => Some(slot),
            Target::GlobalBySlot(slot) => Some(slot),
            Target::GlobalByName => unreachable!(),
            Target::Cell(_) => None,
        }
    }

    pub fn target(&self) -> Target {
        self.target.get()
    }

    pub fn set_target(&self, target: Target) {
        self.target.set(target)
    }

    pub fn at(&self, name: &'a Name<'a>) -> Self {
        Self { name, target: self.target }
    }

    fn make_cell(&self, slot: usize) {
        self.set_target(Target::Cell(slot))
    }

    pub(crate) fn is_cell(&self) -> bool {
        matches!(self.target(), Target::Cell(_))
    }
}

impl PartialEq for Variable<'_> {
    fn eq(&self, other: &Self) -> bool {
        PointerCompare(self.target) == PointerCompare(other.target)
    }
}

impl Eq for Variable<'_> {}

impl Hash for Variable<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        PointerCompare(self.target).hash(state)
    }
}

impl<'a> Variable<'a> {
    fn local(bump: &'a Bump, name: &'a Name<'a>, slot: usize) -> Self {
        Self {
            name,
            target: bump.alloc(Cell::new(Target::Local(slot))),
        }
    }

    fn cell(bump: &'a Bump, name: &'a Name<'a>, slot: usize) -> Self {
        Self {
            name,
            target: bump.alloc(Cell::new(Target::Cell(slot))),
        }
    }

    fn global(bump: &'a Bump, name: &'a Name<'a>, slot: usize) -> Self {
        Self {
            name,
            target: bump.alloc(Cell::new(Target::GlobalBySlot(slot))),
        }
    }

    fn global_by_name(bump: &'a Bump, name: &'a Name<'a>) -> Self {
        Self {
            name,
            target: bump.alloc(Cell::new(Target::GlobalByName)),
        }
    }
}

type TargetRef<'a> = &'a Cell<Target>;

#[derive(Debug, Clone, Copy)]
pub enum Target {
    Local(usize),
    GlobalByName,
    GlobalBySlot(usize),
    Cell(usize),
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ty, slot) = match self {
            Target::Local(slot) => ("local", slot),
            Target::GlobalBySlot(slot) => ("global", slot),
            Target::GlobalByName => unreachable!(),
            Target::Cell(slot) => ("cell", slot),
        };
        write!(f, "({ty} @{slot})")
    }
}

impl<'a> Variable<'a> {
    fn as_sexpr(&self) -> String {
        let name = self.name.slice();
        match self.target() {
            Target::Local(slot) => format!("(local {name} @{slot})"),
            Target::GlobalByName => format!("(global-by-name {name})"),
            Target::GlobalBySlot(slot) => format!("(global {name} @{slot})"),
            Target::Cell(slot) => format!("(cell {name} @{slot})"),
        }
    }

    pub(crate) fn loc(&self) -> Loc<'a> {
        self.name.loc()
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
    Name(Variable<'a>),
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
        stack_size_at_callsite: &'a Cell<usize>,
    },
    Attribute {
        lhs: &'a Expression<'a>,
        attribute: Name<'a>,
    },
    Super {
        super_: Variable<'a>,
        this: Variable<'a>,
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
            Expression::Name(variable) => variable.name.loc(),
            Expression::Assign { target, value, .. } => target.loc().until(value.loc()),
            Expression::Call { callee, r_paren, .. } => callee.loc().until(r_paren.loc()),
            Expression::Attribute { lhs, attribute } => lhs.loc().until(attribute.loc()),
            Expression::Super { super_, this: _, attribute } => super_.loc().until(attribute.loc()),
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
                    .collect_vec()
                    .join(" "),
                stack_size_at_callsite = stack_size_at_callsite.get(),
            ),
            Expression::Name(variable) => variable.as_sexpr(),
            Expression::Attribute { lhs, attribute } =>
                format!("(attr {} {})", lhs.as_sexpr(), attribute.slice()),
            Expression::Super { super_, this, attribute } => format!(
                "(super {} {} {})",
                super_.as_sexpr(),
                this.as_sexpr(),
                attribute.slice(),
            ),
        }
    }
}

#[derive_variant_types]
#[derive(Debug, Clone, Copy)]
pub enum AssignmentTarget<'a> {
    Variable(Variable<'a>),
    Attribute {
        lhs: &'a Expression<'a>,
        attribute: Name<'a>,
    },
}

impl<'a> AssignmentTarget<'a> {
    fn as_sexpr(&self) -> String {
        match self {
            AssignmentTarget::Variable(variable) => variable.as_sexpr(),
            AssignmentTarget::Attribute { lhs, attribute } =>
                format!("(setattr {} {})", lhs.as_sexpr(), attribute.slice()),
        }
    }

    fn loc(&self) -> Loc<'a> {
        match self {
            AssignmentTarget::Variable(variable) => variable.loc(),
            AssignmentTarget::Attribute { lhs, attribute } => lhs.loc().until(attribute.loc()),
        }
    }

    fn slice(&self) -> &'a str {
        self.loc().slice()
    }
}

pub(crate) struct Program<'a> {
    pub(crate) stmts: &'a [Statement<'a>],
    pub(crate) global_name_offsets: HashMap<InternedString, Variable<'a>>,
    pub(crate) global_cell_count: usize,
    pub(crate) scopes: StackFrame<'a>,
}

impl Program<'_> {
    pub(crate) fn stmts_as_sexpr(&self, indent: usize) -> String {
        let mut sexpr = String::from("(program");
        if !self.stmts.is_empty() {
            sexpr.push('\n');
        }
        for stmt in self.stmts {
            sexpr.push_str(&stmt.as_sexpr(indent));
        }
        sexpr.truncate(sexpr.trim_end().len());
        sexpr.push(')');
        sexpr
    }
}

pub(crate) fn resolve_names<'a>(
    bump: &'a Bump,
    global_names: &'a [Name<'a>],
    program: &'a [crate::parse::Statement<'a>],
) -> Result<Program<'a>, Error<'a>> {
    let mut scopes = Scopes::new(bump, global_names);
    let stmts = &*bump.alloc_slice_copy(
        &program
            .iter()
            .map(|stmt| resolve_stmt(bump, &mut scopes, stmt))
            .collect::<Result<Vec<_>, _>>()?,
    );
    assert!(scopes.is_in_globals());
    assert!(scopes
        .scopes
        .first()
        .cells
        .values()
        .all(|cell_ref| matches!(cell_ref, CellRef::Local(_))));
    let LocalScope { names, layout } = scopes.scopes.first().locals.first().clone();
    iter_function_scopes(&layout).for_each(|scope| adjust_local_refs(0, scope));
    Ok(Program {
        stmts,
        global_name_offsets: names,
        global_cell_count: scopes.scopes.first().cells.len(),
        scopes: layout,
    })
}

fn resolve_stmt<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    stmt: &'a parse::Statement<'a>,
) -> Result<Statement<'a>, Error<'a>> {
    use parse::Statement::*;
    Ok(match stmt {
        Expression { expr, semi: _ } => Statement::Expression(resolve_expr(bump, scopes, expr)?),
        Print { print: _, expr, semi: _ } => Statement::Print(resolve_expr(bump, scopes, expr)?),
        Var { var: _, name, init, semi: _ } => {
            let init = init
                .as_ref()
                .map(|init| resolve_expr(bump, scopes, init))
                .transpose()?;
            let variable = scopes.add(name)?;
            Statement::Var(variable, init)
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
            condition: resolve_expr(bump, scopes, condition)?,
            then: bump.alloc(resolve_stmt(bump, scopes, then)?),
            or_else: or_else
                .map(|or_else| Ok(&*bump.alloc(resolve_stmt(bump, scopes, or_else)?)))
                .transpose()?,
        },
        While { while_token: _, condition, body } => Statement::While {
            condition: resolve_expr(bump, scopes, condition)?,
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
                            .map(|condition| resolve_expr(bump, scopes, condition))
                            .transpose()?,
                        update
                            .as_ref()
                            .map(|update| resolve_expr(bump, scopes, update))
                            .transpose()?,
                        bump.alloc(resolve_stmt(bump, scopes, body)?),
                    ))
                })?;
                Ok((init, condition, update, body))
            })?;
            Statement::For { init, condition, update, body }
        }
        function @ Function { name, .. } => {
            let target = scopes.add(name)?;
            let function =
                resolve_function(bump, scopes, function, FunctionKindWithBase::Function)?;
            Statement::Function { target, function }
        }
        Return { return_token, expr, semi: _ } =>
            if !scopes.is_in_function() {
                Err(Error::TopLevelReturn { at: ErrorAtToken::at(*return_token) })?
            }
            else if let IsInit::Yes = scopes.scopes.last().is_init {
                if let Some(expr) = expr {
                    Err(Error::ReturnValueInInit { at: ErrorAtToken(*return_token, *expr) })?
                }

                // FIXME: this allocates unnecessarily
                let loc = bump.alloc(return_token.loc());
                let ident = bump.alloc(parse::Expression::Ident(Name::new(interned::THIS, loc)));
                let this = resolve_expr(bump, scopes, ident)?;
                Statement::InitReturn(this)
            }
            else {
                Statement::Return(
                    expr.as_ref()
                        .map(|expr| resolve_expr(bump, scopes, expr))
                        .transpose()?,
                )
            },
        Class {
            class: _,
            name,
            base,
            methods,
            close_brace: _,
        } => {
            let base = base
                .as_ref()
                .map(|base| resolve_expr(bump, scopes, base))
                .transpose()?;
            let target = scopes.add(name)?;
            let methods = bump.alloc_slice_copy(
                &methods
                    .iter()
                    .map(|method| {
                        resolve_function(
                            bump,
                            scopes,
                            method,
                            FunctionKindWithBase::Method(base.is_some().into()),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            Statement::Class { target, base, methods }
        }
    })
}

fn resolve_expr<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    expr: &'a parse::Expression<'a>,
) -> Result<Expression<'a>, Error<'a>> {
    use parse::Expression::*;
    Ok(match expr {
        Literal(lit) => Expression::Literal(*lit),
        Unary(op, expr) => Expression::Unary(*op, bump.alloc(resolve_expr(bump, scopes, expr)?)),
        Binary { lhs, op, rhs } => Expression::Binary {
            lhs: bump.alloc(resolve_expr(bump, scopes, lhs)?),
            op: *op,
            rhs: bump.alloc(resolve_expr(bump, scopes, rhs)?),
        },
        Grouping { l_paren, expr, r_paren } => Expression::Grouping {
            l_paren: *l_paren,
            expr: bump.alloc(resolve_expr(bump, scopes, expr)?),
            r_paren: *r_paren,
        },
        Ident(name) => Expression::Name(
            scopes
                .lookup(name)
                .unwrap_or_else(|| Variable::global_by_name(bump, name)),
        ),
        Assign { target, equal, value } => {
            let target = match target {
                parse::AssignmentTarget::Name(name) => AssignmentTarget::Variable(
                    scopes
                        .lookup(name)
                        .unwrap_or_else(|| Variable::global_by_name(bump, name)),
                ),
                parse::AssignmentTarget::Attribute { lhs, attribute } =>
                    AssignmentTarget::Attribute {
                        lhs: bump.alloc(resolve_expr(bump, scopes, lhs)?),
                        attribute: *attribute,
                    },
            };
            Expression::Assign {
                target,
                equal: *equal,
                value: bump.alloc(resolve_expr(bump, scopes, value)?),
            }
        }
        Call { callee, l_paren, arguments, r_paren } => {
            let stack_size_at_callsite = bump.alloc(Cell::new(scopes.current_stack_size()));
            scopes.call(stack_size_at_callsite);
            Expression::Call {
                callee: bump.alloc(resolve_expr(bump, scopes, callee)?),
                l_paren: *l_paren,
                arguments: bump.alloc_slice_copy(
                    &arguments
                        .iter()
                        .map(|arg| resolve_expr(bump, scopes, arg))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
                r_paren: *r_paren,
                // FIXME: adding a `+ 1` here makes test 236 fail in miri. Why? Is there an
                // underlying issue with the GC?
                stack_size_at_callsite,
            }
        }
        Attribute { lhs, attribute } => Expression::Attribute {
            lhs: bump.alloc(resolve_expr(bump, scopes, lhs)?),
            attribute: *attribute,
        },
        This(this) => Expression::Name(
            scopes
                .lookup(this)
                .ok_or(Error::TopLevelThis { at: *this })?,
        ),
        Super { super_, attribute } => {
            let this = &*bump.alloc(Name::new(interned::THIS, super_.loc_ref()));
            let this = scopes
                .lookup(this)
                .ok_or(Error::TopLevelSuper { at: *super_ })?;
            let super_ = scopes
                .lookup(super_)
                .ok_or(Error::SuperInParentlessClass { at: *super_ })?;
            Expression::Super { super_, this, attribute: *attribute }
        }
    })
}

fn resolve_function<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    function: &'a parse::Statement<'a>,
    kind: FunctionKindWithBase,
) -> Result<Function<'a>, Error<'a>> {
    let parse::Statement::Function {
        fun: _,
        name,
        parameters,
        body,
        close_brace: _,
    } = function
    else {
        unreachable!()
    };
    let is_init = is_function_init(name, kind);
    let FunctionScope { parameters, body, cells } = scopes.with_function(is_init, |scopes| {
        let this = match kind {
            FunctionKindWithBase::Function => Either::Left(iter::empty()),
            FunctionKindWithBase::Method(has_base) => {
                if has_base.into() {
                    // FIXME: this is only needed if the method body actually uses `super`, however
                    // conservatively assuming `super` might be used is a one-time cost at class
                    // creation time
                    // FIXME: this allocates one `super` per method, also the source location is
                    // wrong
                    let super_ = bump.alloc(Loc::debug_loc(bump, "super"));
                    let super_ = &*bump.alloc(Name::new(interned::SUPER, super_));
                    let super_ = scopes.add(super_).unwrap();
                    scopes.local_to_cell(super_);
                }
                // FIXME: this allocates one `this` per method, also the source location is
                // wrong
                let this = bump.alloc(Loc::debug_loc(bump, "this"));
                let this = &*bump.alloc(Name::new(interned::THIS, this));
                Either::Right(iter::once(this))
            }
        };
        let variables: Vec<_> = this
            .clone()
            .chain(parameters.iter())
            .map(|name| scopes.add(name))
            .try_collect()?;

        let init_return = is_init
            .then(this)
            .map(|&this| {
                let this = bump.alloc(parse::Expression::Ident(this));
                Ok(Statement::InitReturn(resolve_expr(bump, scopes, this)?))
            })
            .next();

        Ok((
            bump.alloc_slice_copy(&variables),
            &*bump.alloc_slice_copy(
                &body
                    .iter()
                    .map(|stmt| resolve_stmt(bump, scopes, stmt))
                    .chain(init_return)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        ))
    })?;
    let cells = cell_slots(bump, scopes, &cells);
    Ok(Function {
        name: *name,
        parameters,
        body,
        cells,
        compiled_body: bump.alloc(Cell::new(Either::Left(bump))),
    })
}

fn cell_slots<'a>(
    bump: &'a Bump,
    scopes: &mut Scopes<'a>,
    nonlocal_names: &IndexMap<Variable<'a>, CellRef>,
) -> &'a [Option<usize>] {
    let cells = bump.alloc_slice_fill_copy(nonlocal_names.len(), None);
    for (variable, cell_ref) in nonlocal_names {
        match cell_ref {
            CellRef::Local(_) => (),
            CellRef::NonLocal(my_target) => {
                let (Target::Local(i) | Target::Cell(i)) = my_target.target()
                else {
                    unreachable!()
                };
                match variable.target() {
                    Target::Local(_) => {
                        let slot = scopes.local_to_cell(*variable);
                        cells[i] = Some(slot);
                    }
                    Target::GlobalByName => unreachable!(),
                    Target::GlobalBySlot(_) => unreachable!(),
                    Target::Cell(slot) => cells[i] = Some(slot),
                }
            }
        }
    }
    cells
}

fn is_function_init(name: &Name, kind: FunctionKindWithBase) -> IsInit {
    if name.id() == interned::INIT && matches!(kind, FunctionKindWithBase::Method(_)) {
        IsInit::Yes
    }
    else {
        IsInit::No
    }
}

fn iter_function_scopes<'a>(frame: &'a StackFrame<'a>) -> impl Iterator<Item = &'a StackFrame<'a>> {
    frame
        .iter()
        .flat_map(|frame_entry| -> Box<dyn Iterator<Item = _>> {
            match frame_entry {
                FrameEntry::Local(_) | FrameEntry::Call(_) => Box::new(empty()),
                FrameEntry::ChildScope(scope) => Box::new(iter_function_scopes(scope)),
                FrameEntry::FunctionScope(scope) =>
                    Box::new(once(scope).chain(iter_function_scopes(scope))),
            }
        })
}

fn adjust_local_refs(mut offset: usize, frame: &StackFrame) {
    for entry in frame.iter() {
        match entry {
            FrameEntry::Local(variable) =>
                if let Target::Local(_) = variable.target() {
                    variable.set_target(Target::Local(offset));
                    offset += 1;
                },
            FrameEntry::Call(stack_size_at_callsite) => stack_size_at_callsite.set(offset),
            FrameEntry::ChildScope(frame) => adjust_local_refs(offset, frame),
            FrameEntry::FunctionScope(frame) => adjust_local_refs(0, frame),
        }
    }
}
