use std::collections::BTreeSet;
use std::fmt;

use crate::bytecode::vm::stack::Stack;
use crate::bytecode::Bytecode;
use crate::bytecode::Bytecode::*;
use crate::bytecode::CallInner;
use crate::gc::Gc;
use crate::gc::GcStr;
use crate::nonempty;
use crate::parse::BinOp;
use crate::parse::BinOpKind;
use crate::parse::FunctionKind;
use crate::parse::LiteralKind;
use crate::parse::UnaryOpKind;
use crate::scope::AssignmentTarget;
use crate::scope::Expression;
use crate::scope::Function;
use crate::scope::Statement;
use crate::scope::Target;
use crate::scope::Variable;
use crate::value::nanboxed;
use crate::value::Value;

struct Compiler<'a> {
    gc: &'a Gc,
    code: Vec<Bytecode>,
    constants: Vec<nanboxed::Value<'a>>,
    metadata: Vec<Metadata<'a>>,
    error_locations: Vec<ContainingExpression<'a>>,
    defined_local_slots: Vec<LocalSlotsDefinedInFrame>,
    frame_stack: nonempty::Vec<BTreeSet<usize>>,
}

#[derive(Clone, Copy)]
pub enum Metadata<'a> {
    Function {
        function: &'a Function<'a>,
        code_size: u32,
    },
    Class {
        name: &'a str,
        methods: &'a [Function<'a>],
        base_error_location: Option<usize>,
    },
}

impl<'s> fmt::Debug for Metadata<'s> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metadata::Function { function, code_size } => f
                .debug_struct("Metadata::Function")
                .field_with("function", |f| {
                    write!(
                        f,
                        "<function {} arity={}>",
                        function.name.id(),
                        function.parameters.len(),
                    )
                })
                .field("code_size", code_size)
                .finish(),
            Metadata::Class { name, methods, base_error_location } => f
                .debug_struct("Metadata::Class")
                .field("name", name)
                .field_with("methods", |f| {
                    let mut list = f.debug_list();
                    methods.iter().for_each(|method| {
                        list.entry_with(|f| {
                            write!(
                                f,
                                "<method {} arity={}>",
                                method.name.id(),
                                method.parameters.len(),
                            )
                        });
                    });
                    list.finish()
                })
                .field("has_base", &base_error_location.is_some())
                .finish(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ContainingExpression<'a> {
    Enter { at: usize, expr: &'a Expression<'a> },
    Exit { at: usize },
}

impl ContainingExpression<'_> {
    pub(crate) fn at(&self) -> usize {
        match self {
            ContainingExpression::Enter { at, expr: _ } | ContainingExpression::Exit { at } => *at,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LocalSlotsDefinedInFrame {
    Enter { at: usize, defined: Vec<usize> },
    Exit { at: usize },
}

impl LocalSlotsDefinedInFrame {
    pub(crate) fn at(&self) -> usize {
        match self {
            LocalSlotsDefinedInFrame::Enter { at, defined: _ }
            | LocalSlotsDefinedInFrame::Exit { at } => *at,
        }
    }
}

#[expect(clippy::type_complexity, reason = "FIXME")]
pub(crate) fn compile_program<'a>(
    gc: &'a Gc,
    program: &'a [Statement<'a>],
) -> (
    Vec<Bytecode>,
    Vec<nanboxed::Value<'a>>,
    Vec<Metadata<'a>>,
    Vec<ContainingExpression<'a>>,
    Vec<LocalSlotsDefinedInFrame>,
) {
    let mut compiler = Compiler {
        gc,
        code: Vec::new(),
        constants: Vec::new(),
        metadata: Vec::new(),
        error_locations: Vec::new(),
        defined_local_slots: Vec::new(),
        frame_stack: nonempty::Vec::default(),
    };

    for stmt in program {
        compiler.compile_stmt(stmt);
    }

    compiler.code.push(End);

    let Compiler {
        gc: _,
        code,
        constants,
        metadata,
        error_locations,
        defined_local_slots,
        frame_stack,
    } = compiler;
    assert_eq!(frame_stack.len(), 1);
    (
        code,
        constants,
        metadata,
        error_locations,
        defined_local_slots,
    )
}

impl<'a> Compiler<'a> {
    fn compile_stmt(&mut self, stmt: &'a Statement<'a>) {
        self.enter_statement(self.code.len());
        match stmt {
            Statement::Expression(expr) => {
                self.compile_expr(expr);
                self.code.push(Pop);
            }
            Statement::Print(expr) => {
                self.compile_expr(expr);
                self.code.push(Print);
            }
            Statement::Var(variable, initialiser) => {
                if let Some(initialiser) = initialiser {
                    self.compile_expr(initialiser);
                }
                else {
                    self.code.push(ConstNil);
                }
                self.compile_define(variable);
            }
            Statement::Block(block) =>
                for stmt in *block {
                    self.compile_stmt(stmt)
                },
            Statement::If { condition, then, or_else } => {
                self.compile_expr(condition);
                let jump_index = self.code.len();
                self.code.push(if or_else.is_some() {
                    PopJumpIfFalse(0)
                }
                else {
                    JumpIfFalse(0)
                });
                self.compile_stmt(then);
                let jump_target = if let Some(or_else) = or_else {
                    let jump_index = self.code.len();
                    self.code.push(PopJumpIfTrue(0));
                    let jump_target = self.code.len();
                    self.compile_stmt(or_else);
                    self.code[jump_index] = PopJumpIfTrue(self.code.len().try_into().unwrap());
                    jump_target
                }
                else {
                    self.code.len()
                };
                self.code[jump_index] = if or_else.is_some() {
                    PopJumpIfFalse(jump_target.try_into().unwrap())
                }
                else {
                    JumpIfFalse(jump_target.try_into().unwrap())
                };
            }
            Statement::While { condition, body } => {
                let jump_index = self.code.len();
                self.code.push(Jump(0));
                let jump_target = self.code.len();
                self.compile_stmt(body);
                self.code[jump_index] = Jump(self.code.len().try_into().unwrap());
                self.compile_expr(condition);
                self.code.push(JumpIfTrue(jump_target.try_into().unwrap()));
            }
            Statement::For { init, condition, update, body } => {
                if let Some(init) = init {
                    self.compile_stmt(init);
                }

                let jump_index = self.code.len();
                if condition.is_some() {
                    self.code.push(Jump(0));
                }
                let jump_target = self.code.len();
                self.compile_stmt(body);
                if let Some(update) = update {
                    self.compile_expr(update);
                    self.code.push(Pop);
                }
                if let Some(condition) = condition {
                    self.code[jump_index] = Jump(self.code.len().try_into().unwrap());
                    self.compile_expr(condition);
                    self.code.push(JumpIfTrue(jump_target.try_into().unwrap()));
                }
                else {
                    self.code.push(Jump(jump_target.try_into().unwrap()));
                }
            }
            Statement::Function { target, function } => {
                if target.is_cell() {
                    self.code.push(ConstNil);
                    self.compile_define(target);
                }
                self.compile_function(function, FunctionKind::Function);
                self.compile_store(target);
            }
            Statement::Return(expr) => {
                if let Some(expr) = expr {
                    self.compile_expr(expr);
                }
                else {
                    self.code.push(ConstNil);
                }
                self.code.push(Return);
            }
            Statement::InitReturn(_) => {
                self.code.push(BoundMethodGetInstance);
                self.code.push(Return);
            }
            Statement::Class { target, base, methods } => {
                if target.is_cell() {
                    self.code.push(ConstNil);
                    self.compile_define(target);
                }
                let base_error_location = if let Some(base) = base {
                    self.compile_expr(base);
                    Some(self.code.len().checked_sub(1).unwrap())
                }
                else {
                    None
                };
                methods
                    .iter()
                    .for_each(|method| self.compile_function(method, FunctionKind::Method));
                let meta_index = self.metadata.len();
                self.metadata.push(Metadata::Class {
                    name: target.name.slice(),
                    methods,
                    base_error_location,
                });
                self.code.push(BuildClass(meta_index.try_into().unwrap()));
                self.compile_store(target);
            }
        }
        self.exit_statement(self.code.len());
    }

    fn compile_expr(&mut self, expr: &'a Expression<'a>) {
        self.enter_expression(self.code.len(), expr);
        match expr {
            Expression::Literal(literal) => {
                // FIXME: deduplicate constants
                match literal.kind {
                    LiteralKind::Number(x) => {
                        self.code.push(ConstNumber(x.into()));
                    }
                    LiteralKind::String(s) => {
                        self.constants
                            .push(Value::String(GcStr::new_in(self.gc, s)).into_nanboxed());
                        self.code
                            .push(Const((self.constants.len() - 1).try_into().unwrap()));
                    }
                    LiteralKind::True => self.code.push(ConstTrue),
                    LiteralKind::False => self.code.push(ConstFalse),
                    LiteralKind::Nil => self.code.push(ConstNil),
                }
            }
            Expression::Unary(operator, operand) => {
                self.compile_expr(operand);
                let op = match operator.kind {
                    UnaryOpKind::Minus => UnaryMinus,
                    UnaryOpKind::Not => UnaryNot,
                };
                self.code.push(op);
            }
            Expression::Binary {
                lhs,
                op: BinOp { kind: BinOpKind::And, .. },
                rhs,
            } => {
                self.compile_expr(lhs);
                self.code.push(Dup);
                let jump_index = self.code.len();
                self.code.push(JumpIfFalse(0));
                self.code.push(Pop);
                self.compile_expr(rhs);
                let jump_target = self.code.len().try_into().unwrap();
                self.code[jump_index] = JumpIfFalse(jump_target);
            }
            Expression::Binary {
                lhs,
                op: BinOp { kind: BinOpKind::Or, .. },
                rhs,
            } => {
                self.compile_expr(lhs);
                self.code.push(Dup);
                let jump_index = self.code.len();
                self.code.push(JumpIfTrue(0));
                self.code.push(Pop);
                self.compile_expr(rhs);
                let jump_target = self.code.len().try_into().unwrap();
                self.code[jump_index] = JumpIfTrue(jump_target);
            }
            Expression::Binary { lhs, op, rhs } => {
                self.compile_expr(lhs);
                self.compile_expr(rhs);
                let op = match op.kind {
                    BinOpKind::EqualEqual => Equal,
                    BinOpKind::NotEqual => NotEqual,
                    BinOpKind::Less => Less,
                    BinOpKind::LessEqual => LessEqual,
                    BinOpKind::Greater => Greater,
                    BinOpKind::GreaterEqual => GreaterEqual,
                    BinOpKind::Plus => Add,
                    BinOpKind::Minus => Subtract,
                    BinOpKind::Times => Multiply,
                    BinOpKind::Divide => Divide,
                    BinOpKind::Power => Power,
                    BinOpKind::Assign | BinOpKind::And | BinOpKind::Or => unreachable!(),
                };
                self.code.push(op);
            }
            Expression::Grouping { l_paren: _, expr, r_paren: _ } => self.compile_expr(expr),
            Expression::Name(variable) => self.compile_load(variable),
            Expression::Assign { target, equal: _, value } => {
                self.compile_expr(value);
                self.code.push(Dup);
                match target {
                    AssignmentTarget::Variable(variable) => {
                        self.compile_store(variable);
                    }
                    AssignmentTarget::Attribute { lhs, attribute } => {
                        self.compile_expr(lhs);
                        self.code.push(StoreAttr(attribute.id()));
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
                // TODO: if callee is an attribute lookup, compile into `LookupMethod` and
                // `CallMethod` as described here:
                // https://doc.pypy.org/en/latest/interpreter-optimizations.html#lookup-method-call-method
                self.compile_expr(callee);
                for arg in *arguments {
                    self.compile_expr(arg);
                }
                let argument_count = arguments.len().try_into().unwrap();
                let inner = CallInner {
                    argument_count,
                    stack_size_at_callsite: u32::try_from(stack_size_at_callsite.get()).unwrap(),
                };
                let call = if usize::try_from(argument_count).unwrap()
                    >= (Stack::<nanboxed::Value>::ELEMENT_COUNT_IN_GUARD_AREA - 1)
                {
                    Call(inner)
                }
                else {
                    ShortCall(inner)
                };
                self.code.push(call);
                self.code.push(Pop2);
            }
            Expression::Attribute { lhs, attribute } => {
                self.compile_expr(lhs);
                self.code.push(LoadAttr(attribute.id()));
            }
            Expression::Super { super_, this, attribute } => {
                self.compile_load(this);
                self.compile_load(super_);
                self.code.push(Super(attribute.id()));
            }
        }
        self.exit_expression(self.code.len());
    }

    fn compile_load(&mut self, variable: &Variable<'_>) {
        let op = match variable.target() {
            Target::Local(slot) => Local(slot.try_into().unwrap()),
            Target::GlobalByName => GlobalByName(variable.name.id()),
            Target::GlobalBySlot(slot) => Global(slot.try_into().unwrap()),
            Target::Cell(slot) => Cell(slot.try_into().unwrap()),
        };
        self.code.push(op);
    }

    fn compile_store(&mut self, variable: &Variable<'_>) {
        match variable.target() {
            Target::Local(slot) | Target::GlobalBySlot(slot) => {
                // FIXME: this should only be necessary in `compile_define`
                self.frame_stack.last_mut().insert(slot);
            }
            Target::Cell(_) | Target::GlobalByName => (),
        }
        let op = match variable.target() {
            Target::Local(slot) => StoreLocal(slot.try_into().unwrap()),
            Target::GlobalByName => StoreGlobalByName(variable.name.id()),
            Target::GlobalBySlot(slot) => StoreGlobal(slot.try_into().unwrap()),
            Target::Cell(slot) => StoreCell(slot.try_into().unwrap()),
        };
        self.code.push(op);
    }

    fn compile_define(&mut self, variable: &Variable<'_>) {
        match variable.target() {
            Target::Cell(slot) => self.code.push(DefineCell(slot.try_into().unwrap())),
            _ => {
                match variable.target() {
                    Target::Local(slot) | Target::GlobalBySlot(slot) => {
                        self.frame_stack.last_mut().insert(slot);
                    }
                    Target::Cell(_) | Target::GlobalByName => (),
                }
                self.compile_store(variable);
            }
        }
    }

    fn compile_function(&mut self, function: &'a Function<'a>, kind: FunctionKind) {
        let begin_index = self.code.len();
        self.code.push(BeginFunction(0));
        self.frame_stack.push(BTreeSet::new());

        let skip_this_if_method = match kind {
            FunctionKind::Function => 0,
            FunctionKind::Method => 1,
        };

        for param in function.parameters.iter().skip(skip_this_if_method).rev() {
            self.compile_define(param);
        }

        if kind == FunctionKind::Method {
            self.code.push(BoundMethodGetInstance);
            self.compile_define(&function.parameters[0]);
        }

        for stmt in function.body {
            self.compile_stmt(stmt);
        }

        if self.code.last() != Some(&Return) {
            self.code.push(ConstNil);
            self.code.push(Return);
        }
        let code_size = (self.code.len() - begin_index - 1).try_into().unwrap();
        self.code[begin_index] = BeginFunction(code_size);
        let meta_index = self.metadata.len();
        self.metadata
            .push(Metadata::Function { function, code_size });
        self.frame_stack.pop().unwrap();
        self.code
            .push(BuildFunction(meta_index.try_into().unwrap()));
    }

    fn enter_expression(&mut self, at: usize, expr: &'a Expression<'a>) {
        self.error_locations
            .push(ContainingExpression::Enter { at, expr });
    }

    fn enter_statement(&mut self, at: usize) {
        self.defined_local_slots
            .push(LocalSlotsDefinedInFrame::Enter {
                at,
                defined: self.frame_stack.last().iter().copied().collect(),
            });
    }

    fn exit_expression(&mut self, at: usize) {
        self.error_locations.push(ContainingExpression::Exit { at });
    }

    fn exit_statement(&mut self, at: usize) {
        self.defined_local_slots
            .push(LocalSlotsDefinedInFrame::Exit { at });
    }
}
