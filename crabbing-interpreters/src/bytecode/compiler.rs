use std::fmt;

use crate::gc::Gc;
use crate::gc::GcStr;
use crate::interner::InternedString;
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
use crate::value::Value;

const NIL_CONSTANT: u32 = 0;
const FALSE_CONSTANT: u32 = 1;
const TRUE_CONSTANT: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bytecode {
    Pop,
    Const(u32),
    UnaryMinus,
    UnaryNot,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Local(u32),
    Global(u32),
    Cell(u32),
    Dup,
    StoreAttr(InternedString),
    LoadAttr(InternedString),
    StoreLocal(u32),
    StoreGlobal(u32),
    StoreCell(u32),
    DefineCell(u32),
    Call {
        argument_count: u32,
        stack_size_at_callsite: u32,
    },
    Print,
    GlobalByName(InternedString),
    StoreGlobalByName(InternedString),
    JumpIfTrue(u32),
    JumpIfFalse(u32),
    PopJumpIfTrue(u32),
    PopJumpIfFalse(u32),
    Jump(u32),
    BeginFunction(u32),
    Return,
    BuildFunction(u32),
    End,
    Pop2,
    BuildClass(u32),
    #[allow(unused)]
    PrintStack,
    BoundMethodGetInstance,
    Super(InternedString),
}

use Bytecode::*;

impl fmt::Display for Bytecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pop => write!(f, "pop"),
            Const(constant) => write!(f, "const {constant}"),
            UnaryMinus => write!(f, "unary_minus"),
            UnaryNot => write!(f, "unary_not"),
            Equal => write!(f, "equal"),
            NotEqual => write!(f, "not_equal"),
            Less => write!(f, "less"),
            LessEqual => write!(f, "less_equal"),
            Greater => write!(f, "greater"),
            GreaterEqual => write!(f, "greater_equal"),
            Add => write!(f, "add"),
            Subtract => write!(f, "subtract"),
            Multiply => write!(f, "multiply"),
            Divide => write!(f, "divide"),
            Power => write!(f, "power"),
            Local(slot) => write!(f, "load_local {slot}"),
            Global(slot) => write!(f, "load_global {slot}"),
            Cell(slot) => write!(f, "load_cell {slot}"),
            Dup => write!(f, "dup"),
            StoreAttr(string) => write!(f, "store_attr {string}"),
            LoadAttr(string) => write!(f, "load_attr {string}"),
            StoreLocal(slot) => write!(f, "store_local {slot}"),
            StoreGlobal(slot) => write!(f, "store_global {slot}"),
            StoreCell(slot) => write!(f, "store_cell {slot}"),
            DefineCell(slot) => write!(f, "define_cell {slot}"),
            Call { argument_count, stack_size_at_callsite } =>
                write!(f, "call +{stack_size_at_callsite} arity={argument_count}"),
            Print => write!(f, "print"),
            GlobalByName(string) => write!(f, "global_by_name {string}"),
            StoreGlobalByName(string) => write!(f, "store_global_by_name {string}"),
            JumpIfTrue(target) => write!(f, "jump_if_true {target}"),
            JumpIfFalse(target) => write!(f, "jump_if_false {target}"),
            PopJumpIfTrue(target) => write!(f, "pop_jump_if_true {target}"),
            PopJumpIfFalse(target) => write!(f, "pop_jump_if_false {target}"),
            Jump(target) => write!(f, "jump {target}"),
            BeginFunction(size) => write!(f, "function {size}"),
            Return => write!(f, "return"),
            BuildFunction(meta_index) => write!(f, "build_function {meta_index}"),
            End => write!(f, "end"),
            Pop2 => write!(f, "pop2"),
            BuildClass(meta_index) => write!(f, "build_class {meta_index}"),
            PrintStack => write!(f, "print_stack"),
            BoundMethodGetInstance => write!(f, "bound_method_get_instance"),
            Super(name) => write!(f, "super {name}"),
        }
    }
}

struct Compiler<'a> {
    gc: &'a Gc,
    code: Vec<Bytecode>,
    constants: Vec<Value<'a>>,
    metadata: Vec<Metadata<'a>>,
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
        has_base: bool,
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
            Metadata::Class { name, methods, has_base } => f
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
                .field("has_base", has_base)
                .finish(),
        }
    }
}

pub(crate) fn compile_program<'a>(
    gc: &'a Gc,
    program: &'a [Statement<'a>],
) -> (Vec<Bytecode>, Vec<Value<'a>>, Vec<Metadata<'a>>) {
    let mut compiler = Compiler {
        gc,
        code: Vec::new(),
        constants: vec![Value::Nil, Value::Bool(false), Value::Bool(true)],
        metadata: Vec::new(),
    };

    for stmt in program {
        compiler.compile_stmt(stmt);
    }

    compiler.code.push(End);

    let Compiler { gc: _, code, constants, metadata } = compiler;
    (code, constants, metadata)
}

impl<'a> Compiler<'a> {
    fn compile_stmt(&mut self, stmt: &'a Statement<'a>) {
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
                    self.code.push(Const(NIL_CONSTANT));
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
                    self.code.push(Const(NIL_CONSTANT));
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
                    self.code.push(Const(NIL_CONSTANT));
                }
                self.code.push(Return);
            }
            Statement::InitReturn(_) => {
                self.code.push(BoundMethodGetInstance);
                self.code.push(Return);
            }
            Statement::Class { target, base, methods } => {
                if target.is_cell() {
                    self.code.push(Const(NIL_CONSTANT));
                    self.compile_define(target);
                }
                if let Some(base) = base {
                    self.compile_expr(base);
                }
                methods
                    .iter()
                    .for_each(|method| self.compile_function(method, FunctionKind::Method));
                let meta_index = self.metadata.len();
                self.metadata.push(Metadata::Class {
                    name: target.name.slice(),
                    methods,
                    has_base: base.is_some(),
                });
                self.code.push(BuildClass(meta_index.try_into().unwrap()));
                self.compile_store(target);
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expression<'a>) {
        match expr {
            Expression::Literal(literal) => {
                // FIXME: deduplicate constants
                let constant = match literal.kind {
                    LiteralKind::Number(x) => {
                        self.constants.push(Value::Number(x));
                        (self.constants.len() - 1).try_into().unwrap()
                    }
                    LiteralKind::String(s) => {
                        self.constants
                            .push(Value::String(GcStr::new_in(self.gc, s)));
                        (self.constants.len() - 1).try_into().unwrap()
                    }
                    LiteralKind::True => TRUE_CONSTANT,
                    LiteralKind::False => FALSE_CONSTANT,
                    LiteralKind::Nil => NIL_CONSTANT,
                };
                self.code.push(Const(constant));
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
            Expression::Name(variable) => {
                let op = match variable.target() {
                    Target::Local(slot) => Local(slot.try_into().unwrap()),
                    Target::GlobalByName => GlobalByName(variable.name.id()),
                    Target::GlobalBySlot(slot) => Global(slot.try_into().unwrap()),
                    Target::Cell(slot) => Cell(slot.try_into().unwrap()),
                };
                self.code.push(op);
            }
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
                self.code.push(Call {
                    argument_count: arguments.len().try_into().unwrap(),
                    stack_size_at_callsite: u32::try_from(*stack_size_at_callsite).unwrap(),
                });
                self.code.push(Pop2);
            }
            Expression::Attribute { lhs, attribute } => {
                self.compile_expr(lhs);
                self.code.push(LoadAttr(attribute.id()));
            }
            Expression::Super { super_, this, attribute } => {
                self.compile_expr(&Expression::Name(*this));
                self.compile_expr(&Expression::Name(*super_));
                self.code.push(Super(attribute.id()));
            }
        }
    }

    fn compile_store(&mut self, variable: &Variable<'_>) {
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
            _ => self.compile_store(variable),
        }
    }

    fn compile_function(&mut self, function: &'a Function<'a>, kind: FunctionKind) {
        let begin_index = self.code.len();
        self.code.push(BeginFunction(0));

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
            self.code.push(Const(NIL_CONSTANT));
            self.code.push(Return);
        }
        let code_size = (self.code.len() - begin_index - 1).try_into().unwrap();
        self.code[begin_index] = BeginFunction(code_size);
        let meta_index = self.metadata.len();
        self.metadata
            .push(Metadata::Function { function, code_size });
        self.code
            .push(BuildFunction(meta_index.try_into().unwrap()));
    }
}
