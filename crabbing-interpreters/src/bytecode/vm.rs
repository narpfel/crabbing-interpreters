use std::cell::Cell;
use std::cell::RefCell;

use rustc_hash::FxHashMap as HashMap;
use variant_types::IntoEnum;
use variant_types::IntoVariant;
use Bytecode::*;

use crate::bytecode::compiler::ContainingExpression;
use crate::bytecode::compiler::Metadata;
use crate::bytecode::validate_bytecode;
use crate::bytecode::Bytecode;
use crate::bytecode::CallInner;
use crate::environment::Environment;
use crate::environment::ENV_SIZE;
use crate::eval::ControlFlow;
use crate::eval::Error;
use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::gc::Trace;
use crate::interner::interned;
use crate::interner::InternedString;
use crate::scope::AssignmentTargetTypes;
use crate::scope::Expression;
use crate::scope::ExpressionTypes;
use crate::scope::Target;
use crate::value::nanboxed;
use crate::value::BoundMethodInner;
use crate::value::Cells;
use crate::value::ClassInner;
use crate::value::FunctionInner;
use crate::value::InstanceInner;
use crate::value::NativeError;
use crate::value::Value;
use crate::value::Value::*;
use crate::Report;

trait Cast {
    type Target;

    fn cast(self) -> Self::Target;
}

impl Cast for u32 {
    type Target = usize;

    fn cast(self) -> Self::Target {
        usize::try_from(self).unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InvalidBytecode {
    NoEnd,
    JumpOutOfBounds,
}

impl Report for InvalidBytecode {
    fn print(&self) {
        eprintln!("{self:?}")
    }

    fn exit_code(&self) -> i32 {
        66
    }
}

pub(crate) struct Vm<'a, 'b> {
    bytecode: &'b [Bytecode],
    constants: &'b [nanboxed::Value<'a>],
    metadata: &'b [Metadata<'a>],
    error_locations: &'b [ContainingExpression<'a>],
    env: &'b mut Environment<'a>,
    pc: usize,
    stack: Box<[nanboxed::Value<'a>; ENV_SIZE]>,
    sp: usize,
    offset: u32,
    call_stack: Box<[(usize, u32, Cells<'a>); ENV_SIZE]>,
    call_sp: usize,
    cell_vars: Cells<'a>,
    execution_counts: Box<[u64; Bytecode::all_discriminants().len()]>,
}

impl<'a, 'b> Vm<'a, 'b> {
    pub(crate) fn new(
        bytecode: &'b [Bytecode],
        constants: &'b [nanboxed::Value<'a>],
        metadata: &'b [Metadata<'a>],
        error_locations: &'b [ContainingExpression<'a>],
        env: &'b mut Environment<'a>,
        global_cells: Cells<'a>,
    ) -> Result<Self, InvalidBytecode> {
        let stack =
            Box::try_from(vec![Value::Nil.into_nanboxed(); ENV_SIZE].into_boxed_slice()).unwrap();
        let call_stack = Box::try_from(
            vec![(0, 0, Cells::from_iter_in(env.gc, [].into_iter())); ENV_SIZE].into_boxed_slice(),
        )
        .unwrap();

        validate_bytecode(bytecode, metadata)?;

        Ok(Self {
            bytecode,
            constants,
            metadata,
            error_locations,
            env,
            pc: 0,
            stack,
            sp: 0,
            offset: 0,
            call_stack,
            call_sp: 0,
            cell_vars: global_cells,
            execution_counts: Box::new([0; Bytecode::all_discriminants().len()]),
        })
    }

    pub(crate) fn next_bytecode(&self) -> Bytecode {
        self.bytecode[self.pc]
    }

    pub(crate) fn pc(&self) -> usize {
        self.pc
    }

    pub(crate) fn execution_counts(&self) -> &[u64; Bytecode::all_discriminants().len()] {
        &self.execution_counts
    }

    fn collect_if_necessary(&self) {
        if self.env.gc.collection_necessary() {
            self.constants.trace();
            self.stack[..self.sp].trace();
            self.call_stack[..self.call_sp].trace();
            self.cell_vars.trace();
            self.env.trace();
            unsafe { self.env.gc.sweep() };
        }
    }

    fn push_stack(&mut self, value: nanboxed::Value<'a>) {
        #[cfg(feature = "debug_print")]
        {
            println!("pushing: {value}", value = value.parse());
            println!("     at: {:>5}   {:?}", self.pc, self.bytecode[self.pc]);
        }
        self.stack[self.sp] = value;
        self.sp += 1;
    }

    fn pop_stack(&mut self) -> nanboxed::Value<'a> {
        #[cfg(feature = "debug_print")]
        {
            println!("popping: {}", self.stack[self.sp - 1].parse());
            println!("     at: {:>5}   {:?}", self.pc, self.bytecode[self.pc]);
        }
        self.sp -= 1;
        self.stack[self.sp]
    }

    fn peek_stack(&self) -> nanboxed::Value<'a> {
        self.stack[self.sp - 1]
    }

    fn get_constant(&self, index: u32) -> nanboxed::Value<'a> {
        self.constants[index.cast()]
    }

    #[inline(never)]
    fn print_stack(&self) {
        println!(
            "stack at {:>4}    {}: {:#?}",
            self.pc,
            self.bytecode[self.pc],
            &self.stack[..self.sp]
        );
    }

    fn error_location<ExpressionType>(&self) -> ExpressionType
    where
        ExpressionType: IntoEnum<Enum = Expression<'a>>,
        Expression<'a>: IntoVariant<ExpressionType>,
    {
        self.error_location_at(self.pc)
    }

    #[cold]
    fn error_location_at<ExpressionType>(&self, pc: usize) -> ExpressionType
    where
        ExpressionType: IntoEnum<Enum = Expression<'a>>,
        Expression<'a>: IntoVariant<ExpressionType>,
    {
        let index = self
            .error_locations
            .partition_point(|containing_expr| containing_expr.at() <= pc);

        let mut depth = 0;
        for containing_expr in self.error_locations[..index].iter().rev() {
            match containing_expr {
                ContainingExpression::Enter { at: _, expr } =>
                    if depth == 0 {
                        return expr.into_variant();
                    }
                    else {
                        depth -= 1;
                    },
                ContainingExpression::Exit { at: _ } => depth += 1,
            }
        }
        unreachable!()
    }
}

pub fn run_bytecode<'a>(
    vm: &mut Vm<'a, '_>,
) -> Result<Value<'a>, ControlFlow<Value<'a>, Box<Error<'a>>>> {
    loop {
        let bytecode = vm.bytecode[vm.pc];
        match execute_bytecode(vm, bytecode) {
            Ok(()) => (),
            Err(None) => return Ok(Value::Nil),
            Err(Some(err)) => Err(err)?,
        }
    }
}

macro_rules! outline {
    ($($stmt:stmt);* $(;)?) => {
        #[allow(clippy::redundant_closure_call)]
        (
            #[cold]
            #[inline(never)]
            || {
                $( $stmt )*
            }
        )()
    };
}

#[inline(always)]
pub(crate) fn execute_bytecode<'a>(
    vm: &mut Vm<'a, '_>,
    bytecode: Bytecode,
) -> Result<(), Option<Box<Error<'a>>>> {
    #[cfg(feature = "debug_print")]
    {
        println!(
            "{pc:>5}   {bytecode} ({sp})",
            bytecode = vm.bytecode[vm.pc],
            pc = vm.pc,
            sp = vm.sp,
        );
    }

    #[cfg(feature = "count_bytecode_execution")]
    {
        vm.execution_counts[bytecode.discriminant()] += 1;
    }

    let previous_pc = vm.pc;

    match bytecode {
        Pop => {
            vm.pop_stack();
        }
        Const(i) => {
            vm.push_stack(vm.get_constant(i));
        }
        UnaryMinus => {
            let value = match vm.pop_stack().parse() {
                Number(x) => Number(-x).into_nanboxed(),
                value => {
                    let expr = vm.error_location();
                    Err(Box::new(Error::InvalidUnaryOp {
                        value,
                        at: expr,
                        op: expr.0,
                    }))?
                }
            };
            vm.push_stack(value);
        }
        UnaryNot => {
            let value = vm.pop_stack();
            vm.push_stack(Bool(!value.is_truthy()).into_nanboxed());
        }
        Equal => any_binop(vm, |_, lhs, rhs| Ok(Bool(lhs == rhs)))?,
        NotEqual => any_binop(vm, |_, lhs, rhs| Ok(Bool(lhs != rhs)))?,
        Less => number_binop(vm, |lhs, rhs| Bool(lhs < rhs))?,
        LessEqual => number_binop(vm, |lhs, rhs| Bool(lhs <= rhs))?,
        Greater => number_binop(vm, |lhs, rhs| Bool(lhs > rhs))?,
        GreaterEqual => number_binop(vm, |lhs, rhs| Bool(lhs >= rhs))?,
        Add => any_binop(vm, |vm, lhs, rhs| match (lhs, rhs) {
            (Number(lhs), Number(rhs)) => Ok(Number(lhs + rhs)),
            (String(lhs), String(rhs)) =>
                Ok(String(GcStr::new_in(vm.env.gc, &format!("{lhs}{rhs}")))),
            _ => {
                let expr = vm.error_location();
                Err(Box::new(Error::InvalidBinaryOp {
                    at: expr,
                    lhs,
                    op: expr.op,
                    rhs,
                }))?
            }
        })?,
        Subtract => number_binop(vm, |lhs, rhs| Number(lhs - rhs))?,
        Multiply => number_binop(vm, |lhs, rhs| Number(lhs * rhs))?,
        Divide => number_binop(vm, |lhs, rhs| Number(lhs / rhs))?,
        Power => number_binop(vm, |lhs, rhs| Number(lhs.powf(rhs)))?,
        Local(slot) => {
            vm.push_stack(vm.env[vm.offset + slot]);
        }
        Global(slot) => {
            vm.push_stack(vm.env[slot]);
        }
        Cell(slot) => {
            vm.push_stack(vm.cell_vars[slot.cast()].get().get());
        }
        Dup => {
            let value = vm.peek_stack();
            vm.push_stack(value);
        }
        StoreAttr(name) => {
            let assignment_target = vm.pop_stack().parse();
            let value = vm.pop_stack();
            match assignment_target {
                Instance(instance) => instance.attributes.borrow_mut().insert(name, value),
                _ => {
                    let expr: ExpressionTypes::Assign = vm.error_location();
                    Err(Box::new(Error::NoFields {
                        lhs: assignment_target,
                        at: expr.target.into_variant(),
                    }))?
                }
            };
        }
        LoadAttr(name) => {
            #[inline(never)]
            fn load_attr<'a>(
                vm: &mut Vm<'a, '_>,
                name: InternedString,
            ) -> Result<(), Box<Error<'a>>> {
                let value = vm.pop_stack().parse();
                let value = match value {
                    Instance(instance) => instance
                        .attributes
                        .borrow()
                        .get(&name)
                        .copied()
                        .or_else(|| {
                            instance
                                .class
                                .lookup_method(name)
                                .map(|method| match method.parse() {
                                    Value::Function(method) => Value::BoundMethod(GcRef::new_in(
                                        vm.env.gc,
                                        BoundMethodInner { method, instance },
                                    ))
                                    .into_nanboxed(),
                                    _ => unreachable!(),
                                })
                        })
                        .ok_or_else(|| {
                            let expr = vm.error_location();
                            Box::new(Error::UndefinedProperty {
                                at: expr,
                                lhs: value,
                                attribute: expr.attribute,
                            })
                        })?,
                    _ => Err(Box::new(Error::NoProperty {
                        lhs: value,
                        at: vm.error_location(),
                    }))?,
                };
                vm.push_stack(value);
                Ok(())
            }
            load_attr(vm, name)?
        }
        StoreLocal(slot) => {
            let value = vm.pop_stack();
            vm.env[vm.offset + slot] = value;
        }
        StoreGlobal(slot) => {
            let value = vm.pop_stack();
            vm.env.set(
                vm.cell_vars,
                vm.offset.cast(),
                Target::GlobalBySlot(slot.cast()),
                value,
            );
        }
        StoreCell(slot) => {
            let value = vm.pop_stack();
            vm.cell_vars[slot.cast()].get().set(value);
        }
        DefineCell(slot) => {
            let value = vm.pop_stack();
            vm.cell_vars[slot.cast()].set(GcRef::new_in(vm.env.gc, Cell::new(value)));
        }
        Call(CallInner { argument_count, stack_size_at_callsite }) => {
            let callee = vm.stack[vm.sp - 1 - argument_count.cast()].parse();
            match callee {
                Function(function) => {
                    if function.parameters.len() != argument_count.cast() {
                        Err(Box::new(Error::ArityMismatch {
                            callee,
                            expected: function.parameters.len(),
                            at: vm.error_location(),
                        }))?
                    }
                    vm.call_stack[vm.call_sp] = (vm.pc, vm.offset, vm.cell_vars);
                    vm.call_sp += 1;
                    vm.pc = function.code_ptr - 1;
                    vm.offset += stack_size_at_callsite;
                    vm.cell_vars = function.cells;
                }
                BoundMethod(bound_method) => {
                    let method = bound_method.method;
                    if method.parameters.len() - 1 != argument_count.cast() {
                        Err(Box::new(Error::ArityMismatch {
                            callee,
                            expected: method.parameters.len() - 1,
                            at: vm.error_location(),
                        }))?
                    }
                    vm.call_stack[vm.call_sp] = (vm.pc, vm.offset, vm.cell_vars);
                    vm.call_sp += 1;
                    vm.pc = method.code_ptr - 1;
                    vm.offset += stack_size_at_callsite;
                    vm.cell_vars = method.cells;
                }
                NativeFunction(native_fn) => {
                    // FIXME: This can be more efficient
                    let mut args: Vec<_> = (0..argument_count)
                        .map(|_| vm.pop_stack().parse())
                        .collect();
                    args.reverse();
                    let value = native_fn(args).map_err(|err| {
                        Box::new(match err {
                            NativeError::Error(err) => err,
                            NativeError::ArityMismatch { expected } => Error::ArityMismatch {
                                callee,
                                expected,
                                at: vm.error_location(),
                            },
                        })
                    })?;
                    vm.push_stack(value.into_nanboxed());
                }
                Class(class) => {
                    let instance = GcRef::new_in(
                        vm.env.gc,
                        InstanceInner {
                            class,
                            attributes: RefCell::new(HashMap::default()),
                        },
                    );
                    match class
                        .lookup_method(interned::INIT)
                        .map(nanboxed::Value::parse)
                    {
                        Some(Value::Function(init)) => {
                            if init.parameters.len() - 1 != argument_count.cast() {
                                Err(Box::new(Error::ArityMismatch {
                                    callee,
                                    expected: init.parameters.len() - 1,
                                    at: vm.error_location(),
                                }))?
                            }
                            vm.stack[vm.sp - 1 - argument_count.cast()] =
                                BoundMethod(GcRef::new_in(
                                    vm.env.gc,
                                    BoundMethodInner { method: init, instance },
                                ))
                                .into_nanboxed();
                            vm.call_stack[vm.call_sp] = (vm.pc, vm.offset, vm.cell_vars);
                            vm.call_sp += 1;
                            vm.pc = init.code_ptr - 1;
                            vm.offset += stack_size_at_callsite;
                            vm.cell_vars = init.cells;
                        }
                        Some(_) => unreachable!(),
                        None if argument_count == 0 =>
                            vm.push_stack(Value::Instance(instance).into_nanboxed()),
                        None => Err(Box::new(Error::ArityMismatch {
                            callee,
                            expected: 0,
                            at: vm.error_location(),
                        }))?,
                    }
                }
                _ => Err(Box::new(Error::Uncallable {
                    callee,
                    at: vm.error_location(),
                }))?,
            }
        }
        Print => outline! {
            let value = vm.pop_stack().parse();
            println!("{value}");
        },
        GlobalByName(name) => {
            let variable = vm.env.get_global_slot_by_id(name).ok_or_else(|| {
                let expr: ExpressionTypes::Name = vm.error_location();
                Box::new(Error::UndefinedName { at: *expr.0.name })
            })?;

            let value = vm.env[u32::try_from(variable).unwrap()];
            vm.push_stack(value);
        }
        StoreGlobalByName(name) => {
            let slot = vm.env.get_global_slot_by_id(name).ok_or_else(|| {
                let expr: ExpressionTypes::Assign = vm.error_location();
                let target: AssignmentTargetTypes::Variable = expr.target.into_variant();
                Box::new(Error::UndefinedName { at: *target.0.name })
            })?;
            let value = vm.pop_stack();
            vm.env.set(
                vm.cell_vars,
                vm.offset.cast(),
                Target::GlobalBySlot(slot),
                value,
            );
        }
        JumpIfTrue(target) => {
            let value = vm.pop_stack();
            if value.is_truthy() {
                vm.pc = target.cast() - 1;
            }
        }
        JumpIfFalse(target) => {
            let value = vm.pop_stack();
            if !value.is_truthy() {
                vm.pc = target.cast() - 1;
            }
        }
        PopJumpIfTrue(target) => {
            let value = vm.peek_stack();
            if value.is_truthy() {
                vm.pop_stack();
                vm.pc = target.cast() - 1;
            }
        }
        PopJumpIfFalse(target) => {
            let value = vm.peek_stack();
            if !value.is_truthy() {
                vm.pop_stack();
                vm.pc = target.cast() - 1;
            }
        }
        Jump(target) => {
            vm.pc = target.cast() - 1;
        }
        BeginFunction(target) => {
            vm.pc += target.cast();
        }
        Return => {
            vm.call_sp -= 1;
            (vm.pc, vm.offset, vm.cell_vars) = vm.call_stack[vm.call_sp]
        }
        BuildFunction(metadata_index) => {
            let Metadata::Function { function, code_size } = vm.metadata[metadata_index.cast()]
            else {
                unreachable!()
            };
            let code_ptr = vm.pc - code_size.cast();
            let cells = GcRef::from_iter_in(
                vm.env.gc,
                function.cells.iter().map(|cell| match cell {
                    Some(idx) => Cell::new(vm.cell_vars[*idx].get()),
                    None => Cell::new(GcRef::new_in(
                        vm.env.gc,
                        Cell::new(Value::Nil.into_nanboxed()),
                    )),
                }),
            );
            let value = Value::Function(GcRef::new_in(
                vm.env.gc,
                FunctionInner {
                    name: function.name.slice(),
                    parameters: function.parameters,
                    code: function.body,
                    cells,
                    code_ptr,
                    compiled_body: function.compiled_body,
                },
            ));
            vm.push_stack(value.into_nanboxed());
        }
        End => {
            assert_eq!(vm.sp, 0);
            return Err(None);
        }
        Pop2 => {
            let value = vm.pop_stack();
            vm.pop_stack();
            vm.push_stack(value);
        }
        BuildClass(metadata_index) => {
            let Metadata::Class { name, methods, base_error_location } =
                vm.metadata[metadata_index.cast()]
            else {
                unreachable!()
            };
            let methods = methods
                .iter()
                .rev()
                .map(|method| (method.name.id(), vm.pop_stack()))
                .collect();
            let base = if let Some(error_location) = base_error_location {
                match vm.pop_stack().parse() {
                    Class(class) => Some(class),
                    base => Err(Box::new(Error::InvalidBase {
                        base,
                        at: vm.error_location_at(error_location),
                    }))?,
                }
            }
            else {
                None
            };
            let class = GcRef::new_in(vm.env.gc, ClassInner { name, base, methods });
            if let Some(base) = base {
                let base = Class(base).into_nanboxed();
                class.methods.values().for_each(|method| {
                    let Function(method) = method.parse()
                    else {
                        unreachable!()
                    };
                    method.cells[0].set(GcRef::new_in(vm.env.gc, Cell::new(base)));
                });
            }
            vm.push_stack(Class(class).into_nanboxed());
        }
        PrintStack => {
            vm.print_stack();
        }
        b @ BoundMethodGetInstance => match vm.peek_stack().parse() {
            BoundMethod(bound_method) =>
                vm.push_stack(Value::Instance(bound_method.instance).into_nanboxed()),
            value =>
                unreachable!("invalid operand for bytecode `{b}`: {value}, expected `BoundMethod`"),
        },
        Super(name) => {
            let super_class = match vm.pop_stack().parse() {
                Value::Class(class) => class,
                value => unreachable!("invalid base class value: {value}"),
            };
            let this = match vm.pop_stack().parse() {
                Value::Instance(this) => this,
                value => unreachable!("`this` is not an instance: {value}"),
            };
            let value = super_class
                .lookup_method(name)
                .map(|method| match method.parse() {
                    Value::Function(method) => Value::BoundMethod(GcRef::new_in(
                        vm.env.gc,
                        BoundMethodInner { method, instance: this },
                    )),
                    _ => unreachable!(),
                })
                .ok_or_else(|| {
                    let super_ = vm.error_location();
                    Box::new(Error::UndefinedSuperProperty {
                        at: super_,
                        super_: Value::Instance(this),
                        attribute: super_.attribute,
                    })
                })?;
            vm.push_stack(value.into_nanboxed());
        }
        ConstNil => vm.push_stack(Value::Nil.into_nanboxed()),
        ConstTrue => vm.push_stack(Value::Bool(true).into_nanboxed()),
        ConstFalse => vm.push_stack(Value::Bool(false).into_nanboxed()),
        ConstNumber(number) => vm.push_stack(Value::Number(number.into()).into_nanboxed()),
    }
    vm.pc += 1;
    if cfg!(miri) || previous_pc > vm.pc {
        vm.collect_if_necessary();
    }
    Ok(())
}

#[inline(always)]
#[track_caller]
fn any_binop<'a, 'b>(
    vm: &mut Vm<'a, 'b>,
    op: impl for<'c> FnOnce(
        &'c mut Vm<'a, 'b>,
        Value<'a>,
        Value<'a>,
    ) -> Result<Value<'a>, Box<Error<'a>>>,
) -> Result<(), Box<Error<'a>>> {
    let rhs = vm.pop_stack().parse();
    let lhs = vm.pop_stack().parse();
    let result = op(vm, lhs, rhs)?;
    vm.push_stack(result.into_nanboxed());
    Ok(())
}

#[inline(always)]
#[track_caller]
fn number_binop<'a, 'b>(
    vm: &mut Vm<'a, 'b>,
    op: impl FnOnce(f64, f64) -> Value<'a>,
) -> Result<(), Box<Error<'a>>> {
    let rhs = vm.pop_stack().parse();
    let lhs = vm.pop_stack().parse();
    let result = match (lhs, rhs) {
        (Number(lhs), Number(rhs)) => op(lhs, rhs),
        _ => {
            let expr = vm.error_location();
            Err(Error::InvalidBinaryOp { at: expr, lhs, op: expr.op, rhs })?
        }
    };
    vm.push_stack(result.into_nanboxed());
    Ok(())
}
