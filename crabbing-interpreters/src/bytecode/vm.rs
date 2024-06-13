use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Deref;
use std::ptr::NonNull;

use rustc_hash::FxHashMap as HashMap;
use variant_types::IntoEnum;
use variant_types::IntoVariant;
use Bytecode::*;

use super::CompiledBytecodes;
use crate::bytecode::compiler::ContainingExpression;
use crate::bytecode::compiler::Metadata;
use crate::bytecode::validate_bytecode;
use crate::bytecode::vm::stack::Stack;
use crate::bytecode::vm::stack_ref::SetSpOnDrop;
use crate::bytecode::vm::stack_ref::StackRef;
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

#[cfg_attr(feature = "mmap", path = "vm/mmap_stack.rs")]
pub(crate) mod stack;

const USEABLE_STACK_SIZE_IN_ELEMENTS: usize = ENV_SIZE.next_power_of_two();

impl<T> Stack<T> {
    #![cfg_attr(not(feature = "mmap"), allow(unused))]

    pub(crate) const ELEMENT_COUNT_IN_GUARD_AREA: usize =
        (Self::GUARD_PAGE_COUNT * Self::PAGE_SIZE) / std::mem::size_of::<T>();
    const GUARD_PAGE_COUNT: usize = 1;
    const PAGE_SIZE: usize = 4096;
    const SIZE_IN_BYTES: usize = Self::SIZE_IN_PAGES * Self::PAGE_SIZE;
    const SIZE_IN_PAGES: usize =
        2 * Self::GUARD_PAGE_COUNT + Self::USEABLE_SIZE_IN_BYTES / Self::PAGE_SIZE;
    const START_OFFSET: usize = Self::PAGE_SIZE * Self::GUARD_PAGE_COUNT;
    const USEABLE_SIZE_IN_BYTES: usize = USEABLE_STACK_SIZE_IN_ELEMENTS * std::mem::size_of::<T>();
    const _ASSERT_CORRECT_ALIGNMENT: () = assert!(Self::PAGE_SIZE >= std::mem::align_of::<T>());
    const _ASSERT_PAGE_SIZE_IS_MULTIPLE_OF_ELEMENT_SIZE: () =
        assert!(Self::PAGE_SIZE % std::mem::size_of::<T>() == 0);
    const _ASSERT_STACK_HAS_SIZE: () = assert!(Self::SIZE_IN_PAGES > 2);
}

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
    TooManyArgsInShortCall,
    ConstNumberIsNaN,
}

impl Report for InvalidBytecode {
    fn print(&self) {
        eprintln!("{self:?}")
    }

    fn exit_code(&self) -> i32 {
        66
    }
}

#[derive(Debug, Clone, Copy)]
struct CallFrame<'a> {
    pc: usize,
    offset: u32,
    cells: Cells<'a>,
}

unsafe impl Trace for CallFrame<'_> {
    fn trace(&self) {
        self.cells.trace();
    }
}

pub(crate) struct Vm<'a, 'b> {
    bytecode: &'b [Bytecode],
    constants: &'b [nanboxed::Value<'a>],
    metadata: &'b [Metadata<'a>],
    error_locations: &'b [ContainingExpression<'a>],
    env: Environment<'a>,
    stack_base: NonNull<nanboxed::Value<'a>>,
    stack_pointer: NonNull<nanboxed::Value<'a>>,
    offset: u32,
    call_stack: Stack<CallFrame<'a>>,
    cell_vars: Cells<'a>,
    execution_counts: Box<[u64; Bytecode::all_discriminants().len()]>,
    error: Option<Box<Error<'a>>>,
}

impl Drop for Vm<'_, '_> {
    fn drop(&mut self) {
        drop(unsafe { Stack::from_raw_parts(self.stack_base, self.stack_pointer) })
    }
}

impl<'a, 'b> Vm<'a, 'b> {
    pub(crate) fn new(
        bytecode: &'b [Bytecode],
        constants: &'b [nanboxed::Value<'a>],
        metadata: &'b [Metadata<'a>],
        error_locations: &'b [ContainingExpression<'a>],
        env: Environment<'a>,
        global_cells: Cells<'a>,
    ) -> Result<Self, InvalidBytecode> {
        let gc = env.gc;
        validate_bytecode(bytecode, metadata)?;
        let stack = Stack::new(Value::Nil.into_nanboxed());
        let (stack_base, stack_pointer) = stack.into_raw_parts();
        Ok(Self {
            bytecode,
            constants,
            metadata,
            error_locations,
            env,
            stack_base,
            stack_pointer,
            offset: 0,
            call_stack: Stack::new(CallFrame {
                pc: 0,
                offset: 0,
                cells: Cells::from_iter_in(gc, [].into_iter()),
            }),
            cell_vars: global_cells,
            execution_counts: Box::new([0; Bytecode::all_discriminants().len()]),
            error: None,
        })
    }

    pub(crate) fn set_error(&mut self, error: Option<Box<Error<'a>>>) {
        self.error = error;
    }

    pub(crate) unsafe fn set_stack_pointer(&mut self, stack_pointer: NonNull<nanboxed::Value<'a>>) {
        self.stack_pointer = stack_pointer;
    }

    pub(crate) fn error(mut self) -> Option<Box<Error<'a>>> {
        self.error.take()
    }

    pub(crate) fn execution_counts(&self) -> &[u64; Bytecode::all_discriminants().len()] {
        &self.execution_counts
    }

    fn stack(&self, sp: NonNull<nanboxed::Value<'a>>) -> StackRef<nanboxed::Value<'a>> {
        StackRef::new(self, sp)
    }

    fn stack_mut<'slf>(
        &'slf mut self,
        sp: &'slf mut NonNull<nanboxed::Value<'a>>,
    ) -> SetSpOnDrop<'slf, nanboxed::Value<'a>> {
        SetSpOnDrop::new(self.stack_base, sp)
    }

    #[inline(always)]
    fn collect_if_necessary(&self, sp: NonNull<nanboxed::Value<'a>>) {
        if self.env.gc.collection_necessary() {
            #[cold]
            #[inline(never)]
            extern "rust-cold" fn do_collect<'a>(
                vm: &Vm<'a, '_>,
                sp: NonNull<nanboxed::Value<'a>>,
            ) {
                vm.constants.trace();
                vm.stack(sp).trace();
                vm.call_stack.trace();
                vm.cell_vars.trace();
                vm.env.trace();
                unsafe { vm.env.gc.sweep() };
            }
            do_collect(self, sp);
        }
    }

    fn get_constant(&self, index: u32) -> nanboxed::Value<'a> {
        self.constants[index.cast()]
    }

    #[inline(never)]
    fn print_stack(&self, pc: usize, sp: NonNull<nanboxed::Value<'a>>) {
        println!(
            "stack at {:>4}    {}: {:#?}",
            pc,
            self.bytecode[pc],
            self.stack(sp).deref(),
        );
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

    pub(crate) fn run_threaded(&mut self, compiled_bytecodes: CompiledBytecodes) {
        let compiled_bytecode = unsafe { compiled_bytecodes.get_unchecked(0) };
        (compiled_bytecode.function)(self, 0, self.stack_pointer, compiled_bytecodes);
    }
}

pub fn run_bytecode<'a>(
    vm: &mut Vm<'a, '_>,
) -> Result<Value<'a>, ControlFlow<Value<'a>, Box<Error<'a>>>> {
    let mut pc = 0;
    let mut sp = vm.stack_pointer;
    loop {
        let bytecode = vm.bytecode[pc];
        match execute_bytecode(vm, &mut pc, &mut sp, bytecode) {
            Ok(()) => (),
            Err(None) => {
                assert_eq!(vm.stack_base, sp);
                return Ok(Value::Nil);
            }
            Err(Some(err)) => {
                vm.stack_pointer = sp;
                Err(err)?;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn execute_bytecode<'a>(
    vm: &mut Vm<'a, '_>,
    pc: &mut usize,
    sp: &mut NonNull<nanboxed::Value<'a>>,
    bytecode: Bytecode,
) -> Result<(), Option<Box<Error<'a>>>> {
    #[cfg(feature = "count_bytecode_execution")]
    {
        vm.execution_counts[bytecode.discriminant()] += 1;
    }

    let previous_pc = *pc;

    match bytecode {
        Pop => {
            vm.stack_mut(sp).pop();
        }
        Const(i) => {
            let constant = vm.get_constant(i);
            vm.stack_mut(sp).push(constant);
        }
        UnaryMinus => {
            let operand = vm.stack_mut(sp).pop();
            let value = match operand.parse() {
                Number(x) => Number(-x).into_nanboxed(),
                value => {
                    let expr = vm.error_location_at(*pc);
                    Err(Box::new(Error::InvalidUnaryOp {
                        value,
                        at: expr,
                        op: expr.0,
                    }))?
                }
            };
            vm.stack_mut(sp).push(value);
        }
        UnaryNot => {
            let operand = vm.stack_mut(sp).pop();
            vm.stack_mut(sp)
                .push(Bool(!operand.is_truthy()).into_nanboxed());
        }
        Equal => any_binop(vm, sp, |_, lhs, rhs| Ok(Bool(lhs == rhs)))?,
        NotEqual => any_binop(vm, sp, |_, lhs, rhs| Ok(Bool(lhs != rhs)))?,
        Less => number_binop(vm, *pc, sp, |lhs, rhs| Bool(lhs < rhs))?,
        LessEqual => number_binop(vm, *pc, sp, |lhs, rhs| Bool(lhs <= rhs))?,
        Greater => number_binop(vm, *pc, sp, |lhs, rhs| Bool(lhs > rhs))?,
        GreaterEqual => number_binop(vm, *pc, sp, |lhs, rhs| Bool(lhs >= rhs))?,
        Add => {
            let rhs = vm.stack(*sp).short_peek_at(0);
            let lhs = vm.stack(*sp).short_peek_at(1);
            let fast_path_result = lhs.data() + rhs.data();
            if fast_path_result.is_nan() {
                #[expect(improper_ctypes_definitions)]
                #[cold]
                #[inline(never)]
                extern "rust-cold" fn add_slow_path<'a>(
                    vm: &mut Vm<'a, '_>,
                    pc: usize,
                    mut sp: NonNull<nanboxed::Value<'a>>,
                ) -> Result<NonNull<nanboxed::Value<'a>>, Option<Box<Error<'a>>>> {
                    let sp = &mut sp;
                    let rhs = vm.stack_mut(sp).pop();
                    let lhs = vm.stack_mut(sp).pop();
                    let result = match (lhs.parse(), rhs.parse()) {
                        (Number(lhs), Number(rhs)) => Number(lhs + rhs),
                        (String(lhs), String(rhs)) =>
                            String(GcStr::new_in(vm.env.gc, &format!("{lhs}{rhs}"))),
                        (lhs, rhs) => {
                            let expr = vm.error_location_at(pc);
                            Err(Box::new(Error::InvalidBinaryOp {
                                at: expr,
                                lhs,
                                op: expr.op,
                                rhs,
                            }))?
                        }
                    };
                    vm.stack_mut(sp).push(result.into_nanboxed());
                    Ok(*sp)
                }
                *sp = add_slow_path(vm, *pc, *sp)?;
            }
            else {
                vm.stack_mut(sp).pop();
                vm.stack_mut(sp).pop();
                vm.stack_mut(sp)
                    .push(Number(fast_path_result).into_nanboxed());
            }
        }
        Subtract => nan_preserving_number_binop(vm, *pc, sp, |lhs, rhs| lhs - rhs)?,
        Multiply => nan_preserving_number_binop(vm, *pc, sp, |lhs, rhs| lhs * rhs)?,
        Divide => nan_preserving_number_binop(vm, *pc, sp, |lhs, rhs| lhs / rhs)?,
        Power => nan_preserving_number_binop(vm, *pc, sp, |lhs, rhs| lhs.powf(rhs))?,
        Local(slot) => {
            let value = vm.env[vm.offset + slot];
            vm.stack_mut(sp).push(value);
        }
        Global(slot) => {
            let value = vm.env[slot];
            vm.stack_mut(sp).push(value);
        }
        Cell(slot) => {
            let value = vm.cell_vars[slot.cast()].get().get();
            vm.stack_mut(sp).push(value);
        }
        Dup => {
            let value = vm.stack(*sp).peek();
            vm.stack_mut(sp).push(value);
        }
        StoreAttr(name) => {
            let assignment_target = vm.stack_mut(sp).pop().parse();
            let value = vm.stack_mut(sp).pop();
            match assignment_target {
                Instance(instance) => instance.attributes.borrow_mut().insert(name, value),
                _ => {
                    let expr: ExpressionTypes::Assign = vm.error_location_at(*pc);
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
                pc: usize,
                mut sp: NonNull<nanboxed::Value<'a>>,
                name: InternedString,
            ) -> Result<NonNull<nanboxed::Value<'a>>, Box<Error<'a>>> {
                let sp = &mut sp;
                let value = vm.stack_mut(sp).pop().parse();
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
                            let expr = vm.error_location_at(pc);
                            Box::new(Error::UndefinedProperty {
                                at: expr,
                                lhs: value,
                                attribute: expr.attribute,
                            })
                        })?,
                    _ => Err(Box::new(Error::NoProperty {
                        lhs: value,
                        at: vm.error_location_at(pc),
                    }))?,
                };
                vm.stack_mut(sp).push(value);
                Ok(*sp)
            }
            *sp = load_attr(vm, *pc, *sp, name)?;
        }
        LoadMethod(name) => {
            #[inline(never)]
            fn load_method<'a>(
                vm: &mut Vm<'a, '_>,
                pc: usize,
                mut sp: NonNull<nanboxed::Value<'a>>,
                name: InternedString,
            ) -> Result<NonNull<nanboxed::Value<'a>>, Box<Error<'a>>> {
                let sp = &mut sp;
                let nanboxed_value = vm.stack_mut(sp).pop();
                let value = nanboxed_value.parse();
                let (instance, method) = match value {
                    Instance(instance) => instance
                        .attributes
                        .borrow()
                        .get(&name)
                        .copied()
                        .map(|attribute| (Value::Nil.into_nanboxed(), attribute))
                        .or_else(|| {
                            instance
                                .class
                                .lookup_method(name)
                                .map(|method| match method.parse() {
                                    Value::Function(_) => (nanboxed_value, method),
                                    _ => unreachable!(),
                                })
                        })
                        .ok_or_else(|| {
                            let expr = vm.error_location_at(pc);
                            Box::new(Error::UndefinedProperty {
                                at: expr,
                                lhs: value,
                                attribute: expr.attribute,
                            })
                        })?,
                    _ => Err(Box::new(Error::NoProperty {
                        lhs: value,
                        at: vm.error_location_at(pc),
                    }))?,
                };
                vm.stack_mut(sp).push(method);
                vm.stack_mut(sp).push(instance);
                Ok(*sp)
            }
            *sp = load_method(vm, *pc, *sp, name)?;
        }
        StoreLocal(slot) => {
            let value = vm.stack_mut(sp).pop();
            vm.env[vm.offset + slot] = value;
        }
        StoreGlobal(slot) => {
            let value = vm.stack_mut(sp).pop();
            vm.env.set(
                vm.cell_vars,
                vm.offset.cast(),
                Target::GlobalBySlot(slot.cast()),
                value,
            );
        }
        StoreCell(slot) => {
            let value = vm.stack_mut(sp).pop();
            vm.cell_vars[slot.cast()].get().set(value);
        }
        DefineCell(slot) => {
            let value = vm.stack_mut(sp).pop();
            vm.cell_vars[slot.cast()].set(GcRef::new_in(vm.env.gc, Cell::new(value)));
        }
        Call(CallInner { argument_count, stack_size_at_callsite }) => {
            execute_call(
                vm,
                pc,
                sp,
                argument_count,
                stack_size_at_callsite,
                BoundsCheckedPeek,
            )?;
        }
        ShortCall(CallInner { argument_count, stack_size_at_callsite }) => {
            execute_call(
                vm,
                pc,
                sp,
                argument_count,
                stack_size_at_callsite,
                ShortPeek,
            )?;
        }
        CallMethod(CallInner { argument_count, stack_size_at_callsite }) => {
            execute_call_method(
                vm,
                pc,
                sp,
                argument_count,
                stack_size_at_callsite,
                BoundsCheckedPeek,
            )?;
        }
        ShortCallMethod(CallInner { argument_count, stack_size_at_callsite }) => {
            execute_call_method(
                vm,
                pc,
                sp,
                argument_count,
                stack_size_at_callsite,
                ShortPeek,
            )?;
        }
        Print => {
            #[cold]
            #[inline(never)]
            extern "rust-cold" fn print<'a>(
                vm: &mut Vm<'a, '_>,
                mut sp: NonNull<nanboxed::Value<'a>>,
            ) -> NonNull<nanboxed::Value<'a>> {
                let value = vm.stack_mut(&mut sp).pop().parse();
                println!("{value}");
                sp
            }
            *sp = print(vm, *sp);
        }
        GlobalByName(name) => {
            let variable = vm.env.get_global_slot_by_id(name).ok_or_else(|| {
                let expr: ExpressionTypes::Name = vm.error_location_at(*pc);
                Box::new(Error::UndefinedName { at: *expr.0.name })
            })?;

            let value = vm.env[u32::try_from(variable).unwrap()];
            vm.stack_mut(sp).push(value);
        }
        StoreGlobalByName(name) => {
            let slot = vm.env.get_global_slot_by_id(name).ok_or_else(|| {
                let expr: ExpressionTypes::Assign = vm.error_location_at(*pc);
                let target: AssignmentTargetTypes::Variable = expr.target.into_variant();
                Box::new(Error::UndefinedName { at: *target.0.name })
            })?;
            let value = vm.stack_mut(sp).pop();
            vm.env.set(
                vm.cell_vars,
                vm.offset.cast(),
                Target::GlobalBySlot(slot),
                value,
            );
        }
        JumpIfTrue(target) => {
            let value = vm.stack_mut(sp).pop();
            if value.is_truthy() {
                *pc = target.cast() - 1;
            }
        }
        JumpIfFalse(target) => {
            let value = vm.stack_mut(sp).pop();
            if !value.is_truthy() {
                *pc = target.cast() - 1;
            }
        }
        PopJumpIfTrue(target) => {
            let value = vm.stack(*sp).peek();
            if value.is_truthy() {
                vm.stack_mut(sp).pop();
                *pc = target.cast() - 1;
            }
        }
        PopJumpIfFalse(target) => {
            let value = vm.stack(*sp).peek();
            if !value.is_truthy() {
                vm.stack_mut(sp).pop();
                *pc = target.cast() - 1;
            }
        }
        Jump(target) => {
            *pc = target.cast() - 1;
        }
        BeginFunction(target) => {
            *pc += target.cast();
        }
        Return => {
            CallFrame {
                pc: *pc,
                offset: vm.offset,
                cells: vm.cell_vars,
            } = vm.call_stack.pop();
        }
        BuildFunction(metadata_index) => {
            let Metadata::Function { function, code_size } = vm.metadata[metadata_index.cast()]
            else {
                unreachable!()
            };
            let code_ptr = *pc - code_size.cast();
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
            vm.stack_mut(sp).push(value.into_nanboxed());
        }
        End => {
            assert!(vm.stack(*sp).is_empty());
            return Err(None);
        }
        Pop2 => {
            let value = vm.stack_mut(sp).pop();
            vm.stack_mut(sp).pop();
            vm.stack_mut(sp).push(value);
        }
        Pop23 => {
            let value = vm.stack_mut(sp).pop();
            vm.stack_mut(sp).pop();
            vm.stack_mut(sp).pop();
            vm.stack_mut(sp).push(value);
        }
        BuildClass(metadata_index) => {
            #[expect(improper_ctypes_definitions)]
            #[cold]
            #[inline(never)]
            extern "rust-cold" fn build_class<'a>(
                vm: &mut Vm<'a, '_>,
                mut sp: NonNull<nanboxed::Value<'a>>,
                metadata_index: u32,
            ) -> Result<NonNull<nanboxed::Value<'a>>, Box<Error<'a>>> {
                let sp = &mut sp;
                let Metadata::Class { name, methods, base_error_location } =
                    vm.metadata[metadata_index.cast()]
                else {
                    unreachable!()
                };
                let methods = methods
                    .iter()
                    .rev()
                    .map(|method| (method.name.id(), vm.stack_mut(sp).pop()))
                    .collect();
                let base = if let Some(error_location) = base_error_location {
                    let base = vm.stack_mut(sp).pop();
                    match base.parse() {
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
                vm.stack_mut(sp).push(Class(class).into_nanboxed());
                Ok(*sp)
            }
            *sp = build_class(vm, *sp, metadata_index)?;
        }
        PrintStack => {
            vm.print_stack(*pc, *sp);
        }
        b @ BoundMethodGetInstance => {
            let value = vm.stack(*sp).peek();
            match value.parse() {
                BoundMethod(bound_method) => vm
                    .stack_mut(sp)
                    .push(Value::Instance(bound_method.instance).into_nanboxed()),
                Instance(_) => vm.stack_mut(sp).push(value),
                value => unreachable!(
                    "invalid operand for bytecode `{b}`: {value}, expected `BoundMethod`"
                ),
            }
        }
        Super(name) => {
            let value = vm.stack_mut(sp).pop();
            let super_class = match value.parse() {
                Value::Class(class) => class,
                value => unreachable!("invalid base class value: {value}"),
            };
            let value = vm.stack_mut(sp).pop();
            let this = match value.parse() {
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
                    let super_ = vm.error_location_at(*pc);
                    Box::new(Error::UndefinedSuperProperty {
                        at: super_,
                        super_: Value::Instance(this),
                        attribute: super_.attribute,
                    })
                })?;
            vm.stack_mut(sp).push(value.into_nanboxed());
        }
        ConstNil => vm.stack_mut(sp).push(Value::Nil.into_nanboxed()),
        ConstTrue => vm.stack_mut(sp).push(Value::Bool(true).into_nanboxed()),
        ConstFalse => vm.stack_mut(sp).push(Value::Bool(false).into_nanboxed()),
        ConstNumber(number) => vm.stack_mut(sp).push(
            // SAFETY: `validate_bytecode` makes sure that `number` is not `NaN`
            unsafe { nanboxed::Value::from_f64_unchecked(number.into()) },
        ),
    }
    unsafe {
        *pc = pc.unchecked_add(1);
    }
    if cfg!(miri) || previous_pc > *pc {
        vm.collect_if_necessary(*sp);
    }
    Ok(())
}

#[inline(always)]
#[track_caller]
fn any_binop<'a>(
    vm: &mut Vm<'a, '_>,
    sp: &mut NonNull<nanboxed::Value<'a>>,
    op: impl FnOnce(&mut Vm<'a, '_>, Value<'a>, Value<'a>) -> Result<Value<'a>, Box<Error<'a>>>,
) -> Result<(), Box<Error<'a>>> {
    let rhs = vm.stack_mut(sp).pop().parse();
    let lhs = vm.stack_mut(sp).pop().parse();
    let result = op(vm, lhs, rhs)?;
    vm.stack_mut(sp).push(result.into_nanboxed());
    Ok(())
}

#[inline(always)]
#[track_caller]
fn number_binop<'a>(
    vm: &mut Vm<'a, '_>,
    pc: usize,
    sp: &mut NonNull<nanboxed::Value<'a>>,
    op: impl FnOnce(f64, f64) -> Value<'a>,
) -> Result<(), Box<Error<'a>>> {
    let rhs = vm.stack(*sp).short_peek_at(0);
    let lhs = vm.stack(*sp).short_peek_at(1);
    if lhs.data().is_nan() || rhs.data().is_nan() {
        #[expect(improper_ctypes_definitions)]
        #[cold]
        #[inline(never)]
        extern "rust-cold" fn number_binop_slow_path<'a>(
            vm: &mut Vm<'a, '_>,
            pc: usize,
            mut sp: NonNull<nanboxed::Value<'a>>,
            op: impl FnOnce(f64, f64) -> Value<'a>,
        ) -> Result<NonNull<nanboxed::Value<'a>>, Box<Error<'a>>> {
            let sp = &mut sp;
            let rhs = vm.stack_mut(sp).pop();
            let lhs = vm.stack_mut(sp).pop();
            let result = match (lhs.parse(), rhs.parse()) {
                (Number(lhs), Number(rhs)) => op(lhs, rhs),
                (lhs, rhs) => {
                    let expr = vm.error_location_at(pc);
                    Err(Error::InvalidBinaryOp { at: expr, lhs, op: expr.op, rhs })?
                }
            };
            vm.stack_mut(sp).push(result.into_nanboxed());
            Ok(*sp)
        }
        *sp = number_binop_slow_path(vm, pc, *sp, op)?;
        Ok(())
    }
    else {
        vm.stack_mut(sp).pop();
        vm.stack_mut(sp).pop();
        vm.stack_mut(sp)
            .push(op(lhs.data(), rhs.data()).into_nanboxed());
        Ok(())
    }
}

#[inline(always)]
#[track_caller]
fn nan_preserving_number_binop<'a>(
    vm: &mut Vm<'a, '_>,
    pc: usize,
    sp: &mut NonNull<nanboxed::Value<'a>>,
    op: impl Fn(f64, f64) -> f64,
) -> Result<(), Box<Error<'a>>> {
    let rhs = vm.stack(*sp).short_peek_at(0);
    let lhs = vm.stack(*sp).short_peek_at(1);
    let fast_path_result = op(lhs.data(), rhs.data());
    if fast_path_result.is_nan() {
        #[expect(improper_ctypes_definitions)]
        #[cold]
        #[inline(never)]
        extern "rust-cold" fn nan_preserving_number_binop_slow_path<'a>(
            vm: &mut Vm<'a, '_>,
            pc: usize,
            mut sp: NonNull<nanboxed::Value<'a>>,
            op: impl Fn(f64, f64) -> f64,
        ) -> Result<NonNull<nanboxed::Value<'a>>, Box<Error<'a>>> {
            let sp = &mut sp;
            let rhs = vm.stack_mut(sp).pop();
            let lhs = vm.stack_mut(sp).pop();
            let result = match (lhs.parse(), rhs.parse()) {
                (Number(lhs), Number(rhs)) => Number(op(lhs, rhs)).into_nanboxed(),
                (lhs, rhs) => {
                    let expr = vm.error_location_at(pc);
                    Err(Error::InvalidBinaryOp { at: expr, lhs, op: expr.op, rhs })?
                }
            };
            vm.stack_mut(sp).push(result);
            Ok(*sp)
        }
        *sp = nan_preserving_number_binop_slow_path(vm, pc, *sp, op)?;
        Ok(())
    }
    else {
        vm.stack_mut(sp).pop();
        vm.stack_mut(sp).pop();
        vm.stack_mut(sp)
            .push(Number(fast_path_result).into_nanboxed());
        Ok(())
    }
}

trait Peeker {
    fn peek_at<T>(&self, stack: &Stack<T>, index: u32) -> T
    where
        T: Copy;
}

struct ShortPeek;

impl Peeker for ShortPeek {
    fn peek_at<T>(&self, stack: &Stack<T>, index: u32) -> T
    where
        T: Copy,
    {
        stack.short_peek_at(index)
    }
}

struct BoundsCheckedPeek;

impl Peeker for BoundsCheckedPeek {
    fn peek_at<T>(&self, stack: &Stack<T>, index: u32) -> T
    where
        T: Copy,
    {
        stack.peek_at(index)
    }
}

#[inline(always)]
fn execute_call<'a>(
    vm: &mut Vm<'a, '_>,
    pc: &mut usize,
    sp: &mut NonNull<nanboxed::Value<'a>>,
    argument_count: u32,
    stack_size_at_callsite: u32,
    peeker: impl Peeker,
) -> Result<(), Option<Box<Error<'a>>>> {
    let callee = peeker.peek_at(&vm.stack(*sp), argument_count).parse();

    match callee {
        Function(function) => execute_function_call(
            vm,
            pc,
            function,
            function.parameters.len(),
            argument_count,
            stack_size_at_callsite,
            || callee,
        )?,
        BoundMethod(bound_method) => {
            let method = bound_method.method;
            execute_function_call(
                vm,
                pc,
                method,
                method.parameters.len() - 1,
                argument_count,
                stack_size_at_callsite,
                || callee,
            )?
        }
        NativeFunction(native_fn) => {
            #[cold]
            #[inline(never)]
            fn call_native_function<'a>(
                vm: &mut Vm<'a, '_>,
                pc: usize,
                mut sp: NonNull<nanboxed::Value<'a>>,
                argument_count: u32,
                native_fn: fn(Vec<Value>) -> Result<Value, NativeError>,
                callee: Value<'a>,
            ) -> Result<NonNull<nanboxed::Value<'a>>, Box<Error<'a>>> {
                let sp = &mut sp;
                // FIXME: This can be more efficient
                let mut args: Vec<_> = (0..argument_count)
                    .map(|_| vm.stack_mut(sp).pop().parse())
                    .collect();
                args.reverse();
                let value = native_fn(args).map_err(|err| {
                    Box::new(match err {
                        NativeError::Error(err) => err,
                        NativeError::ArityMismatch { expected } => Error::ArityMismatch {
                            callee,
                            expected,
                            at: vm.error_location_at(pc),
                        },
                    })
                })?;
                vm.stack_mut(sp).push(value.into_nanboxed());
                Ok(*sp)
            }
            *sp = call_native_function(vm, *pc, *sp, argument_count, native_fn, callee)?;
        }
        Class(class) => {
            #[cold]
            #[inline(never)]
            fn call_class<'a>(
                vm: &mut Vm<'a, '_>,
                mut pc: usize,
                mut sp: NonNull<nanboxed::Value<'a>>,
                class: GcRef<'a, ClassInner<'a>>,
                argument_count: u32,
                callee: Value<'a>,
                stack_size_at_callsite: u32,
            ) -> Result<(usize, NonNull<nanboxed::Value<'a>>), Option<Box<Error<'a>>>> {
                let pc = &mut pc;
                let sp = &mut sp;
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
                        *vm.stack_mut(sp).peek_at_mut(argument_count) = BoundMethod(GcRef::new_in(
                            vm.env.gc,
                            BoundMethodInner { method: init, instance },
                        ))
                        .into_nanboxed();
                        execute_function_call(
                            vm,
                            pc,
                            init,
                            init.parameters.len() - 1,
                            argument_count,
                            stack_size_at_callsite,
                            || callee,
                        )?
                    }
                    Some(_) => unreachable!(),
                    None if argument_count == 0 => vm
                        .stack_mut(sp)
                        .push(Value::Instance(instance).into_nanboxed()),
                    None => Err(Box::new(Error::ArityMismatch {
                        callee,
                        expected: 0,
                        at: vm.error_location_at(*pc),
                    }))?,
                }
                Ok((*pc, *sp))
            }
            (*pc, *sp) = call_class(
                vm,
                *pc,
                *sp,
                class,
                argument_count,
                callee,
                stack_size_at_callsite,
            )?
        }
        _ => Err(Box::new(Error::Uncallable {
            callee,
            at: vm.error_location_at(*pc),
        }))?,
    }

    Ok(())
}

#[inline(always)]
fn execute_call_method<'a>(
    vm: &mut Vm<'a, '_>,
    pc: &mut usize,
    sp: &mut NonNull<nanboxed::Value<'a>>,
    argument_count: u32,
    stack_size_at_callsite: u32,
    peeker: impl Peeker,
) -> Result<(), Option<Box<Error<'a>>>> {
    let callee = peeker.peek_at(&vm.stack(*sp), argument_count + 1).parse();
    let instance = peeker.peek_at(&vm.stack(*sp), argument_count);

    if instance.parse() == Value::Nil {
        unsafe {
            let result = vm.stack_mut(sp).swap(argument_count, argument_count + 1);
            result.unwrap();
        }
        execute_call(vm, pc, sp, argument_count, stack_size_at_callsite, peeker)
    }
    else {
        match callee {
            Function(function) => execute_function_call(
                vm,
                pc,
                function,
                function.parameters.len() - 1,
                argument_count,
                stack_size_at_callsite,
                || {
                    let Value::Instance(instance) = instance.parse()
                    else {
                        unreachable!()
                    };
                    Value::BoundMethod(GcRef::new_in(
                        vm.env.gc,
                        BoundMethodInner { method: function, instance },
                    ))
                },
            )?,
            _ => unreachable!("classes can only contain functions"),
        }
        Ok(())
    }
}

#[inline(always)]
fn execute_function_call<'a>(
    vm: &mut Vm<'a, '_>,
    pc: &mut usize,
    function: GcRef<'a, FunctionInner<'a>>,
    parameter_count: usize,
    argument_count: u32,
    stack_size_at_callsite: u32,
    callee: impl FnOnce() -> Value<'a>,
) -> Result<(), Option<Box<Error<'a>>>> {
    if parameter_count != argument_count.cast() {
        Err(Box::new(Error::ArityMismatch {
            callee: callee(),
            expected: parameter_count,
            at: vm.error_location_at(*pc),
        }))?
    }
    vm.call_stack.push(CallFrame {
        pc: *pc,
        offset: vm.offset,
        cells: vm.cell_vars,
    });
    *pc = function.code_ptr - 1;
    vm.offset += stack_size_at_callsite;
    vm.cell_vars = function.cells;
    Ok(())
}

mod stack_ref {
    use std::marker::PhantomData;
    use std::mem::ManuallyDrop;
    use std::ops::Deref;
    use std::ptr::NonNull;

    use crate::bytecode::vm::stack::Stack;
    use crate::bytecode::vm::Vm;
    use crate::value::nanboxed;

    pub(super) struct SetSpOnDrop<'a, T> {
        sp: &'a mut NonNull<T>,
        stack: ManuallyDrop<Stack<T>>,
    }

    impl<'a, T> SetSpOnDrop<'a, T> {
        pub(super) fn new(stack_base: NonNull<T>, sp: &'a mut NonNull<T>) -> Self {
            let stack = ManuallyDrop::new(unsafe { Stack::from_raw_parts(stack_base, *sp) });
            Self { sp, stack }
        }

        pub(super) unsafe fn swap(&mut self, i: u32, j: u32) -> Result<(), ()> {
            unsafe { self.stack.swap(i, j) }
        }
    }

    impl<'a, T> SetSpOnDrop<'a, T>
    where
        T: Copy,
    {
        pub(super) fn pop(&mut self) -> T {
            self.stack.pop()
        }

        pub(super) fn push(&mut self, value: T) {
            self.stack.push(value);
        }

        pub(super) fn peek_at_mut(&mut self, index: u32) -> &mut T {
            self.stack.peek_at_mut(index)
        }
    }

    impl<T> Drop for SetSpOnDrop<'_, T> {
        fn drop(&mut self) {
            let (_, sp) = unsafe { ManuallyDrop::take(&mut self.stack) }.into_raw_parts();
            *self.sp = sp;
        }
    }

    impl<T> Deref for SetSpOnDrop<'_, T> {
        type Target = Stack<T>;

        fn deref(&self) -> &Self::Target {
            &self.stack
        }
    }

    pub(super) struct StackRef<'a, T> {
        stack: ManuallyDrop<Stack<T>>,
        _lifetime: PhantomData<&'a ()>,
    }

    impl<'vm, 'a> StackRef<'vm, nanboxed::Value<'a>> {
        pub(super) fn new(vm: &'vm Vm<'a, '_>, sp: NonNull<nanboxed::Value<'a>>) -> Self {
            let stack = ManuallyDrop::new(unsafe { Stack::from_raw_parts(vm.stack_base, sp) });
            StackRef { stack, _lifetime: PhantomData }
        }
    }

    impl<T> Deref for StackRef<'_, T> {
        type Target = Stack<T>;

        fn deref(&self) -> &Self::Target {
            &self.stack
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gc;

    // Implementing `Stack` by using `Box` internally breaks stacked borrow rules (but not tree
    // borrow rules), because `Box` is `Unique` and `Vm::stack` lets us get multiple immutable refs
    // to the contained `Box`. This tests makes sure that `Stack` does not do that and is
    // implemented in a way that is compatible with multiple immutable refs existing at the same
    // time.
    #[test]
    fn multiple_stack_refs() {
        let gc = Gc::default();
        let global_cells = Cells::from_iter_in(&gc, [].into_iter());
        let mut vm = Vm::new(
            &[Bytecode::End],
            &[],
            &[],
            &[],
            Environment::new(&gc, HashMap::default(), global_cells),
            global_cells,
        )
        .unwrap();
        let mut sp = vm.stack_pointer;
        vm.stack_mut(&mut sp)
            .push(Value::Number(1.0).into_nanboxed());
        let stack = vm.stack(sp);
        let other_stack = vm.stack(sp);
        assert_eq!(stack.peek().parse(), other_stack.peek().parse());
    }
}
