use std::cell::Cell;
use std::cell::RefCell;

use rustc_hash::FxHashMap as HashMap;
use variant_types::IntoEnum;
use variant_types::IntoVariant;
use Bytecode::*;
use Value::*;

use crate::bytecode::compiler::ContainingExpression;
use crate::bytecode::compiler::Metadata;
use crate::bytecode::Bytecode;
use crate::environment::Environment;
use crate::environment::ENV_SIZE;
use crate::eval::ControlFlow;
use crate::eval::Error;
use crate::gc::GcRef;
use crate::gc::GcStr;
use crate::interner::interned;
use crate::scope::AssignmentTargetTypes;
use crate::scope::Expression;
use crate::scope::ExpressionTypes;
use crate::scope::Target;
use crate::value::Cells;
use crate::value::ClassInner;
use crate::value::FunctionInner;
use crate::value::InstanceInner;
use crate::value::NativeError;
use crate::value::Value;

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

struct Vm<'a, 'b> {
    bytecode: &'b [Bytecode],
    constants: &'b [Value<'a>],
    metadata: &'b [Metadata<'a>],
    error_locations: &'b [ContainingExpression<'a>],
    env: &'b mut Environment<'a>,
    pc: usize,
    stack: Box<[Value<'a>; ENV_SIZE]>,
    sp: usize,
    offset: u32,
    call_stack: Box<[(usize, u32, Cells<'a>); ENV_SIZE]>,
    call_sp: usize,
    cell_vars: Cells<'a>,
}

impl<'a, 'b> Vm<'a, 'b> {
    fn push_stack(&mut self, value: Value<'a>) {
        #[cfg(feature = "debug_print")]
        {
            println!("pushing: {value}");
            println!("     at: {:>5}   {:?}", self.pc, self.bytecode[self.pc]);
        }
        self.stack[self.sp] = value;
        self.sp += 1;
    }

    fn pop_stack(&mut self) -> Value<'a> {
        #[cfg(feature = "debug_print")]
        {
            println!("popping: {}", self.stack[self.sp - 1]);
            println!("     at: {:>5}   {:?}", self.pc, self.bytecode[self.pc]);
        }
        self.sp -= 1;
        self.stack[self.sp]
    }

    fn peek_stack(&self) -> Value<'a> {
        self.stack[self.sp - 1]
    }

    #[expect(unused)]
    fn get_stack_entry(&self, index: u32) -> Value<'a> {
        self.stack[index.cast()]
    }

    fn get_constant(&self, index: u32) -> Value<'a> {
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
    bytecode: &[Bytecode],
    constants: &[Value<'a>],
    metadata: &[Metadata<'a>],
    error_locations: &[ContainingExpression<'a>],
    env: &mut Environment<'a>,
    global_cells: Cells<'a>,
) -> Result<Value<'a>, ControlFlow<Value<'a>, Box<Error<'a>>>> {
    let stack = Box::try_from(vec![Value::Nil; ENV_SIZE].into_boxed_slice()).unwrap();
    let call_stack = Box::try_from(
        vec![(0, 0, Cells::from_iter_in(env.gc, [].into_iter())); ENV_SIZE].into_boxed_slice(),
    )
    .unwrap();

    let mut vm = Vm {
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
    };

    loop {
        #[cfg(feature = "debug_print")]
        {
            println!(
                "{pc:>5}   {bytecode} ({sp})",
                bytecode = vm.bytecode[vm.pc],
                pc = vm.pc,
                sp = vm.sp,
            );
        }

        match vm.bytecode[vm.pc] {
            Pop => {
                vm.pop_stack();
            }
            Const(i) => {
                vm.push_stack(vm.get_constant(i));
            }
            UnaryMinus => {
                let value = match vm.pop_stack() {
                    Number(x) => Number(-x),
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
                vm.push_stack(Bool(!value.is_truthy()));
            }
            Equal => any_binop(&mut vm, |_, lhs, rhs| Ok(Bool(lhs == rhs)))?,
            NotEqual => any_binop(&mut vm, |_, lhs, rhs| Ok(Bool(lhs != rhs)))?,
            Less => number_binop(&mut vm, |lhs, rhs| Bool(lhs < rhs))?,
            LessEqual => number_binop(&mut vm, |lhs, rhs| Bool(lhs <= rhs))?,
            Greater => number_binop(&mut vm, |lhs, rhs| Bool(lhs > rhs))?,
            GreaterEqual => number_binop(&mut vm, |lhs, rhs| Bool(lhs >= rhs))?,
            Add => any_binop(&mut vm, |vm, lhs, rhs| match (lhs, rhs) {
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
            Subtract => number_binop(&mut vm, |lhs, rhs| Number(lhs - rhs))?,
            Multiply => number_binop(&mut vm, |lhs, rhs| Number(lhs * rhs))?,
            Divide => number_binop(&mut vm, |lhs, rhs| Number(lhs / rhs))?,
            Power => number_binop(&mut vm, |lhs, rhs| Number(lhs.powf(rhs)))?,
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
                let assignment_target = vm.pop_stack();
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
                let value = vm.pop_stack();
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
                                .map(|method| match method {
                                    Value::Function(method) => Value::BoundMethod(method, instance),
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
            Call { argument_count, stack_size_at_callsite } => {
                let callee = vm.stack[vm.sp - 1 - argument_count.cast()];
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
                    BoundMethod(method, _instance) => {
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
                        let mut args: Vec<_> =
                            (0..argument_count).map(|_| vm.pop_stack()).collect();
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
                        vm.push_stack(value);
                    }
                    Class(class) => {
                        let instance = GcRef::new_in(
                            vm.env.gc,
                            InstanceInner {
                                class,
                                attributes: RefCell::new(HashMap::default()),
                            },
                        );
                        match class.lookup_method(interned::INIT) {
                            Some(Value::Function(init)) => {
                                if init.parameters.len() - 1 != argument_count.cast() {
                                    Err(Box::new(Error::ArityMismatch {
                                        callee,
                                        expected: init.parameters.len() - 1,
                                        at: vm.error_location(),
                                    }))?
                                }
                                vm.stack[vm.sp - 1 - argument_count.cast()] =
                                    BoundMethod(init, instance);
                                vm.call_stack[vm.call_sp] = (vm.pc, vm.offset, vm.cell_vars);
                                vm.call_sp += 1;
                                vm.pc = init.code_ptr - 1;
                                vm.offset += stack_size_at_callsite;
                                vm.cell_vars = init.cells;
                            }
                            Some(_) => unreachable!(),
                            None if argument_count == 0 => vm.push_stack(Value::Instance(instance)),
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
            Print => {
                let value = vm.pop_stack();
                println!("{value}");
            }
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
                        None => Cell::new(GcRef::new_in(vm.env.gc, Cell::new(Value::Nil))),
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
                vm.push_stack(value);
            }
            End => {
                assert_eq!(vm.sp, 0);
                break Ok(Value::Nil);
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
                    match vm.pop_stack() {
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
                    let base = Class(base);
                    class.methods.values().for_each(|method| {
                        let Function(method) = method
                        else {
                            unreachable!()
                        };
                        method.cells[0].set(GcRef::new_in(vm.env.gc, Cell::new(base)));
                    });
                }
                vm.push_stack(Class(class));
            }
            PrintStack => {
                vm.print_stack();
            }
            b @ BoundMethodGetInstance => match vm.peek_stack() {
                BoundMethod(_, instance) => vm.push_stack(Value::Instance(instance)),
                value => unreachable!(
                    "invalid operand for bytecode `{b}`: {value}, expected `BoundMethod`"
                ),
            },
            Super(name) => {
                let super_class = match vm.pop_stack() {
                    Value::Class(class) => class,
                    value => unreachable!("invalid base class value: {value}"),
                };
                let this = match vm.pop_stack() {
                    Value::Instance(this) => this,
                    value => unreachable!("`this` is not an instance: {value}"),
                };
                let value = super_class
                    .lookup_method(name)
                    .map(|method| match method {
                        Value::Function(method) => Value::BoundMethod(method, this),
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
                vm.push_stack(value);
            }
        }
        vm.pc += 1;
    }
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
    let rhs = vm.pop_stack();
    let lhs = vm.pop_stack();
    let result = op(vm, lhs, rhs)?;
    vm.push_stack(result);
    Ok(())
}

#[inline(always)]
#[track_caller]
fn number_binop<'a, 'b>(
    vm: &mut Vm<'a, 'b>,
    op: impl FnOnce(f64, f64) -> Value<'a>,
) -> Result<(), Box<Error<'a>>> {
    let rhs = vm.pop_stack();
    let lhs = vm.pop_stack();
    let result = match (lhs, rhs) {
        (Number(lhs), Number(rhs)) => op(lhs, rhs),
        _ => {
            let expr = vm.error_location();
            Err(Error::InvalidBinaryOp { at: expr, lhs, op: expr.op, rhs })?
        }
    };
    vm.push_stack(result);
    Ok(())
}
