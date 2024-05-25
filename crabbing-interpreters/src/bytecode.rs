use std::fmt;
use std::ops::Index;

pub(crate) use crate::bytecode::compiler::compile_program;
use crate::bytecode::compiler::Metadata;
use crate::bytecode::vm::execute_bytecode;
pub(crate) use crate::bytecode::vm::run_bytecode;
use crate::bytecode::vm::stack::Stack;
pub(crate) use crate::bytecode::vm::Vm;
use crate::interner::InternedString;
use crate::value::nanboxed;

mod compiler;
mod vm;

macro_rules! bytecode {
    (
        $(#[$attribute:meta])*
        $visibility:vis enum $name:ident {
            $(

                $(#[$variant_attribute:meta])*
                // FIXME: only accept struct variants here to allow generating patterns
                $variant_name:ident $( ( $($ty:ty),* $(,)? ) )?
            ),* $(,)?
        }
    ) => {
        $(
            #[$attribute]
        )*
        $visibility enum $name {
            $(
                $(
                    #[$variant_attribute]
                )*
                $variant_name $( ( $( $ty ,)* ) )?,
            )*
        }

        impl $name {
            #[cfg_attr(not(feature = "count_bytecode_execution"), expect(unused))]
            pub(crate) const fn discriminant(self) -> usize {
                match self {
                    $( $name::$variant_name $( ( $(_ ${ignore($ty)} ,)* ) )? => ${index()}, )*
                }
            }

            pub(crate) const fn all_discriminants() -> [usize; ${count($variant_name)}] {
                [
                    $( ${ignore($variant_name)} ${index()} ),*
                ]
            }

            pub(crate) const fn name(discriminant: usize) -> &'static str {
                match discriminant {
                    $( ${index()} => stringify!($variant_name), )*
                    _ => unreachable!(),
                }
            }

            pub(crate) fn compile(self) -> CompiledBytecode {
                match self {
                    $(
                        $name::$variant_name $( ( $(_ ${ignore($ty)} ,)* ) )? => {
                            #[allow(non_snake_case)]
                            fn $variant_name(vm: &mut Vm, compiled_program: CompiledBytecodes) {
                                let $name::$variant_name $( ( $( $variant_name ${ignore($ty)} ,)* ) )?
                                    = vm.next_bytecode()
                                else {
                                    unsafe {
                                        std::hint::unreachable_unchecked();
                                    }
                                };
                                let result = execute_bytecode(
                                    vm,
                                    $name::$variant_name $( ( $( $variant_name ${ignore($ty)} ,)* ) )?
                                );
                                match result {
                                    Err(value) => vm.set_error(value),
                                    Ok(()) => {
                                        // SAFETY: `compiled_program` has the same length as
                                        // `vm.bytecode` and `vm.pc()` is always in bounds for that
                                        let next_function = unsafe { compiled_program.get(vm.pc()) };
                                        next_function(vm, compiled_program)
                                    }
                                }
                            }
                            $variant_name
                        }
                    )*
                }
            }
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompiledBytecodes<'a>(pub(crate) &'a [CompiledBytecode]);

impl CompiledBytecodes<'_> {
    unsafe fn get(self, index: usize) -> CompiledBytecode {
        unsafe { *self.0.get_unchecked(index) }
    }
}

impl Index<usize> for CompiledBytecodes<'_> {
    type Output = CompiledBytecode;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

type CompiledBytecode = for<'a> fn(&mut Vm<'a, '_>, CompiledBytecodes);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallInner {
    pub argument_count: u32,
    pub stack_size_at_callsite: u32,
}

/// A wrapper around the bytes of an [`f64`]. We do this to keep [`Bytecode`] smaller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Number([u8; 8]);

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Self(value.to_ne_bytes())
    }
}

impl From<Number> for f64 {
    fn from(Number(bytes): Number) -> Self {
        Self::from_ne_bytes(bytes)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", f64::from(*self))
    }
}

bytecode! {
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
        Call(CallInner),
        ShortCall(CallInner),
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
        ConstNil,
        ConstTrue,
        ConstFalse,
        ConstNumber(Number),
    }
}

impl fmt::Display for Bytecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Bytecode::*;
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
            Call(CallInner { argument_count, stack_size_at_callsite }) =>
                write!(f, "call +{stack_size_at_callsite} arity={argument_count}"),
            ShortCall(CallInner { argument_count, stack_size_at_callsite }) => write!(
                f,
                "call +{stack_size_at_callsite} arity={argument_count} (short)",
            ),
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
            ConstNil => write!(f, "const_nil"),
            ConstTrue => write!(f, "const_true"),
            ConstFalse => write!(f, "const_false"),
            ConstNumber(number) => write!(f, "const_number {number}"),
        }
    }
}

fn validate_bytecode(
    bytecodes: &[Bytecode],
    metadata: &[Metadata],
) -> Result<(), vm::InvalidBytecode> {
    let valid_jump_targets = 1..u32::try_from(bytecodes.len()).unwrap();

    if !matches!(bytecodes.last(), Some(Bytecode::End)) {
        return Err(vm::InvalidBytecode::NoEnd);
    }

    for (pc, &bytecode) in bytecodes.iter().enumerate() {
        match bytecode {
            Bytecode::JumpIfTrue(target)
            | Bytecode::JumpIfFalse(target)
            | Bytecode::PopJumpIfTrue(target)
            | Bytecode::PopJumpIfFalse(target)
            | Bytecode::Jump(target)
            | Bytecode::BeginFunction(target) =>
                if !valid_jump_targets.contains(&target) {
                    return Err(vm::InvalidBytecode::JumpOutOfBounds);
                },
            Bytecode::BuildFunction(metadata_index) => {
                let Metadata::Function { function: _, code_size } =
                    metadata[usize::try_from(metadata_index).unwrap()]
                else {
                    unreachable!()
                };
                let code_ptr = u32::try_from(pc)
                    .unwrap()
                    .checked_sub(code_size)
                    .ok_or(vm::InvalidBytecode::JumpOutOfBounds)?;
                if !valid_jump_targets.contains(&code_ptr) {
                    return Err(vm::InvalidBytecode::JumpOutOfBounds);
                }
            }
            Bytecode::ShortCall(CallInner {
                argument_count,
                stack_size_at_callsite: _,
            }) =>
                if usize::try_from(argument_count).unwrap()
                    >= (Stack::<nanboxed::Value>::ELEMENT_COUNT_IN_GUARD_AREA - 1)
                {
                    return Err(vm::InvalidBytecode::TooManyArgsInShortCall);
                },
            Bytecode::Pop
            | Bytecode::Const(_)
            | Bytecode::UnaryMinus
            | Bytecode::UnaryNot
            | Bytecode::Equal
            | Bytecode::NotEqual
            | Bytecode::Less
            | Bytecode::LessEqual
            | Bytecode::Greater
            | Bytecode::GreaterEqual
            | Bytecode::Add
            | Bytecode::Subtract
            | Bytecode::Multiply
            | Bytecode::Divide
            | Bytecode::Power
            | Bytecode::Local(_)
            | Bytecode::Global(_)
            | Bytecode::Cell(_)
            | Bytecode::Dup
            | Bytecode::StoreAttr(_)
            | Bytecode::LoadAttr(_)
            | Bytecode::StoreLocal(_)
            | Bytecode::StoreGlobal(_)
            | Bytecode::StoreCell(_)
            | Bytecode::DefineCell(_)
            | Bytecode::Call(_)
            | Bytecode::Print
            | Bytecode::GlobalByName(_)
            | Bytecode::StoreGlobalByName(_)
            | Bytecode::Return
            | Bytecode::End
            | Bytecode::Pop2
            | Bytecode::BuildClass(_)
            | Bytecode::PrintStack
            | Bytecode::BoundMethodGetInstance
            | Bytecode::Super(_)
            | Bytecode::ConstNil
            | Bytecode::ConstTrue
            | Bytecode::ConstFalse
            | Bytecode::ConstNumber(_) => (),
        }
    }

    Ok(())
}
