use std::fmt;
use std::ops::Index;

pub(crate) use crate::bytecode::compiler::compile_program;
use crate::bytecode::vm::execute_bytecode;
pub(crate) use crate::bytecode::vm::run_bytecode;
pub(crate) use crate::bytecode::vm::Vm;
use crate::eval::Error;
use crate::interner::InternedString;

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
            pub(crate) fn compile(self) -> CompiledBytecode {
                match self {
                    $(
                        $name::$variant_name $( ( $(_ ${ignore($ty)} ,)* ) )? => {
                            #[allow(non_snake_case)]
                            fn $variant_name<'a>(
                                vm: &mut Vm<'a, '_>,
                                compiled_program: CompiledBytecodes,
                                error: &mut Option<Box<Error<'a>>>,
                            ) -> () {
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
                                    Err(value) => *error = value,
                                    Ok(()) => compiled_program[vm.pc()](vm, compiled_program, error),
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

impl Index<usize> for CompiledBytecodes<'_> {
    type Output = CompiledBytecode;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

type CompiledBytecode = for<'a> fn(&mut Vm<'a, '_>, CompiledBytecodes, &mut Option<Box<Error<'a>>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CallInner {
    pub argument_count: u32,
    pub stack_size_at_callsite: u32,
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
