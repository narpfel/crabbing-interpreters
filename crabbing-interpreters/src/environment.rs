use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

use rustc_hash::FxHashMap as HashMap;

use crate::eval::Error;
use crate::gc::Gc;
use crate::gc::GcRef;
use crate::gc::Trace as _;
use crate::interner::interned;
use crate::interner::InternedString;
use crate::parse::Name;
use crate::scope::Slot;
use crate::scope::Target;
use crate::value::Cells;
use crate::value::NativeError;
use crate::value::Value;

#[cfg(not(miri))]
const ENV_SIZE: usize = 100_000;

#[cfg(miri)]
const ENV_SIZE: usize = 1_000;

pub struct Environment<'a> {
    stack: Box<[Value<'a>; ENV_SIZE]>,
    globals: HashMap<InternedString, usize>,
    is_global_defined: Box<[bool]>,
    pub(crate) gc: &'a Gc,
    global_cells: Cells<'a>,
}

impl<'a> Environment<'a> {
    pub fn new(
        gc: &'a Gc,
        globals: HashMap<InternedString, usize>,
        global_cells: Cells<'a>,
    ) -> Self {
        let mut stack: Box<[Value<'a>; ENV_SIZE]> = vec![Value::Nil; ENV_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let mut is_global_defined = vec![false; globals.len()].into_boxed_slice();
        if let Some(&slot) = globals.get(&interned::CLOCK) {
            stack[slot] = Value::NativeFunction(|arguments| {
                if !arguments.is_empty() {
                    return Err(NativeError::ArityMismatch { expected: 0 });
                }
                static START_TIME: OnceLock<Instant> = OnceLock::new();
                Ok(Value::Number(
                    START_TIME.get_or_init(Instant::now).elapsed().as_secs_f64(),
                ))
            });
            is_global_defined[slot] = true;
        }
        Self {
            stack,
            globals,
            is_global_defined,
            gc,
            global_cells,
        }
    }

    pub(crate) fn get(&self, cell_vars: Cells<'a>, offset: usize, slot: Slot) -> Value<'a> {
        let index = match slot {
            Slot::Local(slot) => offset + slot,
            Slot::Global(slot) => slot,
            Slot::Cell(slot) => return cell_vars[slot].get().get(),
        };
        self.stack[index]
    }

    pub(crate) fn get_global_slot_by_name(
        &self,
        name: &'a Name<'a>,
    ) -> Result<usize, Box<Error<'a>>> {
        match self.globals.get(&name.id()).copied() {
            Some(slot) if self.is_global_defined[slot] => Ok(slot),
            _ => Err(Box::new(Error::UndefinedName { at: *name })),
        }
    }

    pub(crate) fn get_global_by_name(
        &self,
        name: &'a Name<'a>,
    ) -> Result<(usize, Value<'a>), Box<Error<'a>>> {
        let slot = self.get_global_slot_by_name(name)?;
        Ok((slot, self.stack[slot]))
    }

    fn define_impl(
        &mut self,
        cell_vars: Cells<'a>,
        offset: usize,
        target: Target,
        value: Value<'a>,
        set_cell: impl FnOnce(&Cell<GcRef<'a, Cell<Value<'a>>>>, Value<'a>),
    ) {
        let index = match target {
            Target::Local(slot) => offset + slot,
            Target::GlobalByName => unreachable!(),
            Target::GlobalBySlot(slot) => {
                // FIXME: this is only necessary when called from `define`, not `set`.
                if let Some(is_defined) = self.is_global_defined.get_mut(slot) {
                    *is_defined = true;
                }
                slot
            }
            Target::Cell(slot) => {
                set_cell(&cell_vars[slot], value);
                return;
            }
        };
        self.stack[index] = value;
    }

    pub(crate) fn define(
        &mut self,
        cell_vars: Cells<'a>,
        offset: usize,
        target: Target,
        value: Value<'a>,
    ) {
        self.define_impl(cell_vars, offset, target, value, |cell, value| {
            cell.set(GcRef::new_in(self.gc, Cell::new(value)))
        })
    }

    pub(crate) fn set(
        &mut self,
        cell_vars: Cells<'a>,
        offset: usize,
        target: Target,
        value: Value<'a>,
    ) {
        self.define_impl(cell_vars, offset, target, value, |cell, value| {
            cell.get().set(value)
        })
    }

    pub(crate) fn collect_if_necessary(&self, last_value: Value<'a>, cell_vars: Cells<'a>) {
        if self.gc.collection_necessary() {
            last_value.trace(&|root| self.gc.mark_recursively(root));
            for value in self.stack.iter() {
                value.trace(&|root| self.gc.mark_recursively(root));
            }
            self.global_cells
                .trace(&|root| self.gc.mark_recursively(root));
            cell_vars.trace(&|root| self.gc.mark_recursively(root));
            unsafe {
                self.gc.sweep();
            }
        }
    }
}
