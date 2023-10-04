use std::cell::Cell;
use std::rc::Rc;

/// # Safety
///
/// This is sound to implement at least for types that are `Copy`, types that
/// perform shallow clones such as `Rc`, and types with derived `Clone`
/// instances that only contain types for which `CloneInCellSafe` would be sound
/// to implement.
pub unsafe trait CloneInCellSafe: Clone {}

unsafe impl<T> CloneInCellSafe for Rc<T> {}

pub trait GetClone {
    type Target;

    fn get_clone(&self) -> Self::Target;
}

impl<T> GetClone for Cell<T>
where
    T: CloneInCellSafe,
{
    type Target = T;

    fn get_clone(&self) -> Self::Target {
        unsafe { (*self.as_ptr()).clone() }
    }
}
