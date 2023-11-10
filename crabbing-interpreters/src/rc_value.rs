use std::ops::Deref;
use std::rc::Rc;

pub struct RcValue<T>(Rc<T>);

impl<T> RcValue<T> {
    pub fn new(value: T) -> Self {
        Self(Rc::new(value))
    }

    pub fn as_ptr(this: &Self) -> *const T {
        Rc::as_ptr(&this.0)
    }
}

impl<T> Clone for RcValue<T> {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl<T> Deref for RcValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> PartialEq for RcValue<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for RcValue<T> {}

impl<T> PartialOrd for RcValue<T> {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        None
    }
}
