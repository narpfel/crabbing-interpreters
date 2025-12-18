use itertools::Itertools as _;

use crate::environment::Environment;
use crate::gc::GcStr;
use crate::interner::interned;
use crate::value::instance::InstanceInner;
use crate::value::Instance;
use crate::value::NativeError;
use crate::value::Value as Unboxed;

#[expect(clippy::result_large_err)]
pub(super) fn read_file<'a>(
    env: &Environment<'a>,
    arguments: Vec<Unboxed<'a>>,
) -> Result<Unboxed<'a>, NativeError<'a>> {
    match &arguments[..] {
        [Unboxed::String(filename)] => Ok(Unboxed::String(GcStr::new_in(
            env.gc,
            &std::fs::read_to_string(&**filename).map_err(|error| NativeError::IoError {
                error,
                filename: (**filename).to_owned(),
            })?,
        ))),
        arguments => Err(NativeError::TypeError {
            expected: "[String]".to_owned(),
            tys: format!("[{}]", arguments.iter().map(|arg| arg.typ()).join(", ")),
        }),
    }
}

fn split_once<'a>(string: &'a str, delimiter: &str) -> Option<&'a str> {
    match delimiter.is_empty() {
        true if string.is_empty() => None,
        true => Some(&string[..string.ceil_char_boundary(1)]),
        false => Some(string.split_once(delimiter)?.0),
    }
}

// TODO: decide whether `split("aba", "b")` and `split("abab", "b")` should behave the
// same or differently
#[expect(clippy::result_large_err)]
pub(super) fn split<'a>(
    env: &Environment<'a>,
    arguments: Vec<Unboxed<'a>>,
) -> Result<Unboxed<'a>, NativeError<'a>> {
    match &arguments[..] {
        [Unboxed::String(string), Unboxed::String(delimiter)] => {
            let state = InstanceInner::new(env.builtin_split_stack);
            let state = Instance::new_in(env.gc, state);
            state.setattr(
                interned::DELIMITER,
                Unboxed::String(*delimiter).into_nanboxed(),
            );
            state.setattr(interned::STRING, Unboxed::String(*string).into_nanboxed());
            let (split, start) = match split_once(string, delimiter) {
                Some(split) => (GcStr::new_in(env.gc, split), split.len() + delimiter.len()),
                None => (*string, string.len()),
            };
            state.setattr(interned::SPLIT, Unboxed::String(split).into_nanboxed());
            #[expect(
                clippy::as_conversions,
                reason = "TODO: check that this does not round"
            )]
            state.setattr(
                interned::START,
                Unboxed::Number(start as f64).into_nanboxed(),
            );

            Ok(Unboxed::Instance(state))
        }
        [Unboxed::Instance(state)] => {
            let string: GcStr = state.getattr(interned::STRING)?.parse().try_into()?;
            let start: f64 = state.getattr(interned::START)?.parse().try_into()?;
            #[expect(clippy::as_conversions, reason = "TODO: check that this fits")]
            let start = start as usize;
            let delimiter: GcStr = state.getattr(interned::DELIMITER)?.parse().try_into()?;
            let string = &string[start..];
            let (split, start) = match split_once(string, &delimiter) {
                Some(split) => (
                    Unboxed::String(GcStr::new_in(env.gc, split)),
                    start + split.len() + delimiter.len(),
                ),
                None => {
                    let split = match string {
                        "" => Unboxed::Nil,
                        _ => Unboxed::String(GcStr::new_in(env.gc, string)),
                    };
                    (split, start + string.len())
                }
            };
            state.setattr(interned::SPLIT, split.into_nanboxed());
            #[expect(
                clippy::as_conversions,
                reason = "TODO: check that this does not round"
            )]
            state.setattr(
                interned::START,
                Unboxed::Number(start as f64).into_nanboxed(),
            );

            Ok(Unboxed::Nil)
        }
        arguments => Err(NativeError::TypeError {
            expected: "[String, String] | [SplitState]".to_owned(),
            tys: format!("[{}]", arguments.iter().map(|arg| arg.typ()).join(", ")),
        }),
    }
}
