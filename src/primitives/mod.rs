//! Primitives: the composable pipeline stages, grouped into `conditioners`
//! (affine/orthogonal transforms passing a residual downstream) and `rounders`
//! (terminal casts to a finite codeword set).

/// Register a group's primitives under its trait: declare each `module => Type`,
/// re-export its contents, and collect the trait's `name()`/`describe()` into the
/// group's catalog rows, in `vqb show p` order.
macro_rules! primitives {
    ($trait:ident: $($module:ident => $ty:ident),+ $(,)?) => {
        $(mod $module;)+
        $(pub use $module::*;)+
        /// This group's `(name, description)` rows for `vqb show p`.
        pub(super) fn catalog() -> Vec<(&'static str, &'static str)> {
            vec![$((<$ty as crate::$trait>::name(), <$ty as crate::$trait>::describe())),+]
        }
    };
}

pub mod catalog;
mod conditioners;
mod rounders;
mod splitters;

pub use conditioners::*;
pub use rounders::*;
pub use splitters::*;
