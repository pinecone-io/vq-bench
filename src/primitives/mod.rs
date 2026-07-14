//! Primitives: the composable pipeline stages, grouped into `conditioners`
//! (affine/orthogonal transforms passing a residual downstream) and `rounders`
//! (terminal casts to a finite codeword set).

pub mod catalog;
mod conditioners;
mod rounders;

pub use conditioners::{AbsMax, MinMax, RandomRotate};
pub use rounders::CastUint;
