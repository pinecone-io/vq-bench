//! Rounders: terminal stages that cast vectors to a finite codeword set. The `cast`
//! variants live one-per-file and share `cast_common` helpers.

mod cast_common;
mod cast_uint;

pub use cast_uint::CastUint;
