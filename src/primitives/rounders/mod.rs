//! Rounders: terminal stages that cast vectors to a finite codeword set. The `cast`
//! variants live one per file, each self-contained.

use ndarray::ArrayView2;

use crate::coding;

/// The code's level count `d`: the child's width when non-terminal, else the input
/// dim stored in the model (rounders store it in `fit` for exactly this fallback).
fn code_dim(model: &[u8], child: Option<ArrayView2<f32>>) -> usize {
    child.map_or_else(|| coding::unpack_model::<usize>(model), |c| c.ncols())
}

mod cast_angular;
mod cast_hamming;
mod cast_normal;
mod cast_sign;
mod cast_uint;
mod kmeans;

pub use cast_angular::CastAngular;
pub use cast_hamming::CastHamming;
pub use cast_normal::{CastNormal, NormalScale};
pub use cast_sign::CastSign;
pub use cast_uint::CastUint;
pub use kmeans::Kmeans;
