//! Rounders: terminal stages that cast vectors to a finite codeword set. The `cast`
//! variants live one per file, each self-contained.

use ndarray::ArrayView2;

use crate::coding;

/// The code's level count `d`: the child's width when non-terminal, else the input
/// dim stored in the model (rounders store it in `fit` for exactly this fallback).
fn code_dim(model: &[u8], child: Option<ArrayView2<f32>>) -> usize {
    child.map_or_else(|| coding::unpack_model::<usize>(model), |c| c.ncols())
}

primitives! { Primitive:
    cast_uint => CastUint,
    cast_normal => CastNormal,
    cast_angular => CastAngular,
    cast_sign => CastSign,
    cast_hamming => CastHamming,
    kmeans => Kmeans,
}
