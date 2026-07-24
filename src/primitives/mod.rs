//! Primitives: the composable pipeline stages, grouped into `conditioners`
//! (affine/orthogonal transforms passing a residual downstream) and `rounders`
//! (terminal casts to a finite codeword set).

pub mod catalog;
mod conditioners;
mod rounders;

pub use conditioners::{
    AbsMax, Center, MinMax, MinMaxDim, Normalize, OptimizeSigns, RandomHadamard, RandomRotate,
    Scale,
};
pub use rounders::{CastAngular, CastHamming, CastNormal, CastSign, CastUint, NormalScale};
