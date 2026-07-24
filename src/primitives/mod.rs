//! Primitives: the composable pipeline stages, grouped into `conditioners`
//! (affine/orthogonal transforms passing a residual downstream) and `rounders`
//! (terminal casts to a finite codeword set).

pub mod catalog;
mod conditioners;
mod rounders;
mod splitters;

pub use conditioners::{
    AbsMax, Center, MinMax, MinMaxDim, Normalize, OptimizePq, OptimizeSigns, RandomHadamard,
    RandomRotate, Scale,
};
pub use rounders::{CastAngular, CastHamming, CastNormal, CastSign, CastUint, Kmeans, NormalScale};
pub use splitters::{SegmentSplit, Split};
