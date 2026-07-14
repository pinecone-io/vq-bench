//! Conditioners: affine/orthogonal transforms that pass a residual downstream.

mod absmax;
mod minmax;
mod random_rotate;

pub use absmax::AbsMax;
pub use minmax::MinMax;
pub use random_rotate::RandomRotate;
