//! vq-bench — a vector-quantization benchmark.
//!
//! Quantizers are built by composing [`Primitive`] stages.
//! The harness only interacts with the [`Quantizer`] interface.

pub(crate) mod math;
mod pipeline;
mod primitive;
mod primitives;
mod quantizer;
mod quantizers;
mod splitter;
mod util;

pub(crate) use util::coding;

pub use math::matmul;
pub use pipeline::Pipeline;
pub use primitive::Primitive;
pub use primitives::{catalog as primitive_catalog, AbsMax, CastUint, MinMax, RandomRotate};
pub use quantizer::{byte_split, AsQuantizer, Quantizer};
pub use quantizers::{catalog, NamedQuantizer};
pub use splitter::Splitter;
