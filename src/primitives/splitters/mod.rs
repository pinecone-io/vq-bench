//! Splitters: fan-out stages. A [`Split`] adapts a [`Splitter`] + one child pipeline
//! per branch into a single terminal [`Primitive`].

mod segment;
mod split;

pub use segment::SegmentSplit;
pub use split::Split;
