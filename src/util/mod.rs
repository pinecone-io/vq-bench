//! Shared infrastructure used across primitives: byte/bit [`coding`] and fixed
//! scalar-quantizer [`codebooks`].

pub(crate) mod codebooks;
pub(crate) mod coding;
#[cfg(test)]
pub(crate) mod testing;
