//! Helpers shared across the `cast` rounders: the stored-dim accessors.

/// Maximum supported code width, in bits.
pub(super) const MAX_BITS: u8 = 8;

/// Serialize the input dim as the model's `u32` header (read back by `checked_dim`/`stored_dim`).
pub(super) fn dim_bytes(d: usize) -> Vec<u8> {
    (d as u32).to_le_bytes().to_vec()
}

/// The input dim stored by `fit`, checked against the dim actually seen.
pub(super) fn checked_dim(model: &[u8], seen: usize) -> usize {
    let d = u32::from_le_bytes(model[..4].try_into().unwrap()) as usize;
    debug_assert_eq!(d, seen, "cast model dim {d} != input dim {seen}");
    d
}

/// The input dim stored by `fit` (no batch to check against).
pub(super) fn stored_dim(model: &[u8]) -> usize {
    u32::from_le_bytes(model[..4].try_into().unwrap()) as usize
}
