//! The catalog of in-tree quantizers — the source of truth for which quantizer
//! **family keys** an experiment config may reference, and their display
//! **family names**.
//!
//! - *family key*: the lowercase id used in configs/CLI (`minmax`).
//! - *family name*: the display name (`MinMax`).
//! - *method name*: a configured instance, family name + params (`MinMax (b=2)`).

/// Every quantizer family key the harness can build. Grows as quantizers are added.
pub fn names() -> &'static [&'static str] {
    &["minmax"]
}

/// Whether `key` is a known family key.
pub fn is_known(key: &str) -> bool {
    names().contains(&key)
}

/// The display family name for a family key (`minmax` → `MinMax`); unknown keys
/// pass through unchanged.
pub fn display(key: &str) -> &str {
    match key {
        "minmax" => "MinMax",
        other => other,
    }
}

/// A one-line description of a family's pipeline, for `vqb show q`. Grows as
/// quantizers are added; unknown keys yield an empty string.
pub fn describe(key: &str) -> &str {
    match key {
        "minmax" => "MinMax → CastUint(b): per-vector rescale to [0,1], then a b-bit uniform lattice",
        _ => "",
    }
}
