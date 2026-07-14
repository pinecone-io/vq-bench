//! Instantiate a quantizer from a resolved method config. Grows with the catalog.

use anyhow::{bail, Context, Result};
use vqb::NamedQuantizer;

use crate::config::ResolvedMethod;

/// Build the quantizer named by `m`, reading its parameters. `seed` is the
/// run master seed (passed to seeded primitives like random rotations);
/// `dim` is the dataset vector dimension (needed by estimators whose scale depends
/// on it, e.g. QJL).
pub fn build(m: &ResolvedMethod, _seed: u64, _dim: usize) -> Result<NamedQuantizer> {
    match m.name.as_str() {
        "minmax" => Ok(vqb::minmax(u8_param(m, "b")?)),
        other => bail!("no factory for quantizer `{other}`"),
    }
}

/// Read an integer parameter as `u8`.
fn u8_param(m: &ResolvedMethod, key: &str) -> Result<u8> {
    let v = m
        .params
        .get(key)
        .with_context(|| format!("`{}` needs param `{key}`", m.name))?;
    let n = v
        .as_u64()
        .with_context(|| format!("`{key}` must be a non-negative integer"))?;
    u8::try_from(n).with_context(|| format!("`{key}`={n} out of range (0..=255)"))
}
