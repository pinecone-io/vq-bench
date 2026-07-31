//! The accessors over the quantizer registry, driving validation, `vqb show q`,
//! and instantiation. The registry ([`quantizers`]) is assembled by the `quantizers!`
//! macro in the parent module, one [`QuantizerSpec`] row per [`Quantizer`] implementor.
//!
//! - *family key*: the lowercase id used in configs/CLI (`minmax`).
//! - *family name*: the display name (`MinMax`).
//! - *method name*: a configured instance, family name + params (`MinMax (b=2)`).

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::{Params, Quantizer};

pub use super::quantizers;

/// One quantizer family's registry row: its trait facts, plus type-erased `build` and
/// `verify_params` entry points. Lifted from the implementor by [`QuantizerSpec::of`].
#[derive(Clone, Copy)]
pub struct QuantizerSpec {
    pub key: &'static str,
    pub family: &'static str,
    pub params: &'static [&'static str],
    pub describe: &'static str,
    pub build: fn(&Params, u64, usize) -> Result<Box<dyn Quantizer>>,
    pub verify: fn(&Params) -> Vec<String>,
}

impl QuantizerSpec {
    /// Family `T`'s registry row.
    pub fn of<T: Quantizer + 'static>() -> Self {
        QuantizerSpec {
            key: T::name(),
            family: T::display_name(),
            params: T::params(),
            describe: T::describe(),
            build: |params, seed, dim| Ok(Box::new(T::build(params, seed, dim)?)),
            verify: T::verify_params,
        }
    }
}

/// Look up a family by its key.
pub fn lookup(key: &str) -> Option<QuantizerSpec> {
    quantizers().into_iter().find(|q| q.key == key)
}

/// Whether `key` is a known family key.
pub fn is_known(key: &str) -> bool {
    lookup(key).is_some()
}

/// The display family name for a family key (`minmax` → `MinMax`); unknown keys
/// pass through unchanged.
pub fn display(key: &str) -> &str {
    lookup(key).map_or(key, |q| q.family)
}

/// A one-line description of a family's pipeline, for `vqb show q`; unknown keys
/// yield an empty string.
pub fn describe(key: &str) -> &'static str {
    lookup(key).map_or("", |q| q.describe)
}

/// Build the quantizer `key` from its params. `seed`/`dim` feed seeded and
/// dim-dependent primitives. Errors on an unknown family or bad param values — the
/// latter is how `validate` surfaces value problems (see `RunConfig::validate`).
pub fn build(key: &str, params: &Params, seed: u64, dim: usize) -> Result<Box<dyn Quantizer>> {
    let spec = lookup(key).ok_or_else(|| anyhow!("no factory for quantizer `{key}`"))?;
    (spec.build)(params, seed, dim).with_context(|| format!("quantizer `{key}`"))
}

/// Param problems for a configured method, checkable without building — the family's
/// [`Quantizer::verify_params`]. An unknown family yields nothing (reported separately
/// by the caller).
pub fn verify_params(key: &str, params: &Params) -> Vec<String> {
    lookup(key).map_or_else(Vec::new, |spec| (spec.verify)(params))
}

/// Read a typed param; `T` is inferred from the call site, so a builder never
/// restates a param's type. This is the single place a param's type is enforced.
pub(crate) fn get<T: FromParam>(params: &Params, key: &str) -> Result<T> {
    let v = params
        .get(key)
        .with_context(|| format!("needs param `{key}`"))?;
    T::from_value(v).with_context(|| format!("param `{key}`"))
}

/// Read an optional typed param, or `default` when the key is absent.
pub(crate) fn get_or<T: FromParam>(params: &Params, key: &str, default: T) -> Result<T> {
    match params.get(key) {
        None => Ok(default),
        Some(v) => T::from_value(v).with_context(|| format!("param `{key}`")),
    }
}

/// A config param type: how to parse one from JSON. Grows one impl per type used.
pub(crate) trait FromParam: Sized {
    fn from_value(v: &Value) -> Result<Self>;
}

impl FromParam for u8 {
    fn from_value(v: &Value) -> Result<u8> {
        let n = v.as_u64().context("must be a non-negative integer")?;
        u8::try_from(n).map_err(|_| anyhow!("={n} out of range 0..=255"))
    }
}

impl FromParam for f32 {
    fn from_value(v: &Value) -> Result<f32> {
        Ok(v.as_f64().context("must be a number")? as f32)
    }
}

impl FromParam for usize {
    fn from_value(v: &Value) -> Result<usize> {
        let n = v.as_u64().context("must be a non-negative integer")?;
        usize::try_from(n).map_err(|_| anyhow!("={n} out of range"))
    }
}

impl FromParam for String {
    fn from_value(v: &Value) -> Result<String> {
        Ok(v.as_str().context("must be a string")?.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `build` resolves a family key; `display` maps it to the display name.
    #[test]
    fn builds_and_displays_by_key() {
        let params: Params = [("b".to_string(), json!(8))].into_iter().collect();
        assert!(build("minmax", &params, 0, 3).is_ok());
        assert!(build("nope", &params, 0, 3).is_err());
        assert_eq!(display("minmax"), "MinMax");
        assert_eq!(display("nope"), "nope");
    }

    /// `verify_params` flags unknown param names without building.
    #[test]
    fn verify_params_flags_unknown_names() {
        let params: Params = [("bits".to_string(), json!(8))].into_iter().collect();
        let problems = verify_params("minmax", &params);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("unknown param `bits`"), "{}", problems[0]);
        assert!(verify_params("minmax", &Params::new()).is_empty());
    }
}
