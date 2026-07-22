//! The accessors over the quantizer registry, driving validation, `vqb show q`,
//! and instantiation. The registry itself (`QUANTIZERS`) is assembled by the
//! `quantizers!` macro in the parent module from each family's `SPEC`. Add a
//! quantizer by writing its builder module (with a `pub const SPEC`) and adding its name to that macro.
//!
//! - *family key*: the lowercase id used in configs/CLI (`minmax`).
//! - *family name*: the display name (`MinMax`).
//! - *method name*: a configured instance, family name + params (`MinMax (b=2)`).

use std::collections::BTreeMap;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::Pipeline;

use super::NamedQuantizer;

pub use super::QUANTIZERS;

/// Builds a family's stage pipeline from its params, the run seed, and the dataset
/// dim (the latter two feed seeded/dim-dependent primitives, e.g. random rotation).
/// A builder validates its own param *values* here (type, range, cross-param) by
/// returning an error; the catalog only tracks accepted param *names* and attaches
/// the display name (see [`build`]).
type BuildFn = fn(&BTreeMap<String, Value>, u64, usize) -> Result<Pipeline>;

/// One quantizer family: its config/CLI key, display name, accepted param names,
/// a one-line description, and how to build its stage pipeline. Each family
/// defines a `pub const SPEC` of this type in its own module.
pub struct QuantizerSpec {
    pub key: &'static str,
    pub family: &'static str,
    pub params: &'static [&'static str],
    pub describe: &'static str,
    pub build: BuildFn,
}

/// Look up a family by its key.
pub fn lookup(key: &str) -> Option<&'static QuantizerSpec> {
    QUANTIZERS.iter().find(|q| q.key == key)
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
pub fn describe(key: &str) -> &str {
    lookup(key).map_or("", |q| q.describe)
}

/// Build the quantizer `key` from its params: the family's stage pipeline, tagged
/// with its display name. `seed`/`dim` feed seeded and dim-dependent primitives.
/// Errors on an unknown family or bad param values — the latter is how `validate`
/// surfaces value problems (see `RunConfig::validate`).
pub fn build(
    key: &str,
    params: &BTreeMap<String, Value>,
    seed: u64,
    dim: usize,
) -> Result<NamedQuantizer> {
    let spec = lookup(key).ok_or_else(|| anyhow!("no factory for quantizer `{key}`"))?;
    let pipeline = (spec.build)(params, seed, dim).with_context(|| format!("quantizer `{key}`"))?;
    Ok(NamedQuantizer {
        name: spec.family.to_string(),
        pipeline,
    })
}

/// Unknown-param problems for a configured method: config keys the family doesn't
/// accept. Value problems (type, range) are reported by `build`, not here. An
/// unknown family yields nothing (reported separately by the caller).
pub fn check_params(key: &str, params: &BTreeMap<String, Value>) -> Vec<String> {
    let Some(spec) = lookup(key) else {
        return Vec::new();
    };
    params
        .keys()
        .filter(|k| !spec.params.contains(&k.as_str()))
        .map(|k| {
            format!(
                "unknown param `{k}` for quantizer `{key}` (accepts: {})",
                spec.params.join(", ")
            )
        })
        .collect()
}

/// Read a typed param; `T` is inferred from the call site, so a builder never
/// restates a param's type. This is the single place a param's type is enforced.
pub(crate) fn get<T: FromParam>(params: &BTreeMap<String, Value>, key: &str) -> Result<T> {
    let v = params
        .get(key)
        .with_context(|| format!("needs param `{key}`"))?;
    T::from_value(v).with_context(|| format!("param `{key}`"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `build` tags the family's pipeline with its display name.
    #[test]
    fn build_names_the_family() {
        let params: BTreeMap<String, Value> = [("b".to_string(), json!(8))].into_iter().collect();
        let q = build("minmax", &params, 0, 3).unwrap();
        assert_eq!(q.name, "MinMax");
    }
}
