//! The [`Quantizer`] trait and the [`AsQuantizer`] adapter.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use ndarray::{Array2, ArrayView2};
use serde_json::Value;

use crate::primitive::type_display_name;
use crate::Pipeline;

/// A method's config params.
pub type Params = BTreeMap<String, Value>;

/// A quantizer: its config identity (key, display name, params, description), how to
/// build itself from config params, and the runtime interface the harness drives —
/// `fit`/`encode`/`reconstruct`/`score`. Most quantizers hold a [`Pipeline`] and
/// implement the four with [`pipeline_quantizer!`], but any direct implementation is
/// equally valid — nothing requires a `Pipeline`. `Send + Sync` so the runner can
/// encode row chunks across threads.
pub trait Quantizer: Send + Sync {
    /// The config/CLI key (`"minmax"`).
    fn name() -> &'static str
    where
        Self: Sized;

    /// The display name (`"MinMax"`). Default: the type's name.
    fn display_name() -> &'static str
    where
        Self: Sized,
    {
        type_display_name::<Self>()
    }

    /// The accepted param names. Default: none.
    fn params() -> &'static [&'static str]
    where
        Self: Sized,
    {
        &[]
    }

    /// One-line pipeline description for `vqb show q`.
    fn describe() -> &'static str
    where
        Self: Sized;

    /// Build from config params; `seed` and `dim` feed seeded and dim-dependent
    /// stages. Validates its own param *values* (type, range, cross-param) by erroring.
    fn build(params: &Params, seed: u64, dim: usize) -> Result<Self>
    where
        Self: Sized;

    /// Param problems checkable without building: config keys the quantizer doesn't
    /// accept. Value problems are reported by [`build`](Self::build), not here.
    fn verify_params(params: &Params) -> Vec<String>
    where
        Self: Sized,
    {
        params
            .keys()
            .filter(|k| !Self::params().contains(&k.as_str()))
            .map(|k| {
                format!(
                    "unknown param `{k}` for quantizer `{}` (accepts: {})",
                    Self::name(),
                    Self::params().join(", ")
                )
            })
            .collect()
    }

    /// Learn a model from `vectors` and an optional query sample.
    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8>;

    /// Using the model, encode vectors into per-vector codes.
    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>>;

    /// Reconstruct one vector per code.
    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32>;

    /// Estimate each query against each candidate code.
    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        candidate_codes: &[&[u8]],
    ) -> Array2<f32>;
}

/// Implement the four [`Quantizer`] runtime methods by delegating to the [`Pipeline`]
/// in `self.0` — the standard `pub struct Family(pub Pipeline);` shape. A family that
/// computes an operation differently writes that method (or all four) directly.
macro_rules! pipeline_quantizer {
    () => {
        fn fit(
            &self,
            vectors: ::ndarray::ArrayView2<f32>,
            queries: Option<::ndarray::ArrayView2<f32>>,
        ) -> Vec<u8> {
            $crate::Primitive::fit(&self.0, vectors, queries)
        }

        fn encode(&self, model: &[u8], vectors: ::ndarray::ArrayView2<f32>) -> Vec<Vec<u8>> {
            $crate::Primitive::encode(&self.0, model, vectors)
        }

        fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> ::ndarray::Array2<f32> {
            // None: the pipeline is the whole chain -- no downstream stage feeds in.
            $crate::Primitive::reconstruct(&self.0, model, codes, None)
        }

        fn score(
            &self,
            model: &[u8],
            queries: ::ndarray::ArrayView2<f32>,
            codes: &[&[u8]],
        ) -> ::ndarray::Array2<f32> {
            $crate::Primitive::score(&self.0, model, queries, codes, None)
        }
    };
}

pub(crate) use pipeline_quantizer;

/// Split total encoded size into `(model bytes, code bytes)`. Owned by the harness,
/// not the quantizer, so a quantizer can't misreport its own size.
pub fn byte_split(model: &[u8], codes: &[Vec<u8>]) -> (usize, usize) {
    (model.len(), codes.iter().map(Vec::len).sum())
}

/// A bare [`Pipeline`] run through the [`Quantizer`] interface, for tests and ad-hoc
/// chains; it has no config identity and cannot be built from params.
pub struct AsQuantizer(pub Pipeline);

impl Quantizer for AsQuantizer {
    fn name() -> &'static str {
        "pipeline"
    }

    fn describe() -> &'static str {
        "a bare stage pipeline"
    }

    fn build(_params: &Params, _seed: u64, _dim: usize) -> Result<Self> {
        Err(anyhow!("a bare pipeline takes no params; wrap one directly"))
    }

    crate::pipeline_quantizer!();
}
