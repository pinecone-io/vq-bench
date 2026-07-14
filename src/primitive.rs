//! The [`Primitive`] trait.

use ndarray::{Array2, ArrayView2};

/// One stage of a quantization pipeline. `Send + Sync` so the runner can encode
/// row chunks across threads (a stage holds only immutable config; models are
/// passed in as bytes).
pub trait Primitive: Send + Sync {
    /// Learn this stage's model from `vectors` (already transformed by upstream
    /// stages) and an optional query sample. Return the serialized model.
    /// Default: no model — a stateless stage (empty bytes).
    fn fit(&self, _vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        Vec::new()
    }

    /// Produce the codes for each vector. Default: an empty code per vector
    /// (a stage that owns no per-vector bits). Pair with `code_bytes == Some(0)`.
    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        vec![Vec::new(); vectors.nrows()]
    }

    /// Transform the batch of vectors using their codes and the model.
    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]);

    /// Transform the batch of queries using the model. Default: no-op (identity).
    fn apply_queries(&self, _model: &[u8], _queries: &mut Array2<f32>) {}

    /// Rebuild the vectors from their codes, the model, and the next stage's reconstruction.
    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32>;

    /// Estimate query-vector scores from the vector codes, the model, and the next stage's scores.
    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32>;

    /// Dimensionality the next stage receives, given this stage's input dim.
    fn out_dim(&self, in_dim: usize) -> usize {
        in_dim
    }

    /// Fixed byte length of this stage's per-vector code, or `None` if it varies
    /// (the caller then length-prefixes it). Depends only on dim and
    /// configuration, never on data.
    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        None
    }
}
