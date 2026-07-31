//! The [`Splitter`] trait — a fan-out stage.

use ndarray::{Array2, ArrayView2};

/// Split each vector into [`n_branches`](Self::n_branches)
/// pieces, quantize each independently, then recombine. `Send + Sync` so a
/// [`Split`](crate::Split) stage stays thread-safe for the runner's parallel encode.
pub trait Splitter: Send + Sync {
    /// The display name `vqb show p` prints. Default: the type's name.
    fn name() -> &'static str
    where
        Self: Sized,
    {
        crate::primitive::type_display_name::<Self>()
    }

    /// One-line description for `vqb show p`.
    fn describe() -> &'static str
    where
        Self: Sized;

    /// Number of branches each vector fans out to.
    fn n_branches(&self) -> usize;

    /// Fit model on vectors, and optionally queries. Default: no model.
    fn fit(&self, _vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        Vec::new()
    }

    /// Generate per-vector codes. Default: an empty code per vector (a splitter
    /// that owns no per-vector bits). Pair with `code_bytes == Some(0)`.
    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        vec![Vec::new(); vectors.nrows()]
    }

    /// Fixed byte length of the node's own per-vector code, or `None` if it
    /// varies.
    fn code_bytes(&self, in_dim: usize) -> Option<usize>;

    /// Split the vectors into one sub-batch per branch.
    fn apply(&self, model: &[u8], vectors: ArrayView2<f32>, codes: &[&[u8]]) -> Vec<Array2<f32>>;

    /// Split the query batch into one sub-batch per branch.
    fn apply_queries(&self, model: &[u8], queries: ArrayView2<f32>) -> Vec<Array2<f32>>;

    /// Recombine the per-branch reconstructions into the parent vectors.
    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: &[Array2<f32>],
    ) -> Array2<f32>;

    /// Combine the per-branch score matrices into parent scores.
    fn score(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        query: ArrayView2<f32>,
        child_scores: &[Array2<f32>],
    ) -> Array2<f32>;

    /// Input dim that branch `branch` receives, given the parent input dim.
    fn branch_in_dim(&self, model: &[u8], in_dim: usize, branch: usize) -> usize;
}
