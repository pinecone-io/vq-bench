//! The [`Quantizer`] interface and the [`AsQuantizer`] adapter.

use ndarray::{Array2, ArrayView2};

use crate::Primitive;

/// The quantizer interface: fit a model, encode vectors to per-vector codes,
/// decode via [`reconstruct`](Self::reconstruct) / [`score`](Self::score).
pub trait Quantizer {
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

/// Split total encoded size into `(model bytes, code bytes)`. Owned by the harness,
/// not the quantizer, so a quantizer can't misreport its own size.
pub fn byte_split(model: &[u8], codes: &[Vec<u8>]) -> (usize, usize) {
    (model.len(), codes.iter().map(Vec::len).sum())
}

/// [`Primitive`] implements [`Quantizer`].
pub struct AsQuantizer<P>(pub P);

impl<P: Primitive> Quantizer for AsQuantizer<P> {
    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        self.0.fit(vectors, queries)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        self.0.encode(model, vectors)
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        // None: the primitive is the whole chain, so there is no downstream stage
        // feeding in a child reconstruction — this primitive is terminal.
        self.0.reconstruct(model, codes, None)
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        candidate_codes: &[&[u8]],
    ) -> Array2<f32> {
        // None: no downstream stage's scores to fold in (see reconstruct).
        self.0.score(model, queries, candidate_codes, None)
    }
}
