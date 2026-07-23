//! Test-only helpers shared across primitive and quantizer unit tests.

use ndarray::{Array2, ArrayView2};

use crate::{AsQuantizer, Pipeline, Primitive, Quantizer};

/// Borrow a slice of owned codes as the `&[&[u8]]` the trait methods expect.
pub(crate) fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
    codes.iter().map(Vec::as_slice).collect()
}

/// Assert two batches match elementwise within `tol`.
pub(crate) fn assert_close(a: &Array2<f32>, b: &Array2<f32>, tol: f32) {
    assert_eq!(a.dim(), b.dim());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < tol, "{x} vs {y} (tol {tol})");
    }
}

/// Fit `stages` as a pipeline over `v`, then assert its asymmetric `score` equals
/// the exact dot with the pipeline's own reconstruction (within `score_tol`), and --
/// when `recon_tol` is given -- that the reconstruction is within it of `v`.
pub(crate) fn assert_pipeline_scores(
    stages: Vec<Box<dyn Primitive>>,
    v: ArrayView2<f32>,
    q: ArrayView2<f32>,
    recon_tol: Option<f32>,
    score_tol: f32,
) {
    let codec = AsQuantizer(Pipeline::new(v.ncols(), stages).unwrap());
    let model = codec.fit(v, None);
    let codes = codec.encode(&model, v);
    let r = refs(&codes);
    let recon = codec.reconstruct(&model, &r);
    if let Some(tol) = recon_tol {
        assert_close(&recon, &v.to_owned(), tol);
    }
    assert_close(&codec.score(&model, q, &r), &q.dot(&recon.t()), score_tol);
}
