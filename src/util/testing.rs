//! Test-only helpers shared across primitive and quantizer unit tests.

use ndarray::Array2;

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
