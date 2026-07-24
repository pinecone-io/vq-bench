//! faer-backed linear algebra over ndarray matrices.

use faer::{Mat, MatRef};
use ndarray::{Array1, Array2, ArrayView2};

/// Borrow a contiguous row-major slice as a faer matrix view.
fn as_faer(data: &[f32], nrows: usize, ncols: usize) -> MatRef<'_, f32> {
    MatRef::from_row_major_slice(data, nrows, ncols)
}

/// Copy a faer matrix into an ndarray.
fn from_faer(m: MatRef<'_, f32>) -> Array2<f32> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Matrix product `a · b`.
pub fn matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let fa = as_faer(a.as_slice().unwrap(), a.nrows(), a.ncols());
    let fb = as_faer(b.as_slice().unwrap(), b.nrows(), b.ncols());
    let c: Mat<f32> = fa * fb;
    from_faer(c.as_ref())
}

/// Orthogonal factor `Q` of the QR decomposition of `a`.
pub fn qr_q(a: ArrayView2<f32>) -> Array2<f32> {
    let a = a.as_standard_layout();
    let fa = as_faer(a.as_slice().unwrap(), a.nrows(), a.ncols());
    let q = fa.qr().compute_Q();
    from_faer(q.as_ref())
}

/// Thin SVD `a = U * diag(s) * V^T`. Returns `(U, s, V^T)` with `U` (m x k),
/// `s` (k, descending), `V^T` (k x n), `k = min(m, n)`.
pub fn svd(a: ArrayView2<f32>) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let a = a.as_standard_layout();
    let fa = as_faer(a.as_slice().unwrap(), a.nrows(), a.ncols());
    let svd = fa.thin_svd().expect("SVD failed to converge");
    let u = from_faer(svd.U());
    let vt = from_faer(svd.V().transpose());
    let s_diag = svd.S();
    let s = Array1::from_shape_fn(s_diag.dim(), |i| s_diag[i]);
    (u, s, vt)
}

/// Nearest orthogonal matrix to `cross` (orthogonal Procrustes): `U * V^T` from
/// `svd(cross)`, i.e. `argmax_R tr(R^T * cross)` over orthogonal R.
pub fn orthogonal_procrustes(cross: ArrayView2<f32>) -> Array2<f32> {
    let (u, _, vt) = svd(cross);
    matmul(u.view(), vt.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn matmul_matches_hand_product() {
        let a = array![[1., 2., 3.], [4., 5., 6.]];
        let b = array![[1., 0.], [0., 1.], [1., 1.]];
        let c = matmul(a.view(), b.view());
        assert_eq!(c, array![[4., 5.], [10., 11.]]);
    }

    #[test]
    fn qr_q_is_orthogonal() {
        let a = array![[1., 2., 0.], [0., 1., 1.], [1., 0., 1.]];
        let q = qr_q(a.view());
        let qtq = matmul(q.t(), q.view());
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expect).abs() < 1e-5,
                    "QᵀQ[{i},{j}] = {}",
                    qtq[[i, j]]
                );
            }
        }
    }
}
