//! faer-backed linear algebra over ndarray matrices.

use faer::{Mat, MatRef};
use ndarray::{Array1, Array2, ArrayView2, Axis};

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

/// Gram matrix `a^T · a`, `(cols × cols)`.
pub fn gram(a: ArrayView2<f32>) -> Array2<f32> {
    let a = a.as_standard_layout();
    let fa = as_faer(a.as_slice().unwrap(), a.nrows(), a.ncols());
    // Transpose inside faer, which reads any stride; `matmul(a.t(), a)` would copy `a`.
    let c: Mat<f32> = fa.transpose() * fa;
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

/// Eigenvalues (descending) and matching eigenvector columns of a symmetric matrix.
pub fn symmetric_eigen(sym: ArrayView2<f32>) -> (Array1<f32>, Array2<f32>) {
    // For symmetric `A = U * diag(s) * V^T` the singular vectors are the eigenvectors,
    // and a `V` column that flipped against its `U` column marks a negative eigenvalue.
    // The singular values sort by magnitude, so the signed values re-sort.
    let (u, s, vt) = svd(sym);
    let signed = Array1::from_shape_fn(s.len(), |i| s[i] * u.column(i).dot(&vt.row(i)).signum());
    let mut order: Vec<usize> = (0..signed.len()).collect();
    order.sort_by(|&a, &b| signed[b].total_cmp(&signed[a]));
    (
        Array1::from_iter(order.iter().map(|&i| signed[i])),
        u.select(Axis(1), &order),
    )
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

    /// `gram` skips the transpose copy `matmul` would make, so it must still agree with it.
    #[test]
    fn gram_matches_matmul_of_the_transpose() {
        let a = array![[1., 2., 3.], [4., 5., 6.]];
        assert_eq!(gram(a.view()), matmul(a.t(), a.view()));
    }

    /// An indefinite symmetric matrix: descending eigenvalues, orthonormal eigenvectors,
    /// and `U * diag(lambda) * U^T` back to the input.
    #[test]
    fn symmetric_eigen_reconstructs_indefinite_input() {
        let a = array![[2., 1., 0.], [1., -3., 1.], [0., 1., 1.]];
        let (vals, vecs) = symmetric_eigen(a.view());
        assert!(vals[0] >= vals[1] && vals[1] >= vals[2], "descending: {vals}");
        assert!(vals[2] < 0.0, "input is indefinite: {vals}");
        let vtv = matmul(vecs.t(), vecs.view());
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((vtv[[i, j]] - expect).abs() < 1e-5, "UᵀU[{i},{j}] = {}", vtv[[i, j]]);
            }
        }
        let scaled = &vecs * &vals.broadcast((3, 3)).unwrap();
        let rebuilt = matmul(scaled.view(), vecs.t());
        for i in 0..3 {
            for j in 0..3 {
                assert!((rebuilt[[i, j]] - a[[i, j]]).abs() < 1e-4, "[{i},{j}] = {}", rebuilt[[i, j]]);
            }
        }
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
