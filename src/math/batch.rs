//! Batched array ops over whole vector/query batches. Broadcasting lives here so
//! primitives read as plain math.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};

/// Per-row minimum and maximum of an `(n × d)` batch, in a single pass.
pub fn row_minmax(x: ArrayView2<f32>) -> (Array1<f32>, Array1<f32>) {
    let mut mins = Array1::from_elem(x.nrows(), f32::INFINITY);
    let mut maxs = Array1::from_elem(x.nrows(), f32::NEG_INFINITY);
    Zip::from(x.rows())
        .and(&mut mins)
        .and(&mut maxs)
        .for_each(|row, mn, mx| {
            for &v in row {
                *mn = mn.min(v);
                *mx = mx.max(v);
            }
        });
    (mins, maxs)
}

/// Elementwise reciprocal, guarding zero: `0.0` stays `0.0` (never `inf`/`NaN`).
pub fn reciprocal(v: ArrayView1<f32>) -> Array1<f32> {
    v.mapv(|x| if x != 0.0 { 1.0 / x } else { 0.0 })
}

/// In place per-row affine: `x[i] = scale[i]·x[i] + offset[i]`.
pub fn affine_rows(x: &mut Array2<f32>, scale: ArrayView1<f32>, offset: ArrayView1<f32>) {
    Zip::from(x.rows_mut())
        .and(scale)
        .and(offset)
        .for_each(|mut row, &s, &o| {
            row.mapv_inplace(|v| s * v + o);
        });
}

/// In place per-column affine: `x[:, j] = scale[j]·x[:, j] + offset[j]`. Traversed
/// row-major (the storage order), each row taking the same `scale`/`offset`.
pub fn affine_cols(x: &mut Array2<f32>, scale: ArrayView1<f32>, offset: ArrayView1<f32>) {
    Zip::from(x.rows_mut()).for_each(|mut row| {
        Zip::from(&mut row)
            .and(scale)
            .and(offset)
            .for_each(|v, &s, &o| *v = s * *v + o);
    });
}

/// In place per-row scaling: `x[i] *= scale[i]`.
pub fn scale_rows(x: &mut Array2<f32>, scale: ArrayView1<f32>) {
    Zip::from(x.rows_mut()).and(scale).for_each(|mut row, &s| {
        row.mapv_inplace(|v| v * s);
    });
}

/// In place per-row offset: `x[i] += offset[i]` (the offset broadcasts across the row).
pub fn offset_rows(x: &mut Array2<f32>, offset: ArrayView1<f32>) {
    Zip::from(x.rows_mut()).and(offset).for_each(|mut row, &o| {
        row.mapv_inplace(|v| v + o);
    });
}

/// In place per-column scaling: `s[:, c] *= factors[c]`. Traversed row-major (the
/// storage order) so it stays cache-friendly and vectorizes; each row is multiplied
/// elementwise by the same `factors`.
pub fn scale_cols(s: &mut Array2<f32>, factors: ArrayView1<f32>) {
    Zip::from(s.rows_mut()).for_each(|mut row| {
        Zip::from(&mut row).and(factors).for_each(|v, &f| *v *= f);
    });
}

/// Outer product `a ⊗ b` as an `(a.len() × b.len())` matrix.
pub fn outer(a: ArrayView1<f32>, b: ArrayView1<f32>) -> Array2<f32> {
    a.insert_axis(Axis(1)).dot(&b.insert_axis(Axis(0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn row_minmax_per_row() {
        let x = array![[1., 5., 3.], [-2., 0., -1.]];
        let (mins, maxs) = row_minmax(x.view());
        assert_eq!(mins, array![1., -2.]);
        assert_eq!(maxs, array![5., 0.]);
    }

    #[test]
    fn reciprocal_guards_zero() {
        let v = array![2.0, -4.0, 0.0, 0.5];
        assert_eq!(reciprocal(v.view()), array![0.5, -0.25, 0.0, 2.0]);
    }

    #[test]
    fn affine_rows_scales_each_row() {
        let mut x = array![[1., 2.], [3., 4.]];
        affine_rows(&mut x, array![2., 0.5].view(), array![1., -1.].view());
        assert_eq!(x, array![[3., 5.], [0.5, 1.]]);
    }

    #[test]
    fn affine_cols_scales_each_column() {
        let mut x = array![[1., 2.], [3., 4.]];
        affine_cols(&mut x, array![2., 0.5].view(), array![1., -1.].view());
        assert_eq!(x, array![[3., 0.], [7., 1.]]);
    }

    #[test]
    fn scale_rows_scales_each_row() {
        let mut x = array![[1., 2.], [3., 4.]];
        scale_rows(&mut x, array![2., 0.5].view());
        assert_eq!(x, array![[2., 4.], [1.5, 2.]]);
    }

    #[test]
    fn offset_rows_adds_per_row_constant() {
        let mut x = array![[1., 2.], [3., 4.]];
        offset_rows(&mut x, array![10., -1.].view());
        assert_eq!(x, array![[11., 12.], [2., 3.]]);
    }

    #[test]
    fn scale_cols_scales_each_column() {
        let mut s = array![[1., 2.], [3., 4.]];
        scale_cols(&mut s, array![10., 100.].view());
        assert_eq!(s, array![[10., 200.], [30., 400.]]);
    }

    #[test]
    fn outer_product() {
        assert_eq!(
            outer(array![1., 2., 3.].view(), array![1., -1.].view()),
            array![[1., -1.], [2., -2.], [3., -3.]]
        );
    }
}
