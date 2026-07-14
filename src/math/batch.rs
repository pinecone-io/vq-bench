//! Batched array ops over whole vector/query batches. Broadcasting lives here so
//! primitives read as plain math.

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

/// Per-row minimum and maximum of an `(n × d)` batch.
pub fn row_minmax(x: ArrayView2<f32>) -> (Array1<f32>, Array1<f32>) {
    let mins = x.map_axis(Axis(1), |r| r.iter().copied().fold(f32::INFINITY, f32::min));
    let maxs = x.map_axis(Axis(1), |r| {
        r.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    });
    (mins, maxs)
}

/// In place per-row affine: `x[i] = scale[i]·x[i] + offset[i]`.
pub fn affine_rows(x: &mut Array2<f32>, scale: &Array1<f32>, offset: &Array1<f32>) {
    Zip::from(x.rows_mut())
        .and(scale)
        .and(offset)
        .for_each(|mut row, &s, &o| {
            row.mapv_inplace(|v| s * v + o);
        });
}

/// In place per-column scaling: `s[:, c] *= factors[c]`.
pub fn scale_cols(s: &mut Array2<f32>, factors: &Array1<f32>) {
    Zip::from(s.columns_mut())
        .and(factors)
        .for_each(|mut col, &f| {
            col.mapv_inplace(|v| v * f);
        });
}

/// Outer product `a ⊗ b` as an `(a.len() × b.len())` matrix.
pub fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    a.view()
        .insert_axis(Axis(1))
        .dot(&b.view().insert_axis(Axis(0)))
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
    fn affine_rows_scales_each_row() {
        let mut x = array![[1., 2.], [3., 4.]];
        affine_rows(&mut x, &array![2., 0.5], &array![1., -1.]);
        assert_eq!(x, array![[3., 5.], [0.5, 1.]]);
    }

    #[test]
    fn scale_cols_scales_each_column() {
        let mut s = array![[1., 2.], [3., 4.]];
        scale_cols(&mut s, &array![10., 100.]);
        assert_eq!(s, array![[10., 200.], [30., 400.]]);
    }

    #[test]
    fn outer_product() {
        assert_eq!(
            outer(&array![1., 2., 3.], &array![1., -1.]),
            array![[1., -1.], [2., -2.], [3., -3.]]
        );
    }
}
