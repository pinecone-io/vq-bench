//! Structured orthonormal transforms in O(d log d): the Walsh-Hadamard transform
//! and the Kac butterfly. Both are normalized to be their own inverse (involutions).

use ndarray::{Array2, ArrayViewMut2};

/// In-place normalized Walsh-Hadamard transform per row: x --> (1/sqrt(n)) H_n x,
/// where n is the block width (must be a power of two). H_n is symmetric with
/// H_n^2 = n I, so (1/sqrt(n)) H_n is orthonormal and its own inverse.
pub fn hadamard(mut block: ArrayViewMut2<f32>) {
    let n = block.ncols();
    assert!(n.is_power_of_two(), "hadamard block width must be a power of two");
    let scale = 1.0 / (n as f32).sqrt();
    for mut row in block.rows_mut() {
        // A block row is contiguous; a &mut [f32] lets the butterfly vectorize.
        let row = row.as_slice_mut().expect("hadamard row must be contiguous");
        // The WHT's stride factors commute, so fold the small strides (1,2,4) into
        // one register-resident 8-point pass -- a single memory sweep instead of
        // three poorly-vectorized ones -- then run the large strides.
        let mut h = 1;
        if n >= 8 {
            for chunk in row.chunks_exact_mut(8) {
                wht8(chunk.try_into().unwrap());
            }
            h = 8;
        }
        // Iterative butterfly: at stride h, each 2h-block splits into two contiguous
        // h-halves combined as (a+b, a-b). split_at_mut gives the compiler two
        // non-aliasing slices, so these passes vectorize.
        while h < n {
            for block in row.chunks_mut(2 * h) {
                let (a, b) = block.split_at_mut(h);
                for (x, y) in a.iter_mut().zip(b.iter_mut()) {
                    let (u, v) = (*x, *y);
                    *x = u + v;
                    *y = u - v;
                }
            }
            h *= 2;
        }
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

/// Unrolled unnormalized 8-point Walsh-Hadamard transform (strides 1, 2, 4).
/// Fixed indices into `[f32; 8]` let the compiler drop bounds checks and keep the
/// whole thing in registers.
#[inline]
fn wht8(x: &mut [f32; 8]) {
    // stride 1
    for i in [0, 2, 4, 6] {
        let (a, b) = (x[i], x[i + 1]);
        x[i] = a + b;
        x[i + 1] = a - b;
    }
    // stride 2
    for i in [0, 4] {
        for j in [i, i + 1] {
            let (a, b) = (x[j], x[j + 2]);
            x[j] = a + b;
            x[j + 2] = a - b;
        }
    }
    // stride 4
    for j in 0..4 {
        let (a, b) = (x[j], x[j + 4]);
        x[j] = a + b;
        x[j + 4] = a - b;
    }
}

/// In-place normalized Kac butterfly per row: split each row into halves a (first
/// d/2) and b (last d/2) and set (a, b) --> ((a+b)/sqrt(2), (a-b)/sqrt(2)). Width
/// must be even. The 1/sqrt(2) makes it orthonormal and its own inverse.
pub fn kac_walk(x: &mut Array2<f32>) {
    let d = x.ncols();
    assert!(d.is_multiple_of(2), "kac_walk needs an even width");
    let half = d / 2;
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    for mut row in x.rows_mut() {
        let row = row.as_slice_mut().expect("kac row must be contiguous");
        let (front, back) = row.split_at_mut(half);
        for (a, b) in front.iter_mut().zip(back.iter_mut()) {
            let (x0, x1) = (*a, *b);
            *a = (x0 + x1) * inv_sqrt2;
            *b = (x0 - x1) * inv_sqrt2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sq_norm(x: &Array2<f32>) -> f32 {
        x.iter().map(|&v| v * v).sum()
    }

    #[test]
    fn hadamard_is_involution_and_isometry() {
        let x = array![[1., 2., 3., 4.], [-1., 0., 2., 5.]];
        let mut y = x.clone();
        hadamard(y.view_mut());
        // norm preserved.
        assert!((sq_norm(&y) - sq_norm(&x)).abs() < 1e-4);
        // applying twice recovers the input.
        hadamard(y.view_mut());
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn hadamard_matches_reference_across_the_base_case() {
        // Reference (1/sqrt(n)) H_n via H[i,j] = (-1)^popcount(i&j). Covers n=16,
        // which exercises the unrolled 8-point base case plus one large-stride pass.
        for n in [8usize, 16, 32] {
            let x: Array2<f32> = Array2::from_shape_fn((3, n), |(r, j)| (r * n + j) as f32 * 0.5);
            let scale = 1.0 / (n as f32).sqrt();
            let want = Array2::from_shape_fn((3, n), |(r, k)| {
                let mut s = 0.0f32;
                for j in 0..n {
                    let sign = if (k & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                    s += sign * x[[r, j]];
                }
                s * scale
            });
            let mut got = x.clone();
            hadamard(got.view_mut());
            for (a, b) in got.iter().zip(want.iter()) {
                assert!((a - b).abs() < 1e-3, "n={n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn kac_walk_is_involution_and_isometry() {
        let x = array![[1., 2., 3., 4., 5., 6.], [0., -1., 2., -3., 4., -5.]];
        let mut y = x.clone();
        kac_walk(&mut y);
        assert!((sq_norm(&y) - sq_norm(&x)).abs() < 1e-4);
        kac_walk(&mut y);
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
