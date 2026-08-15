//! BALANCE_PARTS: reorders dimensions so every part carries the same variance product
//! -
//! Fit: deal the dimensions, largest variance first, into `section_dim`-wide parts
//! Model: the dealt order, one source dimension per output dimension
//! Code for vector x: empty
//! Apply: x_j --> x_order[j]
//! Reconstruct: y_j --> y at the dealt position
//! Score: s --> s  (queries are dealt the same way)
//!
//! A product quantizer's distortion bound is minimized when its parts hold equal
//! variance products (Ge et al. 2013), so this is the deal a downstream splitter needs.

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use crate::{coding, Primitive, SegmentSplit};

/// Variance floor before the log, so a dead dimension deals as very small rather than as
/// `-inf`.
const MIN_VARIANCE: f32 = 1e-20;

pub struct BalanceParts {
    section_dim: usize,
}

impl BalanceParts {
    /// A deal over parts of `section_dim` dimensions (a short last part when the input
    /// dim does not divide evenly), matching `SegmentSplit`'s widths.
    pub fn new(section_dim: usize) -> Self {
        // Fails here rather than in `fit`, where it would reach `SegmentSplit::new`.
        assert!(section_dim > 0);
        Self { section_dim }
    }

    /// The dealt order, read from the model bytes: source dimension per output dimension.
    fn order(model: &[u8]) -> Vec<usize> {
        coding::unpack_model(model)
    }
}

/// Deal dimensions into parts of the given widths so their log-variance sums match:
/// largest variance first, each to the lightest part that still has room.
///
/// Weights are measured up from the smallest variance, which makes the deal
/// scale-invariant. Comparing raw variance products across parts holding different counts
/// is not: once every variance is below 1 (normalized embeddings), a part gets lighter as
/// it fills, so the deal hands one part a contiguous run of the spectrum instead of a
/// balanced mix.
pub(crate) fn balanced_order(variances: ArrayView1<f32>, widths: &[usize]) -> Vec<usize> {
    let mut by_variance: Vec<usize> = (0..variances.len()).collect();
    by_variance.sort_by(|&a, &b| variances[b].total_cmp(&variances[a]));
    let logs: Vec<f32> = variances.iter().map(|&v| v.max(MIN_VARIANCE).ln()).collect();
    let smallest = logs.iter().copied().fold(f32::INFINITY, f32::min);

    let mut parts: Vec<Vec<usize>> = vec![Vec::new(); widths.len()];
    let mut loads = vec![0f32; widths.len()];
    for dim in by_variance {
        let pick = (0..parts.len())
            .filter(|&p| parts[p].len() < widths[p])
            .min_by(|&a, &b| loads[a].total_cmp(&loads[b]))
            .expect("the part widths sum to the dim, so one is always open");
        parts[pick].push(dim);
        loads[pick] += logs[dim] - smallest;
    }
    parts.concat()
}

impl Primitive for BalanceParts {
    fn describe() -> &'static str {
        "reorder dimensions so every part carries the same variance product"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let widths = SegmentSplit::new(vectors.ncols(), self.section_dim).widths().to_vec();
        let variances = vectors.var_axis(Axis(0), 0.0);
        coding::pack_model(balanced_order(variances.view(), &widths))
    }

    // encode omitted: a permutation owns no per-vector bits.

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        *vectors = vectors.select(Axis(1), &Self::order(model));
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        *queries = queries.select(Axis(1), &Self::order(model));
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let child = child_recons.expect("BalanceParts is not terminal");
        let mut out = Array2::zeros(child.raw_dim());
        for (dealt, &source) in Self::order(model).iter().enumerate() {
            out.column_mut(source).assign(&child.column(dealt));
        }
        out
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Queries are dealt the same way, so the child's scores pass through.
        child_scores.expect("BalanceParts is not terminal").to_owned()
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0) // no per-vector bits: the order lives in the model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores};
    use crate::{math, Kmeans};
    use ndarray::{array, Array1, Array2};

    /// Independent columns with the prescribed variances (already the principal axes).
    fn with_variances(n: usize, variances: &[f32], seed: u64) -> Array2<f32> {
        let mut x = math::gaussian(&mut math::seed(seed), (n, variances.len()));
        for (j, &v) in variances.iter().enumerate() {
            x.column_mut(j).mapv_inplace(|e| e * v.sqrt());
        }
        x
    }

    /// The paper's synthetic spectrum, variance e^(-0.1 d).
    fn decaying(d: usize) -> Vec<f32> {
        (0..d).map(|i| (-0.1 * i as f32).exp()).collect()
    }

    /// Per-part log-variance sums, in dealt order.
    fn loads(variances: &[f32], order: &[usize], widths: &[usize]) -> Vec<f32> {
        let mut start = 0;
        widths
            .iter()
            .map(|&w| {
                let load = order[start..start + w].iter().map(|&d| variances[d].ln()).sum();
                start += w;
                load
            })
            .collect()
    }

    /// The deal balances the log-variance sum across parts; the natural (sorted) order
    /// would spread them over ~19 nats.
    #[test]
    fn balances_the_variance_product_across_parts() {
        let spectrum = decaying(32);
        let order = balanced_order(Array1::from(spectrum.clone()).view(), &[8, 8, 8, 8]);
        let loads = loads(&spectrum, &order, &[8, 8, 8, 8]);
        let spread = loads.iter().copied().fold(f32::MIN, f32::max)
            - loads.iter().copied().fold(f32::MAX, f32::min);
        assert!(spread < 0.01, "unbalanced parts: {loads:?}");
    }

    /// Every dimension is dealt exactly once, including into a short trailing part.
    #[test]
    fn deals_every_dimension_once() {
        let order = balanced_order(Array1::from(decaying(12)).view(), &[5, 5, 2]);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..12).collect::<Vec<_>>());
    }

    /// The largest-variance dimensions land in different parts, one each.
    #[test]
    fn spreads_the_largest_dimensions() {
        let order = balanced_order(array![8.0f32, 4.0, 2.0, 1.0, 0.5, 0.25].view(), &[2, 2, 2]);
        assert_eq!(vec![order[0], order[2], order[4]], vec![0, 1, 2]);
    }

    #[test]
    fn round_trips_and_scores() {
        let v = with_variances(200, &decaying(8), 1);
        let q = math::gaussian(&mut math::seed(2), (3, 8));
        let bp = BalanceParts::new(4);
        let model = bp.fit(v.view(), None);
        let mut x = v.clone();
        bp.apply(&model, &mut x, &[]);
        assert_close(&bp.reconstruct(&model, &[], Some(x.view())), &v, 1e-6);
        let mut rq = q.clone();
        bp.apply_queries(&model, &mut rq);
        assert_close(&rq.dot(&x.t()), &q.dot(&v.t()), 1e-3);
    }

    #[test]
    fn composes_in_pipeline() {
        let v = with_variances(80, &decaying(8), 3);
        let q: Array2<f32> = math::gaussian(&mut math::seed(4), (5, 8));
        assert_pipeline_scores(
            vec![Box::new(BalanceParts::new(4)) as Box<dyn Primitive>, Box::new(Kmeans::new(16, 3))],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
