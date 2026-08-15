//! `split(segment)`: slice each vector into contiguous coordinate segments.

use ndarray::{s, Array2, ArrayView2};

use crate::Splitter;

/// Split a `d`-dim vector into contiguous segments of `section_dim` columns (the last
/// segment is short when `d % section_dim != 0`). Segment length is dimension-
/// independent. Each segment is quantized by its own branch; `reconstruct` concatenates
/// the columns back, `score` sums the per-branch scores. No model, no per-vector code
/// of its own.
pub struct SegmentSplit {
    /// Column count of each segment; sums to the input dim.
    widths: Vec<usize>,
}

impl SegmentSplit {
    /// Segments of `section_dim` columns over a `dim`-dimensional input.
    pub fn new(dim: usize, section_dim: usize) -> Self {
        // Always checked, not `debug_assert`: benchmarks run in release, where a zero
        // width would spin the loop below forever instead of failing.
        assert!(section_dim > 0 && dim > 0);
        let mut widths = Vec::new();
        let mut rem = dim;
        while rem > 0 {
            let w = rem.min(section_dim);
            widths.push(w);
            rem -= w;
        }
        Self { widths }
    }

    /// Column count of each segment.
    pub(crate) fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// `(start, end)` column bounds of each segment.
    pub(crate) fn bounds(&self) -> Vec<(usize, usize)> {
        let mut start = 0;
        self.widths
            .iter()
            .map(|&w| {
                let bound = (start, start + w);
                start += w;
                bound
            })
            .collect()
    }

    /// Slice a batch's columns into one owned sub-batch per segment.
    fn slice_cols(&self, x: ArrayView2<f32>) -> Vec<Array2<f32>> {
        self.bounds()
            .iter()
            .map(|&(start, end)| x.slice(s![.., start..end]).to_owned())
            .collect()
    }
}

impl Splitter for SegmentSplit {
    fn describe() -> &'static str {
        "slice each vector into equal-width segments, one branch per segment"
    }

    fn n_branches(&self) -> usize {
        self.widths.len()
    }

    // fit / encode use the trait defaults (no model, empty per-vector code).

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }

    fn apply(&self, _model: &[u8], vectors: ArrayView2<f32>, _codes: &[&[u8]]) -> Vec<Array2<f32>> {
        self.slice_cols(vectors)
    }

    fn apply_queries(&self, _model: &[u8], queries: ArrayView2<f32>) -> Vec<Array2<f32>> {
        self.slice_cols(queries)
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        _codes: &[&[u8]],
        child_recons: &[Array2<f32>],
    ) -> Array2<f32> {
        let n_v = child_recons[0].nrows();
        let dim: usize = self.widths.iter().sum();
        let mut out = Array2::zeros((n_v, dim));
        for (recon, (start, end)) in child_recons.iter().zip(self.bounds()) {
            out.slice_mut(s![.., start..end]).assign(recon);
        }
        out
    }

    fn score(
        &self,
        _model: &[u8],
        _codes: &[&[u8]],
        _query: ArrayView2<f32>,
        child_scores: &[Array2<f32>],
    ) -> Array2<f32> {
        // Each branch already scored queries against its own segment; the full dot
        // product is the sum over disjoint segments.
        let mut out = child_scores[0].clone();
        for child_score in &child_scores[1..] {
            out += child_score;
        }
        out
    }

    fn branch_in_dim(&self, _model: &[u8], _in_dim: usize, branch: usize) -> usize {
        self.widths[branch]
    }
}
