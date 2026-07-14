//! The [`Pipeline`] executor.

use ndarray::{Array2, ArrayView2};

use crate::coding::{put_len, take, take_len};
use crate::Primitive;

/// A linear chain of stages, itself a [`Primitive`].
pub struct Pipeline {
    stages: Vec<Box<dyn Primitive>>,
}

impl Pipeline {
    /// Build a pipeline from its stages (must be non-empty).
    pub fn new(stages: Vec<Box<dyn Primitive>>) -> Self {
        assert!(!stages.is_empty(), "a pipeline needs at least one stage");
        Self { stages }
    }

    /// Input dim seen by each stage, given the pipeline's input dim.
    fn in_dims(&self, d: usize) -> Vec<usize> {
        let mut dim = d;
        self.stages
            .iter()
            .map(|s| {
                let here = dim;
                dim = s.out_dim(dim);
                here
            })
            .collect()
    }

    /// Split each combined per-vector code into one slice per stage.
    fn split_codes<'a>(&self, in_dims: &[usize], combined: &[&'a [u8]]) -> Vec<Vec<&'a [u8]>> {
        let lens: Vec<Option<usize>> = self
            .stages
            .iter()
            .zip(in_dims)
            .map(|(s, &d)| s.code_bytes(d))
            .collect();
        let mut out: Vec<Vec<&'a [u8]>> = (0..self.stages.len())
            .map(|_| Vec::with_capacity(combined.len()))
            .collect();
        for code in combined {
            let mut cur = *code;
            for (i, len) in lens.iter().enumerate() {
                let n = match len {
                    Some(n) => *n,
                    None => take_len(&mut cur),
                };
                out[i].push(take(&mut cur, n));
            }
        }
        out
    }
}

/// Pack the input dim and per-stage models into one blob.
fn pack_model(d: usize, stage_models: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    put_len(&mut buf, d);
    for m in stage_models {
        put_len(&mut buf, m.len());
        buf.extend_from_slice(m);
    }
    buf
}

/// Unpack the input dim and `n_stages` per-stage model slices.
fn unpack_model(model: &[u8], n_stages: usize) -> (usize, Vec<&[u8]>) {
    let mut cur = model;
    let d = take_len(&mut cur);
    let models = (0..n_stages)
        .map(|_| {
            let len = take_len(&mut cur);
            take(&mut cur, len)
        })
        .collect();
    (d, models)
}

impl Primitive for Pipeline {
    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let d = vectors.ncols();
        let mut v = vectors.to_owned();
        let mut q = queries.map(|q| q.to_owned());
        let last = self.stages.len() - 1;
        let mut models = Vec::with_capacity(self.stages.len());
        for (i, stage) in self.stages.iter().enumerate() {
            let model = stage.fit(v.view(), q.as_ref().map(|q| q.view()));
            if i < last {
                let codes = stage.encode(&model, v.view());
                let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
                stage.apply(&model, &mut v, &refs);
                if let Some(q) = q.as_mut() {
                    stage.apply_queries(&model, q);
                }
            }
            models.push(model);
        }
        pack_model(d, &models)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (d, models) = unpack_model(model, self.stages.len());
        debug_assert_eq!(d, vectors.ncols());
        let mut v = vectors.to_owned();
        let mut combined = vec![Vec::new(); vectors.nrows()];
        let mut dim = d;
        let last = self.stages.len() - 1;
        for (i, stage) in self.stages.iter().enumerate() {
            let codes = stage.encode(models[i], v.view());
            match stage.code_bytes(dim) {
                Some(n) => {
                    for (out, code) in combined.iter_mut().zip(&codes) {
                        debug_assert_eq!(code.len(), n);
                        out.extend_from_slice(code);
                    }
                }
                None => {
                    for (out, code) in combined.iter_mut().zip(&codes) {
                        put_len(out, code.len());
                        out.extend_from_slice(code);
                    }
                }
            }
            if i < last {
                let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
                stage.apply(models[i], &mut v, &refs);
            }
            dim = stage.out_dim(dim);
        }
        combined
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let (d, models) = unpack_model(model, self.stages.len());
        let in_dims = self.in_dims(d);
        let stage_codes = self.split_codes(&in_dims, codes);
        for (i, stage) in self.stages.iter().enumerate() {
            stage.apply(models[i], vectors, &stage_codes[i]);
        }
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        let (_, models) = unpack_model(model, self.stages.len());
        for (i, stage) in self.stages.iter().enumerate() {
            stage.apply_queries(models[i], queries);
        }
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let (d, models) = unpack_model(model, self.stages.len());
        let in_dims = self.in_dims(d);
        let stage_codes = self.split_codes(&in_dims, codes);
        let mut downstream = child_recons.map(|c| c.to_owned());
        for i in (0..self.stages.len()).rev() {
            let child = downstream.as_ref().map(|a| a.view());
            downstream = Some(self.stages[i].reconstruct(models[i], &stage_codes[i], child));
        }
        downstream.expect("non-empty pipeline")
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let (d, models) = unpack_model(model, self.stages.len());
        let in_dims = self.in_dims(d);
        let stage_codes = self.split_codes(&in_dims, codes);

        // Forward: the query batch each stage sees.
        let mut stage_queries = Vec::with_capacity(self.stages.len());
        let mut q = queries.to_owned();
        let last = self.stages.len() - 1;
        for (i, stage) in self.stages.iter().enumerate() {
            stage_queries.push(q.clone());
            if i < last {
                stage.apply_queries(models[i], &mut q);
            }
        }

        // Backward: fold each stage's contribution into the child's scores.
        let mut downstream = child_scores.map(|c| c.to_owned());
        for i in (0..self.stages.len()).rev() {
            let child = downstream.as_ref().map(|a| a.view());
            downstream = Some(self.stages[i].score(
                models[i],
                stage_queries[i].view(),
                &stage_codes[i],
                child,
            ));
        }
        downstream.expect("non-empty pipeline")
    }

    fn out_dim(&self, in_dim: usize) -> usize {
        self.stages.iter().fold(in_dim, |dim, s| s.out_dim(dim))
    }

    fn code_bytes(&self, in_dim: usize) -> Option<usize> {
        let mut total = 0;
        let mut dim = in_dim;
        for stage in &self.stages {
            total += stage.code_bytes(dim)?;
            dim = stage.out_dim(dim);
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{byte_split, AsQuantizer, Quantizer};
    use ndarray::{array, Array2, ArrayView2, Axis};

    // --- stub primitives ------------------------------------------------

    fn read_f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Terminal rounder: one signed byte per dim, lossless on small integers.
    struct IntRound;
    impl Primitive for IntRound {
        fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
            vectors
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|&x| (x.round() as i8) as u8).collect())
                .collect()
        }
        fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
            vectors.mapv_inplace(|x| x - x.round());
        }
        fn reconstruct(
            &self,
            _model: &[u8],
            codes: &[&[u8]],
            child_recons: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            let d = codes.first().map_or(0, |c| c.len());
            let mut out = Array2::zeros((codes.len(), d));
            for (i, code) in codes.iter().enumerate() {
                for (j, &b) in code.iter().enumerate() {
                    out[[i, j]] = (b as i8) as f32;
                }
            }
            if let Some(child) = child_recons {
                out += &child;
            }
            out
        }
        fn score(
            &self,
            _model: &[u8],
            queries: ArrayView2<f32>,
            codes: &[&[u8]],
            child_scores: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            let mut out = Array2::zeros((queries.nrows(), codes.len()));
            for (c, code) in codes.iter().enumerate() {
                for qi in 0..queries.nrows() {
                    let dot: f32 = code
                        .iter()
                        .enumerate()
                        .map(|(j, &b)| (b as i8) as f32 * queries[[qi, j]])
                        .sum();
                    out[[qi, c]] = dot;
                }
            }
            if let Some(child) = child_scores {
                out += &child;
            }
            out
        }
        fn code_bytes(&self, in_dim: usize) -> Option<usize> {
            Some(in_dim)
        }
    }

    /// Conditioner: subtract the per-dim mean; correct scores by `⟨q, μ⟩`.
    struct Center;
    impl Primitive for Center {
        fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
            let mean = vectors.mean_axis(Axis(0)).unwrap();
            mean.iter().flat_map(|m| m.to_le_bytes()).collect()
        }
        // encode uses the default (empty codes).
        fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
            let mean = read_f32s(model);
            for mut row in vectors.rows_mut() {
                row.iter_mut().zip(&mean).for_each(|(x, m)| *x -= m);
            }
        }
        fn reconstruct(
            &self,
            model: &[u8],
            _codes: &[&[u8]],
            child_recons: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            let mean = read_f32s(model);
            let mut out = child_recons.expect("center is not terminal").to_owned();
            for mut row in out.rows_mut() {
                row.iter_mut().zip(&mean).for_each(|(x, m)| *x += m);
            }
            out
        }
        fn score(
            &self,
            model: &[u8],
            queries: ArrayView2<f32>,
            _codes: &[&[u8]],
            child_scores: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            let mean = read_f32s(model);
            let mut out = child_scores.expect("center is not terminal").to_owned();
            for qi in 0..queries.nrows() {
                let qm: f32 = queries
                    .row(qi)
                    .iter()
                    .zip(&mean)
                    .map(|(&q, &m)| q * m)
                    .sum();
                out.row_mut(qi).iter_mut().for_each(|s| *s += qm);
            }
            out
        }
        fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
            Some(0)
        }
    }

    /// Identity conditioner that emits a variable-length code (exercises framing).
    struct VarTag;
    impl Primitive for VarTag {
        fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
            (0..vectors.nrows())
                .map(|i| vec![0xAB; i % 4 + 1])
                .collect()
        }
        fn apply(&self, _model: &[u8], _vectors: &mut Array2<f32>, _codes: &[&[u8]]) {}
        fn reconstruct(
            &self,
            _model: &[u8],
            _codes: &[&[u8]],
            child_recons: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            child_recons.expect("vartag is not terminal").to_owned()
        }
        fn score(
            &self,
            _model: &[u8],
            _queries: ArrayView2<f32>,
            _codes: &[&[u8]],
            child_scores: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            child_scores.expect("vartag is not terminal").to_owned()
        }
        fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
            None
        }
    }

    // --- helpers --------------------------------------------------------

    /// Integer-valued data with integer per-column means ([1, 3, 2]).
    fn data() -> Array2<f32> {
        array![[3., 5., 0.], [1., 3., 4.], [-1., 1., 2.], [1., 3., 2.]]
    }
    fn queries() -> Array2<f32> {
        array![[1., 0., -1.], [2., 1., 0.]]
    }
    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }
    fn assert_close(a: &Array2<f32>, b: &Array2<f32>) {
        assert_eq!(a.dim(), b.dim());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-3, "{x} vs {y}");
        }
    }

    // --- tests ----------------------------------------------------------

    #[test]
    fn single_stage_roundtrip() {
        let (v, q) = (data(), queries());
        let codec = AsQuantizer(Pipeline::new(vec![Box::new(IntRound)]));
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        assert_close(&codec.reconstruct(&model, &refs(&codes)), &v);
        assert_close(
            &codec.score(&model, q.view(), &refs(&codes)),
            &q.dot(&v.t()),
        );
    }

    #[test]
    fn two_stage_roundtrip() {
        let (v, q) = (data(), queries());
        let codec = AsQuantizer(Pipeline::new(vec![Box::new(Center), Box::new(IntRound)]));
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        // Centering by the integer mean keeps values integral, so the round trip is exact.
        assert_close(&codec.reconstruct(&model, &refs(&codes)), &v);
        assert_close(
            &codec.score(&model, q.view(), &refs(&codes)),
            &q.dot(&v.t()),
        );
    }

    #[test]
    fn variable_width_framing() {
        let v = data();
        let codec = AsQuantizer(Pipeline::new(vec![Box::new(VarTag), Box::new(IntRound)]));
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        // Reconstruct is only exact if the variable VarTag segment is framed and
        // skipped correctly, leaving IntRound its raw bytes.
        assert_close(&codec.reconstruct(&model, &refs(&codes)), &v);
    }

    #[test]
    fn size_accounting() {
        let v = data(); // 4 vectors, dim 3
        let one = AsQuantizer(Pipeline::new(vec![Box::new(IntRound)]));
        let m1 = one.fit(v.view(), None);
        let c1 = one.encode(&m1, v.view());
        // model: d(4) + len(4) + empty(0) = 8; codes: 4 vectors * 3 bytes = 12.
        assert_eq!(byte_split(&m1, &c1), (8, 12));

        let two = AsQuantizer(Pipeline::new(vec![Box::new(Center), Box::new(IntRound)]));
        let m2 = two.fit(v.view(), None);
        let c2 = two.encode(&m2, v.view());
        // model: d(4) + [len(4)+mean(12)] + [len(4)+empty(0)] = 24; codes unchanged.
        assert_eq!(byte_split(&m2, &c2), (24, 12));
    }

    #[test]
    fn fit_and_encode_on_different_sets() {
        let a = data();
        let b = &data() + 10.0; // integer, and b - mean(a) stays integral
        let codec = AsQuantizer(Pipeline::new(vec![Box::new(Center), Box::new(IntRound)]));

        let model_a = codec.fit(a.view(), None);
        assert_ne!(model_a, codec.fit(b.view(), None));

        let codes = codec.encode(&model_a, b.view());
        assert_close(&codec.reconstruct(&model_a, &refs(&codes)), &b);
    }
}
