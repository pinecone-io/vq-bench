//! The [`Pipeline`] executor.

use anyhow::{bail, Result};
use ndarray::{Array2, ArrayView2};

use crate::coding::{put_len, take, take_len};
use crate::Primitive;

/// A linear chain of stages, itself a [`Primitive`].
pub struct Pipeline {
    stages: Vec<Box<dyn Primitive>>,
    /// Input dim each stage sees; `in_dims[0]` is the pipeline's input dim.
    in_dims: Vec<usize>,
}

impl Pipeline {
    /// Build a pipeline over input dim `d` from its stages (must be non-empty).
    /// Errors if a stage declares an input dim that mismatches the chain.
    pub fn new(d: usize, stages: Vec<Box<dyn Primitive>>) -> Result<Self> {
        assert!(!stages.is_empty(), "a pipeline needs at least one stage");
        let mut dim = d;
        let mut in_dims = Vec::with_capacity(stages.len());
        for (i, s) in stages.iter().enumerate() {
            if let Some(expected) = s.in_dim() {
                if expected != dim {
                    bail!("stage {i} expects input dim {expected}, gets {dim}");
                }
            }
            in_dims.push(dim);
            dim = s.out_dim(dim);
        }
        Ok(Self { stages, in_dims })
    }

    /// Split each combined per-vector code into one slice per stage.
    fn split_codes<'a>(&self, models: &[&[u8]], combined: &[&'a [u8]]) -> Vec<Vec<&'a [u8]>> {
        let lens: Vec<Option<usize>> = self
            .stages
            .iter()
            .zip(models)
            .zip(&self.in_dims)
            .map(|((s, &m), &d)| s.code_bytes(m, d))
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

/// Pack the per-stage models into one blob.
fn pack_model(stage_models: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for m in stage_models {
        put_len(&mut buf, m.len());
        buf.extend_from_slice(m);
    }
    buf
}

/// Unpack `n_stages` per-stage model slices.
fn unpack_model(model: &[u8], n_stages: usize) -> Vec<&[u8]> {
    let mut cur = model;
    (0..n_stages)
        .map(|_| {
            let len = take_len(&mut cur);
            take(&mut cur, len)
        })
        .collect()
}

impl Primitive for Pipeline {
    fn describe() -> &'static str {
        "a chain of primitive stages"
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        assert_eq!(
            vectors.ncols(),
            self.in_dims[0],
            "pipeline built for another dim"
        );
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
        pack_model(&models)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let models = unpack_model(model, self.stages.len());
        assert_eq!(
            vectors.ncols(),
            self.in_dims[0],
            "pipeline built for another dim"
        );
        let mut v = vectors.to_owned();
        let mut combined = vec![Vec::new(); vectors.nrows()];
        let last = self.stages.len() - 1;
        for (i, stage) in self.stages.iter().enumerate() {
            let codes = stage.encode(models[i], v.view());
            match stage.code_bytes(models[i], self.in_dims[i]) {
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
        }
        combined
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let models = unpack_model(model, self.stages.len());
        let stage_codes = self.split_codes(&models, codes);
        for (i, stage) in self.stages.iter().enumerate() {
            stage.apply(models[i], vectors, &stage_codes[i]);
        }
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        let models = unpack_model(model, self.stages.len());
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
        let models = unpack_model(model, self.stages.len());
        let stage_codes = self.split_codes(&models, codes);
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
        let models = unpack_model(model, self.stages.len());
        let stage_codes = self.split_codes(&models, codes);

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

    fn in_dim(&self) -> Option<usize> {
        Some(self.in_dims[0])
    }

    fn out_dim(&self, in_dim: usize) -> usize {
        debug_assert_eq!(in_dim, self.in_dims[0]);
        let last = self.stages.len() - 1;
        self.stages[last].out_dim(self.in_dims[last])
    }

    fn code_bytes(&self, model: &[u8], in_dim: usize) -> Option<usize> {
        debug_assert_eq!(in_dim, self.in_dims[0]);
        // A stage's layout may live in its model, so an unfitted chain has no answer.
        if model.is_empty() {
            return None;
        }
        let models = unpack_model(model, self.stages.len());
        self.stages
            .iter()
            .zip(&models)
            .zip(&self.in_dims)
            .map(|((s, &m), &d)| s.code_bytes(m, d))
            .sum()
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
        fn describe() -> &'static str {
            "one signed byte per dim (test stage)"
        }
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
        fn code_bytes(&self, _model: &[u8], in_dim: usize) -> Option<usize> {
            Some(in_dim)
        }
    }

    /// Conditioner: subtract the per-dim mean; correct scores by `⟨q, μ⟩`.
    struct Center;
    impl Primitive for Center {
        fn describe() -> &'static str {
            "subtract the per-dim mean (test stage)"
        }
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
        fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
            Some(0)
        }
    }

    /// Identity conditioner that emits a variable-length code (exercises framing).
    struct VarTag;
    impl Primitive for VarTag {
        fn describe() -> &'static str {
            "identity with a variable-length code (test stage)"
        }
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
        fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
            None
        }
    }

    /// A stage pinned to a fixed input dim (exercises the chain dim check).
    struct FixedDim(usize);
    impl Primitive for FixedDim {
        fn describe() -> &'static str {
            "identity pinned to a fixed input dim (test stage)"
        }
        fn apply(&self, _model: &[u8], _vectors: &mut Array2<f32>, _codes: &[&[u8]]) {}
        fn reconstruct(
            &self,
            _model: &[u8],
            _codes: &[&[u8]],
            child_recons: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            child_recons.expect("fixeddim is not terminal").to_owned()
        }
        fn score(
            &self,
            _model: &[u8],
            _queries: ArrayView2<f32>,
            _codes: &[&[u8]],
            child_scores: Option<ArrayView2<f32>>,
        ) -> Array2<f32> {
            child_scores.expect("fixeddim is not terminal").to_owned()
        }
        fn in_dim(&self) -> Option<usize> {
            Some(self.0)
        }
        fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
            Some(0)
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
        let codec = AsQuantizer(Pipeline::new(3, vec![Box::new(IntRound)]).unwrap());
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
        let codec =
            AsQuantizer(Pipeline::new(3, vec![Box::new(Center), Box::new(IntRound)]).unwrap());
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
        let codec =
            AsQuantizer(Pipeline::new(3, vec![Box::new(VarTag), Box::new(IntRound)]).unwrap());
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        // Reconstruct is only exact if the variable VarTag segment is framed and
        // skipped correctly, leaving IntRound its raw bytes.
        assert_close(&codec.reconstruct(&model, &refs(&codes)), &v);
    }

    #[test]
    fn size_accounting() {
        let v = data(); // 4 vectors, dim 3
        let one = AsQuantizer(Pipeline::new(3, vec![Box::new(IntRound)]).unwrap());
        let m1 = one.fit(v.view(), None);
        let c1 = one.encode(&m1, v.view());
        // model: len(4) + empty(0) = 4; codes: 4 vectors * 3 bytes = 12.
        assert_eq!(byte_split(&m1, &c1), (4, 12));

        let two =
            AsQuantizer(Pipeline::new(3, vec![Box::new(Center), Box::new(IntRound)]).unwrap());
        let m2 = two.fit(v.view(), None);
        let c2 = two.encode(&m2, v.view());
        // model: [len(4)+mean(12)] + [len(4)+empty(0)] = 20; codes unchanged.
        assert_eq!(byte_split(&m2, &c2), (20, 12));
    }

    #[test]
    fn dim_mismatch_is_error() {
        // A stage pinned to dim 4 in a pipeline built for dim 3 is rejected...
        assert!(Pipeline::new(3, vec![Box::new(FixedDim(4)), Box::new(IntRound)]).is_err());
        // ...while the matching dim builds fine.
        assert!(Pipeline::new(4, vec![Box::new(FixedDim(4)), Box::new(IntRound)]).is_ok());
    }

    #[test]
    fn fit_and_encode_on_different_sets() {
        let a = data();
        let b = &data() + 10.0; // integer, and b - mean(a) stays integral
        let codec =
            AsQuantizer(Pipeline::new(3, vec![Box::new(Center), Box::new(IntRound)]).unwrap());

        let model_a = codec.fit(a.view(), None);
        assert_ne!(model_a, codec.fit(b.view(), None));

        let codes = codec.encode(&model_a, b.view());
        assert_close(&codec.reconstruct(&model_a, &refs(&codes)), &b);
    }
}
