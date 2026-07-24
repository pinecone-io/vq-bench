//! `Split`: adapts a [`Splitter`] + one child [`Pipeline`] per branch into a single
//! terminal [`Primitive`], so fan-out drops into a pipeline chain like any other stage.

use ndarray::{Array2, ArrayView2};

use crate::coding::{put_len, take, take_len};
use crate::{Pipeline, Primitive, Splitter};

/// A fan-out stage: the splitter slices each vector into branches, and `children[i]`
/// quantizes branch `i`. `Split` is **terminal** -- it owns no downstream stage, and
/// every method runs the per-branch work internally and recombines (mirroring how
/// [`Pipeline`] itself implements [`Primitive`]). Non-terminal fan-out is unsupported.
pub struct Split<S: Splitter> {
    splitter: S,
    children: Vec<Pipeline>,
}

impl<S: Splitter> Split<S> {
    /// One child pipeline per branch (`children.len()` must equal `splitter.n_branches()`).
    pub fn new(splitter: S, children: Vec<Pipeline>) -> Self {
        assert_eq!(
            children.len(),
            splitter.n_branches(),
            "one child pipeline per branch"
        );
        Self { splitter, children }
    }

    /// Per-branch input dim, by branch order.
    fn branch_dims(&self, splitter_model: &[u8], in_dim: usize) -> Vec<usize> {
        (0..self.children.len())
            .map(|b| self.splitter.branch_in_dim(splitter_model, in_dim, b))
            .collect()
    }

    /// Fixed code width of each component (splitter, then each child), or `None` if
    /// that component is variable-width.
    fn component_lens(&self, splitter_model: &[u8], in_dim: usize) -> Vec<Option<usize>> {
        let mut lens = vec![self.splitter.code_bytes(in_dim)];
        for (b, &branch_dim) in self.branch_dims(splitter_model, in_dim).iter().enumerate() {
            lens.push(self.children[b].code_bytes(branch_dim));
        }
        lens
    }

    /// Peel each combined per-vector code into the splitter's slice plus one slice
    /// per child branch, framing exactly as [`Pipeline`] does (fixed width or length
    /// prefix). Returns `(splitter_codes, per_branch_codes)`.
    fn split_codes<'a>(
        &self,
        splitter_model: &[u8],
        in_dim: usize,
        combined: &[&'a [u8]],
    ) -> (Vec<&'a [u8]>, Vec<Vec<&'a [u8]>>) {
        let lens = self.component_lens(splitter_model, in_dim);
        let mut splitter_codes = Vec::with_capacity(combined.len());
        let mut branch_codes: Vec<Vec<&[u8]>> = (0..self.children.len())
            .map(|_| Vec::with_capacity(combined.len()))
            .collect();
        for code in combined {
            let mut cur = *code;
            for (i, len) in lens.iter().enumerate() {
                let n_bytes = match len {
                    Some(width) => *width,
                    None => take_len(&mut cur),
                };
                let slice = take(&mut cur, n_bytes);
                if i == 0 {
                    splitter_codes.push(slice);
                } else {
                    branch_codes[i - 1].push(slice);
                }
            }
        }
        (splitter_codes, branch_codes)
    }
}

/// Pack the input dim, the splitter model, and each child model into one blob.
fn pack_model(dim: usize, splitter_model: &[u8], child_models: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    put_len(&mut buf, dim);
    put_len(&mut buf, splitter_model.len());
    buf.extend_from_slice(splitter_model);
    for m in child_models {
        put_len(&mut buf, m.len());
        buf.extend_from_slice(m);
    }
    buf
}

/// Unpack `(dim, splitter_model, child_models)` for `n_children` branches.
fn unpack_model(model: &[u8], n_children: usize) -> (usize, &[u8], Vec<&[u8]>) {
    let mut cur = model;
    let dim = take_len(&mut cur);
    let sm_len = take_len(&mut cur);
    let splitter_model = take(&mut cur, sm_len);
    let child_models = (0..n_children)
        .map(|_| {
            let len = take_len(&mut cur);
            take(&mut cur, len)
        })
        .collect();
    (dim, splitter_model, child_models)
}

impl<S: Splitter> Primitive for Split<S> {
    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let dim = vectors.ncols();
        let splitter_model = self.splitter.fit(vectors, queries);
        let splitter_codes = self.splitter.encode(&splitter_model, vectors);
        let splitter_refs: Vec<&[u8]> = splitter_codes.iter().map(Vec::as_slice).collect();
        let sub_vectors = self.splitter.apply(&splitter_model, vectors, &splitter_refs);
        let sub_queries = queries.map(|q| self.splitter.apply_queries(&splitter_model, q));
        let child_models = self
            .children
            .iter()
            .enumerate()
            .map(|(b, child)| {
                child.fit(
                    sub_vectors[b].view(),
                    sub_queries.as_ref().map(|sub_queries| sub_queries[b].view()),
                )
            })
            .collect::<Vec<_>>();
        pack_model(dim, &splitter_model, &child_models)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (_, splitter_model, child_models) = unpack_model(model, self.children.len());
        let splitter_codes = self.splitter.encode(splitter_model, vectors);
        let splitter_refs: Vec<&[u8]> = splitter_codes.iter().map(Vec::as_slice).collect();
        let sub_vectors = self.splitter.apply(splitter_model, vectors, &splitter_refs);

        // Frame each component per vector -- splitter code, then each branch's code, raw
        // when fixed-width and length-prefixed otherwise (matching Pipeline). Branches are
        // appended as they are encoded, so only one branch's codes is held at a time.
        let lens = self.component_lens(splitter_model, vectors.ncols());
        let mut combined = vec![Vec::new(); vectors.nrows()];
        for (out, splitter_code) in combined.iter_mut().zip(&splitter_codes) {
            append_component(out, lens[0], splitter_code);
        }
        for (b, child) in self.children.iter().enumerate() {
            let child_codes = child.encode(child_models[b], sub_vectors[b].view());
            for (out, code) in combined.iter_mut().zip(&child_codes) {
                append_component(out, lens[b + 1], code);
            }
        }
        combined
    }

    fn apply(&self, _model: &[u8], _vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        unimplemented!("Split is a terminal stage; non-terminal fan-out is unsupported")
    }

    fn apply_queries(&self, _model: &[u8], _queries: &mut Array2<f32>) {
        unimplemented!("Split is a terminal stage; non-terminal fan-out is unsupported")
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        debug_assert!(child_recons.is_none(), "Split is terminal");
        let (dim, splitter_model, child_models) = unpack_model(model, self.children.len());
        let (splitter_codes, branch_codes) = self.split_codes(splitter_model, dim, codes);
        let recons: Vec<Array2<f32>> = self
            .children
            .iter()
            .enumerate()
            .map(|(b, child)| child.reconstruct(child_models[b], &branch_codes[b], None))
            .collect();
        self.splitter.reconstruct(splitter_model, &splitter_codes, &recons)
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        debug_assert!(child_scores.is_none(), "Split is terminal");
        let (dim, splitter_model, child_models) = unpack_model(model, self.children.len());
        let (splitter_codes, branch_codes) = self.split_codes(splitter_model, dim, codes);
        let sub_queries = self.splitter.apply_queries(splitter_model, queries);
        let scores: Vec<Array2<f32>> = self
            .children
            .iter()
            .enumerate()
            .map(|(b, child)| {
                child.score(child_models[b], sub_queries[b].view(), &branch_codes[b], None)
            })
            .collect();
        self.splitter.score(splitter_model, &splitter_codes, queries, &scores)
    }

    fn out_dim(&self, in_dim: usize) -> usize {
        // Segment fan-out recombines to the input dim; Split is terminal so nothing
        // downstream reads this.
        in_dim
    }

    fn code_bytes(&self, in_dim: usize) -> Option<usize> {
        // Sums the splitter's own code with each child's. Relies on branch_in_dim being
        // model-independent (true for segment); a model-dependent splitter returns None.
        self.component_lens(&[], in_dim).into_iter().sum()
    }
}

/// Append one component's code to a vector's combined buffer, length-prefixing iff
/// the component is variable-width.
fn append_component(out: &mut Vec<u8>, len: Option<usize>, code: &[u8]) {
    if len.is_none() {
        put_len(out, code.len());
    } else {
        debug_assert_eq!(len, Some(code.len()));
    }
    out.extend_from_slice(code);
}
