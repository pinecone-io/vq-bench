//! `Split`: adapts a [`Splitter`] + one child [`Pipeline`] per branch into a single
//! terminal [`Primitive`], so fan-out drops into a pipeline chain like any other stage.

use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2};

use crate::coding::{put_len, take, take_len};
use crate::{Pipeline, Primitive, Splitter};

/// A fan-out stage: the splitter slices each vector into branches, and one child
/// pipeline quantizes each branch. Children come from `factory(branch, branch_dim)`
/// against the **fitted** layout, so a splitter may learn its branch widths from the
/// data. `Split` is **terminal** -- it owns no downstream stage, and every method runs
/// the per-branch work internally and recombines (mirroring how [`Pipeline`] itself
/// implements [`Primitive`]). Non-terminal fan-out is unsupported.
pub struct Split<S: Splitter> {
    splitter: S,
    factory: Box<dyn Fn(usize, usize) -> Pipeline + Send + Sync>,
    /// The child pipelines, with the branch layout they were built for.
    children: OnceLock<(Vec<usize>, Vec<Pipeline>)>,
}

impl<S: Splitter> Split<S> {
    /// A `Split` whose children are built from the fitted layout:
    /// `factory(branch, branch_dim)` runs once the splitter's model -- hence each
    /// branch's dim -- exists.
    ///
    /// The factory must be deterministic in `(branch, branch_dim)`. A fresh process
    /// rebuilds the children from persisted model bytes, so determinism here is what
    /// keeps `model + codes + config` sufficient to decode.
    pub fn from_factory<F>(splitter: S, factory: F) -> Self
    where
        F: Fn(usize, usize) -> Pipeline + Send + Sync + 'static,
    {
        Self { splitter, factory: Box::new(factory), children: OnceLock::new() }
    }

    /// Per-branch input dim, by branch order.
    fn branch_dims(&self, splitter_model: &[u8], in_dim: usize) -> Vec<usize> {
        (0..self.splitter.n_branches())
            .map(|b| self.splitter.branch_in_dim(splitter_model, in_dim, b))
            .collect()
    }

    /// The child pipelines for branch layout `dims`, built on first use.
    ///
    /// A stage sees one (model, dim) for its lifetime, so one build suffices -- but the
    /// cache records the layout it was built for and holds every later call to it
    /// rather than trusting that. Always checked, not `debug_assert`: benchmarks run in
    /// release, and children sized for a stale layout would report wrong numbers
    /// instead of failing.
    fn children(&self, dims: &[usize]) -> &[Pipeline] {
        let (built_for, children) = self.children.get_or_init(|| {
            let built = dims.iter().enumerate().map(|(b, &d)| (self.factory)(b, d)).collect();
            (dims.to_vec(), built)
        });
        assert_eq!(built_for.as_slice(), dims, "Split children built for another layout");
        children
    }

    /// The child pipelines for the fitted layout, plus each component's fixed code
    /// width (the splitter's, then one per branch); `None` marks a variable-width
    /// component. The single place the branch layout is derived.
    fn children_and_lens(
        &self,
        splitter_model: &[u8],
        child_models: &[&[u8]],
        in_dim: usize,
    ) -> (&[Pipeline], Vec<Option<usize>>) {
        let dims = self.branch_dims(splitter_model, in_dim);
        let children = self.children(&dims);
        let mut lens = vec![self.splitter.code_bytes(splitter_model, in_dim)];
        for (b, &branch_dim) in dims.iter().enumerate() {
            lens.push(children[b].code_bytes(child_models[b], branch_dim));
        }
        (children, lens)
    }

    /// Peel each combined per-vector code into the splitter's slice plus one slice
    /// per child branch, framing exactly as [`Pipeline`] does (fixed width or length
    /// prefix). Returns `(splitter_codes, per_branch_codes)`.
    fn split_codes<'a>(
        lens: &[Option<usize>],
        combined: &[&'a [u8]],
    ) -> (Vec<&'a [u8]>, Vec<Vec<&'a [u8]>>) {
        let mut splitter_codes = Vec::with_capacity(combined.len());
        let mut branch_codes: Vec<Vec<&[u8]>> = (0..lens.len() - 1)
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
    fn describe() -> &'static str {
        "a splitter with one child pipeline per branch"
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let dim = vectors.ncols();
        let splitter_model = self.splitter.fit(vectors, queries);
        let splitter_codes = self.splitter.encode(&splitter_model, vectors);
        let splitter_refs: Vec<&[u8]> = splitter_codes.iter().map(Vec::as_slice).collect();
        let sub_vectors = self.splitter.apply(&splitter_model, vectors, &splitter_refs);
        let sub_queries = queries.map(|q| self.splitter.apply_queries(&splitter_model, q));
        let dims = self.branch_dims(&splitter_model, dim);
        let child_models = self
            .children(&dims)
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
        let (dim, splitter_model, child_models) = unpack_model(model, self.splitter.n_branches());
        assert_eq!(dim, vectors.ncols(), "Split fitted for another dim");
        let splitter_codes = self.splitter.encode(splitter_model, vectors);
        let splitter_refs: Vec<&[u8]> = splitter_codes.iter().map(Vec::as_slice).collect();
        let sub_vectors = self.splitter.apply(splitter_model, vectors, &splitter_refs);

        // Frame each component per vector -- splitter code, then each branch's code, raw
        // when fixed-width and length-prefixed otherwise (matching Pipeline). Branches are
        // appended as they are encoded, so only one branch's codes is held at a time.
        let (children, lens) = self.children_and_lens(splitter_model, &child_models, dim);
        let mut combined = vec![Vec::new(); vectors.nrows()];
        for (out, splitter_code) in combined.iter_mut().zip(&splitter_codes) {
            append_component(out, lens[0], splitter_code);
        }
        for (b, child) in children.iter().enumerate() {
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
        let (dim, splitter_model, child_models) = unpack_model(model, self.splitter.n_branches());
        let (children, lens) = self.children_and_lens(splitter_model, &child_models, dim);
        let (splitter_codes, branch_codes) = Self::split_codes(&lens, codes);
        let recons: Vec<Array2<f32>> = children
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
        let (dim, splitter_model, child_models) = unpack_model(model, self.splitter.n_branches());
        let (children, lens) = self.children_and_lens(splitter_model, &child_models, dim);
        let (splitter_codes, branch_codes) = Self::split_codes(&lens, codes);
        let sub_queries = self.splitter.apply_queries(splitter_model, queries);
        let scores: Vec<Array2<f32>> = children
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

    fn code_bytes(&self, model: &[u8], in_dim: usize) -> Option<usize> {
        // Sums the splitter's own code with each child's. The branch layout can live in
        // the fitted splitter model, so pre-fit there is no answer -- and answering it
        // would seed the child cache from a layout that does not exist yet.
        if model.is_empty() {
            return None;
        }
        let (dim, splitter_model, child_models) = unpack_model(model, self.splitter.n_branches());
        assert_eq!(dim, in_dim, "Split fitted for another dim");
        self.children_and_lens(splitter_model, &child_models, in_dim).1.into_iter().sum()
    }
}

/// Append one component's code to a vector's combined buffer, length-prefixing iff
/// the component is variable-width.
fn append_component(out: &mut Vec<u8>, len: Option<usize>, code: &[u8]) {
    match len {
        None => put_len(out, code.len()),
        // A component that emits a width other than the one it declares mis-slices
        // every later component in `split_codes`, and the run's numbers come out wrong
        // rather than absent -- so this stays on in release.
        Some(n) => assert_eq!(
            code.len(),
            n,
            "a Split component declared a {n}-byte code and emitted {}",
            code.len()
        ),
    }
    out.extend_from_slice(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{coding, math, Kmeans};
    use ndarray::{s, Axis};

    /// Two branches whose boundary is learned at fit: the split lands just after the
    /// highest-variance column, so the branch widths live in the model.
    struct VarSplit;

    impl VarSplit {
        fn boundary(model: &[u8]) -> usize {
            coding::unpack_model(model)
        }
    }

    impl Splitter for VarSplit {
        fn describe() -> &'static str {
            "split after the highest-variance column (fitted layout)"
        }

        fn n_branches(&self) -> usize {
            2
        }

        fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
            let var = vectors.var_axis(Axis(0), 0.0);
            let argmax = (0..var.len()).max_by(|&a, &b| var[a].total_cmp(&var[b])).unwrap();
            coding::pack_model((argmax + 1).min(vectors.ncols() - 1))
        }

        fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
            Some(0)
        }

        fn apply(
            &self,
            model: &[u8],
            vectors: ArrayView2<f32>,
            _codes: &[&[u8]],
        ) -> Vec<Array2<f32>> {
            let k = Self::boundary(model);
            vec![vectors.slice(s![.., ..k]).to_owned(), vectors.slice(s![.., k..]).to_owned()]
        }

        fn apply_queries(&self, model: &[u8], queries: ArrayView2<f32>) -> Vec<Array2<f32>> {
            let k = Self::boundary(model);
            vec![queries.slice(s![.., ..k]).to_owned(), queries.slice(s![.., k..]).to_owned()]
        }

        fn reconstruct(
            &self,
            model: &[u8],
            _codes: &[&[u8]],
            child_recons: &[Array2<f32>],
        ) -> Array2<f32> {
            let k = Self::boundary(model);
            let mut out =
                Array2::zeros((child_recons[0].nrows(), k + child_recons[1].ncols()));
            out.slice_mut(s![.., ..k]).assign(&child_recons[0]);
            out.slice_mut(s![.., k..]).assign(&child_recons[1]);
            out
        }

        fn score(
            &self,
            _model: &[u8],
            _codes: &[&[u8]],
            _query: ArrayView2<f32>,
            child_scores: &[Array2<f32>],
        ) -> Array2<f32> {
            &child_scores[0] + &child_scores[1]
        }

        fn branch_in_dim(&self, model: &[u8], in_dim: usize, branch: usize) -> usize {
            let k = Self::boundary(model);
            if branch == 0 { k } else { in_dim - k }
        }
    }

    /// A `Split` over `VarSplit` with one k-means branch each.
    fn var_split() -> Split<VarSplit> {
        Split::from_factory(VarSplit, |b, branch_dim| {
            Pipeline::new(
                branch_dim,
                vec![Box::new(Kmeans::new(8, 42 + b as u64)) as Box<dyn Primitive>],
            )
            .unwrap()
        })
    }

    /// Column 0 has by far the highest variance, so the fitted boundary is 1: branch
    /// widths 1 and 5 -- unequal, and knowable only after fit.
    fn skewed() -> Array2<f32> {
        let mut v = math::gaussian(&mut math::seed(1), (50, 6));
        v.column_mut(0).mapv_inplace(|x| 10.0 * x);
        v
    }

    /// A component that emits a width other than the one it declares would mis-slice
    /// every later component, so `append_component` refuses it — in release too, which
    /// is where the benchmarks run.
    #[test]
    #[should_panic(expected = "declared a 2-byte code and emitted 3")]
    fn a_component_that_lies_about_its_width_is_caught() {
        append_component(&mut Vec::new(), Some(2), &[1, 2, 3]);
    }

    /// A component that declares itself variable is framed instead, whatever it emits.
    #[test]
    fn a_variable_component_is_length_prefixed() {
        let mut out = Vec::new();
        append_component(&mut out, None, &[1, 2, 3]);
        assert_eq!(out, [3, 0, 0, 0, 1, 2, 3]);
    }

    #[test]
    fn fitted_branch_widths_round_trip() {
        let v = skewed();
        let q = math::gaussian(&mut math::seed(2), (4, 6));
        let split = var_split();

        // The layout is unknown pre-fit and exact once the model exists.
        assert_eq!(split.code_bytes(&[], 6), None);
        let model = split.fit(v.view(), None);
        let codes = split.encode(&model, v.view());
        let r = refs(&codes);
        assert_eq!(split.code_bytes(&model, 6), Some(codes[0].len()));

        let recon = split.reconstruct(&model, &r, None);
        assert_eq!(recon.dim(), (50, 6));
        // Kmeans scores exactly against its own centroids, so the combined score is the
        // exact dot with the combined reconstruction.
        assert_close(&split.score(&model, q.view(), &r, None), &q.dot(&recon.t()), 1e-3);
    }

    #[test]
    #[should_panic(expected = "Split children built for another layout")]
    fn a_second_layout_is_rejected_not_silently_reused() {
        // Refitting one stage under a different learned layout would otherwise reuse
        // children sized for the first. Widths go from (1, 5) to (5, 1).
        let split = var_split();
        let mut other = math::gaussian(&mut math::seed(3), (50, 6));
        other.column_mut(4).mapv_inplace(|x| 10.0 * x);
        split.fit(skewed().view(), None);
        split.fit(other.view(), None);
    }

    #[test]
    #[should_panic(expected = "Split fitted for another dim")]
    fn rejects_a_width_the_model_was_not_fitted_for() {
        // The model carries the dim its branch widths were derived from, so a batch of
        // another width would split codes on a layout that never described it.
        let split = var_split();
        let model = split.fit(skewed().view(), None);
        split.encode(&model, math::gaussian(&mut math::seed(4), (50, 8)).view());
    }

    #[test]
    fn a_fresh_split_decodes_a_persisted_model() {
        // The size-honesty invariant for a learned layout: model + codes + config must
        // suffice. A second Split, built from config alone, rebuilds its children from
        // the persisted model bytes and must agree exactly -- the unit-level mirror of
        // the harness's run-from-stored-codes path.
        let v = skewed();
        let q = math::gaussian(&mut math::seed(2), (4, 6));

        let (model, codes) = {
            let split = var_split();
            let model = split.fit(v.view(), None);
            let codes = split.encode(&model, v.view());
            (model, codes)
        };

        let fresh = var_split();
        let r = refs(&codes);
        assert_eq!(fresh.code_bytes(&model, 6), Some(codes[0].len()));
        assert_eq!(fresh.encode(&model, v.view()), codes);
        let recon = fresh.reconstruct(&model, &r, None);
        assert_close(&fresh.score(&model, q.view(), &r, None), &q.dot(&recon.t()), 1e-3);
    }
}
