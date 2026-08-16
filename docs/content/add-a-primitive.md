# Adding a Primitive

A primitive is one stage of a quantization pipeline: a conditioner, rounder, or splitter (see [Overview](#overview)). In order to add a primitive to vq-bench, you must do two things:

1. **Write a primitive file `src/primitives/<group>/<name>.rs`**, where `<group>` is `conditioners`, `rounders`, or `splitters`.
2. **Add one line in that group's `mod.rs`**, inside the `primitives! { ... }` list:

```rust
primitives! { Primitive:
    minmax => MinMax,
    // ...
    center => Center,
}
```

In the primitive file, you must specify a public type (e.g., `MinMax` or `Center`), and this type must implement the `Primitive` trait. (Splitters instead implement the `Splitter` trait in `src/splitter.rs`, whose methods return one output per branch.)

## The Primitive trait

The `Primitive` trait (`src/primitive.rs`) specifies everything a stage must do for the pipeline to fit, encode, reconstruct, and score through it.

Required:

| Method | What it does |
|---|---|
| `describe()` | one-line description printed by `vqb show p` |
| `apply(model, vectors, codes)` | transform the batch of vectors into what downstream stages see (a conditioner applies its transformation; a rounder passes down its residual) |
| `reconstruct(model, codes, child_recons)` | rebuild the vectors from the codes and the next stage's reconstruction |
| `score(model, queries, codes, child_scores)` | estimate query–vector dot products from the codes and the next stage's scores |

Defaulted (override when applicable):

| Method | Default | What it does |
|---|---|---|
| `fit(vectors, queries)` | empty model | store learned information into a model |
| `encode(model, vectors)` | empty code per vector | generate per-vector bits |
| `apply_queries(model, queries)` | identity | transform the batch of queries into what downstream stages see |
| `code_bytes(model, in_dim)` | `None` (varies) | specifies the length of the per-vector codes (in bytes) if fixed; `None` on an empty model when the layout is learned at fit, or whenever the codes vary per vector — the pipeline then length-prefixes each one |
| `in_dim()` / `out_dim(in_dim)` | unchanged | specifies the input and output dimensionality of the primitive |

## Example

`src/primitives/conditioners/center.rs`, in full (tests trimmed):

```rust
//! CENTER: subtracts the center from every vector
//! -
//! Model: mu, the mean over the fit vectors
//! Code for vector x: empty
//! Apply: x --> x - mu
//! Reconstruct: y --> y + mu
//! Score: s --> s + <q, mu>

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{coding, math, Primitive};

pub struct Center;

/// The mean mu, read from the model bytes.
fn mean(model: &[u8]) -> Array1<f32> {
    coding::unpack_model(model)
}

impl Primitive for Center {
    fn describe() -> &'static str {
        "subtract the mean over the fit set from every vector"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.mean_axis(Axis(0)).unwrap())
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        let (n_v, d) = vectors.dim();
        *vectors -= &mean(model).broadcast((n_v, d)).unwrap();
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("Center is not terminal").to_owned();
        let (n_v, d) = out.dim();
        out += &mean(model).broadcast((n_v, d)).unwrap();
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_scores.expect("Center is not terminal").to_owned();
        math::offset_rows(&mut out, queries.dot(&mean(model)).view()); // add <q, mu> per query
        out
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}
```

For a rounder — a terminal stage that owns per-vector bits — copy `src/primitives/rounders/cast_uint.rs` instead: it adds `encode`, a fixed `code_bytes`, and scores directly from the packed codes.

Tests live in an inline `#[cfg(test)] mod tests`. `testing::assert_pipeline_scores` already checks that `score` matches the exact dot against the stage's own reconstruction, so most stages need no bespoke harness.

## Verify

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run -- show primitives    # the new row appears
```
