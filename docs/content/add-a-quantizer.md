# Adding a Quantizer

A quantizer is an object that can support the quantization operations required by the harness. After building a quantizer, a run configuration can select in its `methods` list to evaluate it. Quantizers are typically constructed by specifying a pipeline of primitives, but vq-bench permits non-pipelined quantizers as well. In order to add a quantizer to vq-bench, you must do two things:

1. **Write a quantizer file `src/quantizers/<key>.rs`**, where `<key>` is the config name (e.g., `pq`).
2. **Add one line in `src/quantizers/mod.rs`**, inside the `quantizers! { ... }` list:

```rust
quantizers! {
    minmax => MinMax,
    // ...
    pq => Pq,
}
```

In the quantizer file, you must specify a public type (e.g., `MinMax` or `Pq`), and this type must implement the `Quantizer` trait.

## The Quantizer trait

The `Quantizer` trait (`src/quantizer.rs`) specifies the quantizer's metadata, how to build it from config params, and the four quantizer operations performed by the harness.

Required:

| Method | What it does |
|---|---|
| `name()` | the config key (e.g., `"minmax"`) |
| `describe()` | one-line description printed by `vqb show q` |
| `build(params, seed, dim)` | parse the params and construct the quantizer |
| `fit(vectors, queries)` | learn a model from the fit vectors and an optional query sample |
| `encode(model, vectors)` | encode vectors into per-vector codes |
| `reconstruct(model, codes)` | reconstruct one vector per code |
| `score(model, queries, codes)` | estimate the score of each query against each candidate |

For a pipelined quantizer, `crate::pipeline_quantizer!();` implements `fit`, `encode`, `reconstruct`, and `score` by calling the pipeline (assuming that `self.0` is the quantizer's pipeline object).

Defaulted (override when applicable):

| Method | Default | What it does |
|---|---|---|
| `display_name()` | the type's name | the display name used in results (`Pq` overrides it to `"PQ"`) |
| `params()` | none | the accepted param names |
| `verify_params(params)` | flags unknown param names | checks that user parameters are valid |

## The standard shape

Most families are a new type over a `Pipeline` of primitives:

- `pub struct Family(pub Pipeline);`
- `pub fn pipeline(<typed params>, seed, dim) -> Result<Pipeline>` holds the `ensure!` value checks, so another family composing it is validated the same way. A `Pipeline` is itself a `Primitive`, so you can embed another family's `Other::pipeline(...)?` as a single stage.
- `build` reads each param with `get(p, "b")?` (the param's type is inferred from `pipeline`'s signature) and wraps the result.
- `crate::pipeline_quantizer!();` expands the four runtime methods, delegating to `self.0`.

Nothing requires a `Pipeline` — a non-pipelined quantizer skips the macro and implements the four runtime methods directly.

## Example

`src/quantizers/minmax.rs`, in full (tests trimmed):

```rust
//! `minmax`: per-vector rescale to `[0, 1]`, then a `b`-bit uniform lattice.

use anyhow::{ensure, Result};

use super::catalog::get;
use crate::coding::CodeLayout;
use crate::MinMax as MinMaxStage;
use crate::{CastUint, Params, Pipeline, Quantizer};

/// The `minmax` family. `get` type-checks `b`; value/range checks live in `build`.
pub struct MinMax(pub Pipeline);

impl MinMax {
    /// Rescale each vector to `[0, 1]`, then a `bits`-bit uniform lattice.
    pub fn pipeline(bits: u8, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Pipeline::new(
            dim,
            vec![Box::new(MinMaxStage::default()), Box::new(CastUint::new(bits))],
        )
    }
}

impl Quantizer for MinMax {
    fn name() -> &'static str {
        "minmax"
    }

    fn params() -> &'static [&'static str] {
        &["b"]
    }

    fn describe() -> &'static str {
        "MinMax -> CastUint(b)"
    }

    fn build(p: &Params, _seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(get(p, "b")?, dim)?))
    }

    crate::pipeline_quantizer!();
}
```

For a fan-out family (one sub-pipeline per vector segment), copy `src/quantizers/pq.rs` instead: it builds a splitter, one child pipeline per branch, and wraps them in a single `Split` stage.

The new family is immediately selectable from a run configuration — `"name"` matches `name()`, sibling keys must appear in `params()`, and an array sweeps that param:

```json
"methods": [
    { "name": "minmax", "b": [2, 4, 6] }
]
```

## Verify

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run -- show quantizers                 # the new family appears
cargo run -- run <config> --dry-run          # the config builds it
```
