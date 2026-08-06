# AGENTS.md

Contributor guide for adding to the VQ-bench catalog. Read
[docs/content/overview.md](docs/content/overview.md) first for the core idea: a
**quantizer** is a named `Pipeline` of **primitive** stages, and the harness only ever
sees the `Quantizer` trait.

## Layout

```
src/primitive.rs        the Primitive trait -- one pipeline stage
src/quantizer.rs        the Quantizer trait + byte_split (size accounting)
src/splitter.rs         the Splitter trait -- a fan-out stage
src/pipeline.rs         the executor: chains stages, owns code framing
src/primitives/
  conditioners/         transforms that own no codes (one file each)
  rounders/             casts to a finite codeword set; pass on the residual
  splitters/            fan-out
  catalog.rs            what `vqb show primitives` prints
src/quantizers/         one file per family, each a `Quantizer` implementor
src/util/coding.rs      the only place models and codes become bytes
src/math/               numerical ops; the faer backend stops here
src/bin/vqb/            the harness -- not touched when adding to the catalog
```

## How the `vqb` harness works

The library is a pure quantization toolkit; everything about datasets, metrics, and
reporting lives in `src/bin/vqb/`. Adding a primitive or quantizer does not touch it.

`main.rs` parses the CLI and dispatches. The interesting path is `run`:

1. **Config** (`config.rs`) parses the JSON and expands each method's list-valued params
   into one *resolved method* per combination — `{"b": [2,4]}` becomes two runs. It also
   validates dataset names, metric names, and param names, which is all `--dry-run` does.
2. **Per dataset** (`run.rs::run_dataset`) loads the HDF5 (`dataset.rs`), then takes
   *seeded* subsamples: the searched DB (`n_base`), the rows used only for `fit`
   (`n_fit`), and the calibration queries (`n_calib`). Each derives from the master seed,
   so `run` and `encode` see identical rows. Ground truth is the exact dot of each eval
   query against its candidate pool; if the DB was subsampled, the pool is recomputed by
   brute force, since the dataset's shipped neighbors index the full base.
3. **Per method** (`factory.rs` builds it from the catalog) the driver calls
   `fit` → `encode` → `score` → `reconstruct`. Encode is fed the base in row chunks
   across the rayon pool: chunks are independent because encoding is neighbor-blind, and
   the indexed collect keeps row order, so output is byte-identical for any thread count.
   `score` runs each query against its own candidates; `reconstruct` runs on a sample of
   `n_reconstruct` rows.
4. **Capture and reduce.** Raw scores and reconstructions stream into a `.raw` binary
   capture (`raw.rs`) as each method finishes, so a whole run's outputs never sit in
   memory at once. `aggregate.rs` reduces them to the requested metrics (computed in
   `bench.rs`) and `results.rs` defines the JSON that comes out.

Artifacts land under `results/`: `<exp>.json` (aggregated), `raw/<exp>.raw` (the capture
`vqb eval` replays), and `codes/` (per-method code stores). A code store (`codes.rs`) is
a fixed-width file whose header records everything determining the codes — dataset,
method label, `seed`, `n_base`, `n_fit`, `n_calib` — so `run` can safely reuse a prior
`encode`'s output, and refuses when the identity doesn't match. `--fresh` ignores stores
entirely.

Three things the **harness** owns, deliberately kept out of quantizers:

- **Size accounting.** `byte_split` measures the model and codes the quantizer actually
  produced, so a quantizer cannot misreport its own footprint.
- **Memory and parallelism policy.** Chunk size, the thread pool, and suppressing nested
  faer parallelism are the driver's business.
- **Cost measurement.** A counting global allocator (`mem.rs`) captures peak heap during
  a single encode; per-query latencies become avg/p50/p90/p99.

The compute path is behind the default `hdf5` feature; `show`, `data list`, and
`run --dry-run` still build without the system library.

## Commands

```bash
cargo build
cargo test                                  # inline #[cfg(test)] modules; there is no tests/ dir
cargo clippy --all-targets -- -D warnings
cargo run -- show primitives                # confirm a new primitive is listed
cargo run -- show quantizers                # confirm a new family is listed
cargo run -- run <config> --dry-run         # validate params and pipelines, compute nothing
```

There is no CI. `cargo test` and `cargo clippy` are the gate and must both be clean
before handing work back.

**Do not run `cargo fmt`.** The tree is hand-formatted and is not rustfmt-clean — a
bare `cargo fmt` rewrites ~26 files and buries the actual change in reformatting noise.
Match the layout of the code around you instead, and keep lines under ~100 characters.

## Adding a primitive

Copy [`conditioners/center.rs`](src/primitives/conditioners/center.rs) for a
conditioner, [`rounders/cast_uint.rs`](src/primitives/rounders/cast_uint.rs) for a
rounder.

1. New file in the right group directory. One primitive per file.
2. Header doc comment in the house format: `NAME: one-line summary`, a `-` rule, then
   `Model:` / `Code for vector x:` / `Apply:` / `Reconstruct:` / `Score:` written as
   transformations (`x --> x - mu`). Add a short paragraph below only when the algebra
   needs it.
3. `impl Primitive`. Required: `apply`, `reconstruct`, `score`. Defaulted — override
   only when the stage needs it: `fit`, `encode`, `apply_queries`, `in_dim`, `out_dim`,
   `code_bytes`. Where a default is deliberately kept, say so in one line
   (`// encode omitted: a resize owns no per-vector bits.`). The trait requires
   `fn describe()` — the lowercase one-line description `vqb show p` prints; the display
   name defaults to the type's name (`fn name()`, override only if they must differ).
4. Register in one place: a `module => Type` entry in the group's `primitives! { ... }`
   list in its `mod.rs` (list position sets the `vqb show p` order). The macro declares
   the module, glob re-exports it, and collects `name()`/`describe()` into the catalog.
   `Split` (an adapter, not a cataloged stage) stays a manual `mod`/`pub use` after it.
5. Tests in an inline `#[cfg(test)] mod tests`. Reuse `crate::util::testing`:
   `assert_pipeline_scores` already checks that `score` matches the exact dot with the
   pipeline's own reconstruction, so most stages need no bespoke harness;
   `assert_close` and `refs` cover the rest. Generate data with `math::seed` +
   `math::gaussian`.

## Adding a quantizer

Copy [`e_rabitq.rs`](src/quantizers/e_rabitq.rs) for a chain,
[`pq.rs`](src/quantizers/pq.rs) for a fan-out.

1. New `src/quantizers/<key>.rs` holding a type that implements `Quantizer`: `name`
   (the config key), `describe`, `params` (omit if none), `display_name` (omit when
   it's just the type name), `build`, and `fit`/`encode`/`reconstruct`/`score`. The
   standard shape is `pub struct <Family>(pub Pipeline);` with the pipeline written
   directly as `pub fn pipeline(<typed params>, seed, dim) -> Result<Pipeline>` in an
   inherent impl (value checks live there, so composed uses are validated too);
   `build` just parses params into it, and `crate::pipeline_quantizer!();` expands the
   four runtime methods delegating to `self.0`. Nothing requires a `Pipeline` — a
   non-pipelined quantizer implements the four directly instead of the macro.
2. Add a `module => Type` entry to the `quantizers! { ... }` list in
   `src/quantizers/mod.rs`. That is the only registration edit — the macro declares the
   module and collects the type into the registry.
3. Params: name them in `params()`, read them with `catalog::get` / `get_or`, and
   validate *values* in `build` with `ensure!`. Unknown param *names* are caught by
   `verify_params` for free. A param type not yet used means one new `FromParam` impl —
   in `src/quantizers/catalog.rs` for a bare scalar, or beside the type when it has its
   own module (`Rotation` in `src/quantizers/rotation.rs`).
4. Take the shared `rotation` param (`full` | `hadamard`, default `hadamard`): read it
   with `get_or(p, "rotation", Rotation::Hadamard)` and put `rotation.stage(seed)` in
   the chain rather than hardcoding a rotation. `rotation::rotate_to` is the shared
   pad-rotate-truncate dance for hitting a coded-dim budget (see `simhash`/`qjl`).
5. Fanning out: wrap a splitter in a single
   `Split::from_factory(splitter, |branch, branch_dim| ...)` stage. The factory builds
   each branch's child pipeline from the **fitted** layout, so a splitter may learn its
   branch widths from the data (`branch_in_dim` reads them off its model).
6. Composing with another family: call its typed `pipeline(...)` and embed the result
   as a stage (a `Pipeline` is a `Primitive`) — `turboquant_prod` embeds a 1-bit QJL
   via `Qjl::pipeline(1.0, rotation, seed ^ ..., mid_dim)`.
7. Tests: a param-rejection test plus a statistical property test, both through
   `<Family>::build` with a `testing::params(&[("b", json!(4))])` map — the same
   path configs take.

## Invariants

These are the rules a new primitive is most likely to break by pattern-matching on a
single neighbour.

- **Size honesty.** `model + codes + config` must suffice to decode. A primitive struct
  holds only configuration; anything learned from data goes into the model bytes, where
  `byte_split` can charge for it. State smuggled into the struct is a size metric that
  lies.
- **`code_bytes` depends on the model, dim, and config — never on the vector batch.**
  The pipeline uses it (with each stage's fitted model in hand) to split each combined
  code into per-stage slices, so an answer that varies row to row corrupts every
  downstream stage. A stage whose layout is learned at fit returns `None` on an empty
  (unfitted) model.
- **Score from codes, not decoded values.** In `score`, dot the query against the packed
  integer levels and correct algebraically — `cast_uint.rs` uses
  `<q, center(c)> = (<q,c> + 0.5*sum(q)) / N`. Decoding back to f32 in the hot path
  inflates the latency the harness is measuring.
- **Bytes only through `crate::coding`.** `pack_model` / `unpack_model::<T>` for the
  model; one `CodeLayout` per code, driving `pack`/`pack_scalars`, `byte_len`, and
  `unpack::<K>` together so size and layout cannot disagree. Primitives never frame or
  length-prefix — `Pipeline` owns framing.
- **Math only through `crate::math`.** `matmul`, `offset_rows`, `orthogonal_procrustes`,
  `lloyd_kmeans`, `random_orthogonal`, ... A primitive never names a backend crate.
- **Handle both a present and an absent child.** `reconstruct` and `score` receive
  `Option`s because a stage's position in the chain is not fixed. A conditioner owns no
  codes and cannot be last, so it `.expect("X is not terminal")`s. A rounder can be
  last *or* not: its `apply` subtracts its own reconstruction and passes the **residual**
  downstream, so another quantizer can refine it — `turboquant_prod` is
  `CastNormal(b-1)` followed by a 1-bit QJL of the residual. So a rounder must fold in
  `Some(child)` when it's there and stand alone when it isn't, which also means
  recovering its width without a child: the `cast_*` family stores the input dim in
  `fit` and calls `super::code_dim`, `Kmeans` reads it off its centroids.
- **Declare each fact once.** A param's type lives at the `get` call site and is
  inferred, never restated in `params()`; the display name lives in `display_name()`, never
  in the builder.
- **Determinism.** All randomness derives from the run seed via `math::seed(seed)`.
  Never `thread_rng`. Where several sub-quantizers each need a stream, derive them
  (`seed.wrapping_add(branch)`).
- **No lint suppressions to make an intermediate step compile.** Don't reach for
  `allow(dead_code)` — move the step boundary so every commit builds clean on its own.

## Style

- One-line doc comments. No rationale essays; match the density in `src/primitives/`.
- Comment the *why*, and only where the algebra isn't self-evident.
- Don't mirror a source paper's notation when it collides with local naming — a MinMax
  stage has a `scale` and an `offset`, not an `a` and a `b`.
- Write the transformation lines of a primitive header with ASCII arrows (`x --> x - mu`),
  matching every other primitive. Prose elsewhere uses `→` freely.
- Row-vector convention throughout: a batch is `(n_vectors x dim)`.
