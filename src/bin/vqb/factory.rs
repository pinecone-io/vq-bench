//! Instantiate a quantizer from a resolved method config, via the catalog registry.

use anyhow::Result;
use vqb::Quantizer;

use crate::config::ResolvedMethod;

/// Build the quantizer named by `m`, reading its parameters. `seed` is the run
/// master seed (passed to seeded primitives like random rotations); `dim` is the
/// dataset vector dimension the pipeline is built for (also needed by estimators
/// whose scale depends on it, e.g. QJL).
pub fn build(m: &ResolvedMethod, seed: u64, dim: usize) -> Result<Box<dyn Quantizer>> {
    vqb::catalog::build(&m.name, &m.params, seed, dim)
}
