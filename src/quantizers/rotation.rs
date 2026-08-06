//! The `rotation` param shared by the rotation-based families, and the resize dance
//! that rotates into a coded-dim budget.

use anyhow::{bail, Context, Result};
use serde_json::Value;

use super::catalog::FromParam;
use crate::{Primitive, RandomHadamard, RandomRotate, Resize};

/// Which random rotation conditions the data: a full O(d^2) rotation or the
/// O(d log d) randomized Hadamard transform.
#[derive(Clone, Copy)]
pub enum Rotation {
    Full,
    Hadamard,
}

impl Rotation {
    /// The rotation stage, seeded by `seed`.
    pub fn stage(self, seed: u64) -> Box<dyn Primitive> {
        match self {
            Rotation::Full => Box::new(RandomRotate::new(seed)),
            Rotation::Hadamard => Box::new(RandomHadamard::new(seed)),
        }
    }
}

impl FromParam for Rotation {
    fn from_value(v: &Value) -> Result<Self> {
        match v.as_str().context("must be a string")? {
            "full" => Ok(Rotation::Full),
            "hadamard" => Ok(Rotation::Hadamard),
            other => bail!("unknown rotation `{other}` (expected `full` or `hadamard`)"),
        }
    }
}

/// The stages that rotate `dim` input dims into `m` coded dims: pad up so the added
/// dims rotate in, rotate, truncate down to the budget. At `m == dim` the rotation
/// keeps its own width (padded to a multiple of 64 under Hadamard) -- published
/// numbers are pinned to that width, so the pad is not truncated away.
pub(super) fn rotate_to(
    rotation: Rotation,
    seed: u64,
    dim: usize,
    m: usize,
) -> Vec<Box<dyn Primitive>> {
    let wide = dim.max(m);
    let stage = rotation.stage(seed);
    let rotated = stage.out_dim(wide);
    let mut stages: Vec<Box<dyn Primitive>> = Vec::new();
    if wide != dim {
        stages.push(Box::new(Resize::to(wide)));
    }
    stages.push(stage);
    // `m == dim` keeps the pad (see above); otherwise trim only if the rotation
    // widened past the budget.
    if m != dim && rotated != m {
        stages.push(Box::new(Resize::to(m)));
    }
    stages
}

/// Coded dims for a budget of `bits` sign bits per input dim: `m == dim` at `b == 1`,
/// which leaves the rotation's own width (pad included) untouched.
pub(super) fn coded_dim(bits: f32, dim: usize) -> usize {
    ((bits * dim as f32).round() as usize).max(1)
}
