//! Quantizers: named pipelines, the in-tree catalog the harness runs.

pub mod catalog;

use ndarray::{Array2, ArrayView2};

use crate::{Pipeline, Primitive, Quantizer, RandomHadamard, RandomRotate};

/// Register the in-tree quantizer families: declare each builder module and
/// collect its `SPEC` into `QUANTIZERS`. Add a family by writing its module (with
/// a `pub const SPEC`) and adding its name here — the only registration edit.
macro_rules! quantizers {
    ($($name:ident),+ $(,)?) => {
        $(mod $name;)+
        /// Every quantizer family the harness can build.
        pub const QUANTIZERS: &[catalog::QuantizerSpec] = &[$($name::SPEC),+];
    };
}

quantizers! { minmax, scalar, eden_mse, eden_prod, turboquant_mse, rabitq, e_rabitq, qjl, simhash, turboquant_prod, itq, itq_asym, pq, opq }

/// The orthogonal rotation a quantizer applies before rounding: the full dense
/// Haar-random matrix (`O(d^2)`) or the randomized Hadamard transform (`O(d log d)`).
/// The shared `rotation` param, so a config can sweep either without changing the
/// pipeline.
#[derive(Clone, Copy)]
pub(crate) enum Rotation {
    Full,
    Hadamard,
}

impl Rotation {
    /// The rotation stage for input dim `dim`, seeded by `seed`.
    pub(crate) fn stage(self, dim: usize, seed: u64) -> Box<dyn Primitive> {
        match self {
            Rotation::Full => Box::new(RandomRotate::new(seed)),
            Rotation::Hadamard => Box::new(RandomHadamard::new(dim, seed)),
        }
    }
}

/// A named [`Pipeline`], runnable through the [`Quantizer`] interface. `name` is
/// the display **family name** (e.g. `MinMax`); the runner forms a **method name**
/// from it plus parameters (`MinMax (b=2)`). The pipeline is the composition root
/// (a chain, or later a chain holding a `Split` for fan-out).
pub struct NamedQuantizer {
    pub name: String,
    pub pipeline: Pipeline,
}

impl Quantizer for NamedQuantizer {
    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        self.pipeline.fit(vectors, queries)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        self.pipeline.encode(model, vectors)
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        self.pipeline.reconstruct(model, codes, None)
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        candidate_codes: &[&[u8]],
    ) -> Array2<f32> {
        self.pipeline.score(model, queries, candidate_codes, None)
    }
}
