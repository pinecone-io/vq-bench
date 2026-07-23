//! Quantizers: named pipelines, the in-tree catalog the harness runs.

pub mod catalog;

use ndarray::{Array2, ArrayView2};

use crate::{Pipeline, Primitive, Quantizer};

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

quantizers! { minmax, scalar }

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
