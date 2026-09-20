//! `opq_p`: the parametric half of Optimized Product Quantization — align to the
//! principal components and deal them so equal-width segments hold equal variance,
//! then PQ.

use anyhow::Result;

use super::catalog::get;
use super::pq::Pq;
use crate::{BalanceParts, Center, Params, PcaRotate, Pipeline, Quantizer};

/// The `opq_p` family. Closed-form under a Gaussian assumption, so nothing is
/// alternated; `opq` with `init=eigen` runs its alternation behind the same head.
pub struct OpqP(pub Pipeline);

impl OpqP {
    /// The parametric rotation — onto the principal axes, then dealt so equal-width parts
    /// hold the same variance product — then PQ over `section_dim`-column segments with
    /// `centroids` codewords each (distinct seed per segment), composed via
    /// [`Pq::pipeline`], which also validates the params.
    ///
    /// `opq` at `init = eigen` builds the same three stages; the two are held to that by
    /// `opq`'s `eigen_init_at_zero_iters_is_opq_p`, so changing them here without changing
    /// them there fails that test rather than silently making the families incomparable.
    pub fn pipeline(centroids: usize, section_dim: usize, seed: u64, dim: usize) -> Result<Pipeline> {
        let pq = Pq::pipeline(centroids, section_dim, seed, dim)?;
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(PcaRotate),
                Box::new(BalanceParts::new(section_dim)),
                Box::new(pq),
            ],
        )
    }
}

impl Quantizer for OpqP {
    fn name() -> &'static str {
        "opq_p"
    }

    fn display_name() -> &'static str {
        "OPQ-par"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "section_dim"]
    }

    fn describe() -> &'static str {
        "Center -> PcaRotate -> BalanceParts(section_dim) -> PQ"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(get(p, "centroids")?, get(p, "section_dim")?, seed, dim)?))
    }

    crate::pipeline_quantizer!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use crate::util::testing::{assert_close, params, refs};
    use ndarray::Array2;
    use serde_json::json;

    /// The `opq_p` quantizer over input dim `dim`.
    fn opq_p(centroids: usize, section_dim: usize, seed: u64, dim: usize) -> Result<OpqP> {
        let p = params(&[("centroids", json!(centroids)), ("section_dim", json!(section_dim))]);
        OpqP::build(&p, seed, dim)
    }

    #[test]
    fn rejects_bad_params() {
        assert!(opq_p(1, 8, 1, 64).is_err()); // < 2 centroids
        assert!(opq_p(257, 8, 1, 64).is_err()); // > 256 centroids
        assert!(opq_p(16, 0, 1, 64).is_err()); // section_dim < 1
        assert!(opq_p(16, 65, 1, 64).is_err()); // section_dim > dim
        assert!(opq_p(16, 8, 1, 64).is_ok());
    }

    /// OPQ-par scores exactly against its own (lossy) reconstruction: rotate, then
    /// per-segment ADC summed, then rotate back — the query rotation cancels.
    #[test]
    fn score_matches_reconstruction() {
        let v: Array2<f32> = math::gaussian(&mut math::seed(1), (60, 32));
        let q: Array2<f32> = math::gaussian(&mut math::seed(2), (8, 32));
        let codec = opq_p(16, 8, 1, 32).unwrap(); // 8-dim segments (4 of them), k=16
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        let recon = codec.reconstruct(&model, &r);
        assert_close(&codec.score(&model, q.view(), &r), &q.dot(&recon.t()), 1e-3);
    }
}
