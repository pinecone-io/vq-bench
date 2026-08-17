//! `opq`: Optimized Product Quantization — learn a rotation minimizing PQ error,
//! then PQ (segment split + per-segment k-means) on the rotated data.

use anyhow::{bail, Context, Result};
use serde_json::Value;

use super::catalog::{get, get_or, FromParam};
use super::opq_p::OpqP;
use super::pq::Pq;
use crate::{OptimizePq, Params, Pipeline, Primitive, Quantizer};

/// Alternation steps when a config does not say (Ge et al. 2013 use ~15).
const DEFAULT_ITERS: usize = 15;

/// Where the alternation starts when a config does not say. `Eigen` is what `describe`
/// advertises and what Ge et al. 2013 report as the best variant; it also makes `opq`
/// a strict extension of `opq_p` — the same head, plus the alternation — so a comparison
/// between the two measures the alternation rather than the head.
const DEFAULT_INIT: Init = Init::Eigen;

/// What the alternation is handed to start from: the raw data, or the parametric
/// rotation the [`OpqP`] family applies. The alternation is locally optimal, so the two
/// land in different places — Ge et al. 2013 report the parametric start as the best of
/// the variants they test.
#[derive(Clone, Copy)]
pub enum Init {
    Identity,
    Eigen,
}

impl FromParam for Init {
    fn from_value(v: &Value) -> Result<Self> {
        match v.as_str().context("must be a string")? {
            "identity" => Ok(Init::Identity),
            "eigen" => Ok(Init::Eigen),
            other => bail!("unknown init `{other}` (expected `identity` or `eigen`)"),
        }
    }
}

/// The `opq` family. A learned OPQ rotation, then PQ over `section_dim`-column
/// segments with `centroids` codewords each (distinct seed per segment).
pub struct Opq(pub Pipeline);

impl Opq {
    /// A learned OPQ rotation — `iters` alternation steps, behind the parametric head at
    /// `init = Eigen` — then PQ over `section_dim`-column segments with `centroids`
    /// codewords each (distinct seed per segment), composed via [`Pq::pipeline`], which
    /// also validates the params.
    pub fn pipeline(
        centroids: usize,
        section_dim: usize,
        iters: usize,
        init: Init,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        let pq = Pq::pipeline(centroids, section_dim, seed, dim)?;
        let mut stages: Vec<Box<dyn Primitive>> = Vec::new();
        if let Init::Eigen = init {
            stages.extend(OpqP::head(section_dim));
        }
        // At `iters == 0` there is nothing to alternate, and the stage would store a
        // whole identity rotation to say so.
        if iters > 0 {
            stages.push(Box::new(OptimizePq::new(centroids, section_dim, iters, seed)));
        }
        stages.push(Box::new(pq));
        Pipeline::new(dim, stages)
    }
}

impl Quantizer for Opq {
    fn name() -> &'static str {
        "opq"
    }

    fn display_name() -> &'static str {
        "OPQ"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "section_dim", "iters", "init"]
    }

    fn describe() -> &'static str {
        "Center -> PcaRotate -> BalanceParts(section_dim) -> OptimizePq(iters) -> PQ"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(
            get(p, "centroids")?,
            get(p, "section_dim")?,
            get_or(p, "iters", DEFAULT_ITERS)?,
            get_or(p, "init", DEFAULT_INIT)?,
            seed,
            dim,
        )?))
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

    /// The `opq` quantizer over input dim `dim`, at the family's default init and iters.
    fn opq(centroids: usize, section_dim: usize, seed: u64, dim: usize) -> Result<Opq> {
        let p = params(&[("centroids", json!(centroids)), ("section_dim", json!(section_dim))]);
        Opq::build(&p, seed, dim)
    }

    /// Low-rank (strongly correlated) data, where a decorrelating rotation helps PQ.
    fn correlated(n: usize, d: usize, seed: u64) -> Array2<f32> {
        let g = math::gaussian(&mut math::seed(seed), (n, d / 4));
        let mix = math::gaussian(&mut math::seed(seed ^ 0xabc), (d / 4, d));
        math::matmul(g.view(), mix.view())
    }

    /// Mean squared reconstruction error of a fitted quantizer over its own fit set.
    fn recon_error(codec: &impl Quantizer, v: &Array2<f32>) -> f32 {
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let recon = codec.reconstruct(&model, &refs(&codes));
        (&recon - v).mapv(|e| e * e).sum() / v.len() as f32
    }

    #[test]
    fn rejects_bad_params() {
        assert!(opq(1, 8, 1, 64).is_err()); // < 2 centroids
        assert!(opq(257, 8, 1, 64).is_err()); // > 256 centroids
        assert!(opq(16, 0, 1, 64).is_err()); // section_dim < 1
        assert!(opq(16, 65, 1, 64).is_err()); // section_dim > dim
        assert!(opq(16, 8, 1, 64).is_ok());
    }

    /// `init` takes the two named rotations and nothing else; `iters` is a plain count.
    #[test]
    fn reads_init_and_iters() {
        let with = |init: &str, iters: u64| {
            let p = params(&[
                ("centroids", json!(16)),
                ("section_dim", json!(8)),
                ("init", json!(init)),
                ("iters", json!(iters)),
            ]);
            Opq::build(&p, 1, 64)
        };
        assert!(with("identity", 15).is_ok());
        assert!(with("eigen", 0).is_ok());
        assert!(with("pca", 15).is_err());
    }

    /// The `opq` quantizer with an explicit init and iteration count.
    fn opq_with(init: &str, iters: u64, section_dim: usize, seed: u64, dim: usize) -> Opq {
        let p = params(&[
            ("centroids", json!(16)),
            ("section_dim", json!(section_dim)),
            ("init", json!(init)),
            ("iters", json!(iters)),
        ]);
        Opq::build(&p, seed, dim).unwrap()
    }

    /// Running the alternation behind the parametric head reaches a better local optimum
    /// than starting from the raw coordinates (Ge et al. 2013, Table 3).
    #[test]
    fn eigen_init_beats_identity_init() {
        let v = correlated(400, 32, 3);
        let identity = recon_error(&opq_with("identity", 15, 4, 5, 32), &v);
        let eigen = recon_error(&opq_with("eigen", 15, 4, 5, 32), &v);
        assert!(eigen < identity, "eigen {eigen} not better than identity {identity}");
    }

    /// The default init is the parametric head, so a config naming neither `init` nor
    /// `iters` gets what `describe` advertises. Pinned because the alternative reads as
    /// a plausible default and silently turns `opq` into a different pipeline from
    /// `opq_p` rather than an extension of it — which makes the two incomparable.
    #[test]
    fn the_default_init_is_the_parametric_head() {
        let v = correlated(200, 32, 11);
        let default = opq(16, 8, 1, 32).unwrap();
        assert_eq!(
            default.fit(v.view(), None),
            opq_with("eigen", DEFAULT_ITERS as u64, 8, 1, 32).fit(v.view(), None),
            "`opq` with no `init` must match `init=eigen`"
        );
    }

    /// With the parametric head and nothing to alternate, `opq` *is* `opq_p` — no
    /// leftover identity rotation, so the two also carry the same model.
    #[test]
    fn eigen_init_at_zero_iters_is_opq_p() {
        let v = correlated(200, 32, 7);
        let opq = opq_with("eigen", 0, 8, 1, 32);
        let opq_p = OpqP::build(
            &params(&[("centroids", json!(16)), ("section_dim", json!(8))]),
            1,
            32,
        )
        .unwrap();
        assert_eq!(opq.fit(v.view(), None), opq_p.fit(v.view(), None));
    }

    /// OPQ scores exactly against its own (lossy) reconstruction: rotate, then
    /// per-segment ADC summed, then rotate back — the query rotation cancels.
    #[test]
    fn score_matches_reconstruction() {
        let v: Array2<f32> = math::gaussian(&mut math::seed(1), (60, 32));
        let q: Array2<f32> = math::gaussian(&mut math::seed(2), (8, 32));
        let codec = opq(16, 8, 1, 32).unwrap(); // 8-dim segments (4 of them), k=16
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        let recon = codec.reconstruct(&model, &r);
        assert_close(&codec.score(&model, q.view(), &r), &q.dot(&recon.t()), 1e-3);
    }
}
