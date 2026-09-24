//! `pq_rr`: PQ behind a PCA reduction and a random rotation — keep the top `rank`
//! principal axes, spread their variance evenly with a Haar rotation, then PQ.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::pq::Pq;
use crate::{Center, Params, PcaRotate, Pipeline, Quantizer, RandomRotate};

/// Independent seed offset for the balancing rotation.
const ROTATION_SEED: u64 = 0x5052;

/// The `pq_rr` family (Jégou et al. 2010; the `PQ_RR` baseline of Ge et al. 2013). The
/// rotation after the reduction spreads the retained variance across the coordinates
/// so contiguous segments carry comparable energy; at full rank it is distributionally
/// just a random rotation, which is why the two steps go together.
pub struct PqRr(pub Pipeline);

impl PqRr {
    /// `Center -> PcaRotate(rank) -> RandomRotate -> PQ` over the `rank` retained dims,
    /// composed via [`Pq::pipeline`], which also validates `centroids` and `section_dim`.
    pub fn pipeline(
        centroids: usize,
        section_dim: usize,
        rank: usize,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!((1..=dim).contains(&rank), "rank must be in 1..={dim}, got {rank}");
        let pq = Pq::pipeline(centroids, section_dim, seed, rank)?;
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(PcaRotate::truncated(rank)),
                Box::new(RandomRotate::new(seed ^ ROTATION_SEED)),
                Box::new(pq),
            ],
        )
    }
}

impl Quantizer for PqRr {
    fn name() -> &'static str {
        "pq_rr"
    }

    fn display_name() -> &'static str {
        "PQ-RR"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "section_dim", "rank"]
    }

    fn describe() -> &'static str {
        "Center -> PcaRotate(rank) -> RandomRotate -> SegmentSplit(section_dim) -> [Kmeans(centroids)]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(
            get(p, "centroids")?,
            get(p, "section_dim")?,
            get_or(p, "rank", dim)?,
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
    use crate::util::testing::{assert_close, params, refs, with_variances};
    use crate::{AsQuantizer, Primitive, Resize};
    use ndarray::{Array2, Axis};
    use serde_json::json;

    /// The `pq_rr` quantizer over input dim `dim`; `rank = None` leaves the default.
    fn pq_rr(
        centroids: usize,
        section_dim: usize,
        rank: Option<usize>,
        dim: usize,
    ) -> Result<PqRr> {
        let mut p =
            params(&[("centroids", json!(centroids)), ("section_dim", json!(section_dim))]);
        if let Some(rank) = rank {
            p.insert("rank".into(), json!(rank));
        }
        PqRr::build(&p, 1, dim)
    }

    /// A steep geometric spectrum, halving per axis.
    fn spectrum(d: usize) -> Vec<f32> {
        (0..d).map(|j| 2f32.powi(-(j as i32))).collect()
    }

    #[test]
    fn rejects_bad_params() {
        assert!(pq_rr(1, 8, None, 64).is_err()); // < 2 centroids
        assert!(pq_rr(16, 0, None, 64).is_err()); // section_dim < 1
        assert!(pq_rr(16, 8, Some(0), 64).is_err()); // rank < 1
        assert!(pq_rr(16, 8, Some(65), 64).is_err()); // rank > dim
        assert!(pq_rr(16, 8, Some(4), 64).is_err()); // section_dim > rank
        assert!(pq_rr(16, 65, None, 64).is_err()); // section_dim > rank = dim
        assert!(pq_rr(16, 8, None, 64).is_ok());
        assert!(pq_rr(16, 8, Some(32), 64).is_ok());
        assert!(pq_rr(16, 8, Some(36), 64).is_ok()); // ragged rank, as pq allows
    }

    /// The point of the RR step: PCA alone leaves the coordinate variances ordered along
    /// the spectrum, the random rotation after it spreads them roughly evenly -- and the
    /// `section_dim`-wide segments PQ sees more evenly still.
    #[test]
    fn random_rotation_balances_the_variances() {
        let (d, section_dim) = (64, 8);
        let variances: Vec<f32> = (0..d).map(|j| 2f32.powf(-0.125 * j as f32)).collect();
        let v = with_variances(4000, &variances, 3);
        let (center, pca, rotate) = (Center, PcaRotate::truncated(d), RandomRotate::new(5));
        let mut x = v.clone();
        center.apply(&center.fit(x.view(), None), &mut x, &[]);
        pca.apply(&pca.fit(x.view(), None), &mut x, &[]);
        let spread = |x: &Array2<f32>, width: usize| {
            let var = x.var_axis(Axis(0), 0.0);
            let parts: Vec<f32> =
                var.as_slice().unwrap().chunks(width).map(|c| c.iter().sum()).collect();
            let (max, min) = parts.iter().fold((0f32, f32::INFINITY), |(hi, lo), &p| {
                (hi.max(p), lo.min(p))
            });
            max / min
        };
        assert!(spread(&x, 1) > 100.0, "PCA alone keeps the spectrum's spread: {}", spread(&x, 1));
        rotate.apply(&rotate.fit(x.view(), None), &mut x, &[]);
        assert!(spread(&x, 1) < 4.0, "coordinate variances should balance: {}", spread(&x, 1));
        let segments = spread(&x, section_dim);
        assert!(segments < 1.5, "segment variances should balance: {segments}");
    }

    /// PQ-RR scores exactly against its own (lossy) reconstruction, with and without a
    /// reduction: the child's ADC scores pass through the projection unchanged.
    #[test]
    fn score_matches_reconstruction() {
        let v: Array2<f32> = math::gaussian(&mut math::seed(1), (60, 32));
        let q: Array2<f32> = math::gaussian(&mut math::seed(2), (8, 32));
        for rank in [None, Some(24)] {
            let codec = pq_rr(16, 8, rank, 32).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let r = refs(&codes);
            let recon = codec.reconstruct(&model, &r);
            assert_eq!(recon.dim(), v.dim());
            assert_close(&codec.score(&model, q.view(), &r), &q.dot(&recon.t()), 1e-3);
        }
    }

    /// Halving the rank halves the code: the segments cover `rank` dims, not `dim`. The
    /// error then sits above the tail the projection drops -- the retained subspace is
    /// quantized, not discarded -- and below plain PQ of the same leading coordinates,
    /// which here are already principal, so the difference is the balancing rotation.
    #[test]
    fn reduction_shrinks_the_code_and_bounds_the_error() {
        let (d, k, section_dim) = (32, 16, 4);
        let v = with_variances(1500, &spectrum(d), 4);
        let full = pq_rr(16, section_dim, None, d).unwrap();
        let reduced = pq_rr(16, section_dim, Some(k), d).unwrap();
        let mse = |codec: &dyn Quantizer| {
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let recon = codec.reconstruct(&model, &refs(&codes));
            ((&v - &recon).mapv(|e| e * e).mean().unwrap(), codes[0].len())
        };
        let (err, bytes) = mse(&reduced);
        assert_eq!(bytes * 2, mse(&full).1);

        // The empirical tail: what the top-k projection alone cannot keep.
        let head = [Box::new(Center) as Box<dyn Primitive>, Box::new(PcaRotate::truncated(k))];
        let mut y = v.clone();
        let models: Vec<Vec<u8>> = head
            .iter()
            .map(|s| {
                let m = s.fit(y.view(), None);
                s.apply(&m, &mut y, &[]);
                m
            })
            .collect();
        let mut projected = y;
        for (s, m) in head.iter().zip(&models).rev() {
            projected = s.reconstruct(m, &[], Some(projected.view()));
        }
        let floor = (&v - &projected).mapv(|e| e * e).mean().unwrap();
        assert!(err >= floor, "error {err} below the projection floor {floor}");

        let truncated = AsQuantizer(
            Pipeline::new(
                d,
                vec![
                    Box::new(Center) as Box<dyn Primitive>,
                    Box::new(Resize::to(k)),
                    Box::new(Pq::pipeline(16, section_dim, 1, k).unwrap()),
                ],
            )
            .unwrap(),
        );
        let (dropped, _) = mse(&truncated);
        assert!(err <= dropped, "pq_rr {err} should not lose to truncate-then-PQ {dropped}");
    }
}
