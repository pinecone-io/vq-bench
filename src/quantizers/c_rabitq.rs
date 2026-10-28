//! `c_rabitq`: clustered RaBitQ scored by the paper's 4-bit query estimator.
//!
//! Encode assigns each vector to its nearest k-means centroid, rotates the unit
//! residual, and stores the sign bits, the residual norm, the sign alignment, and
//! `<residual, centroid>`. Score quantizes the rotated query to 4 bits, estimates
//! the residual inner product, and adds the exact centroid dot.
//!
//! This is not catalog `rabitq` with k-means swapped in for `Center`. That family
//! still scores with `CastAngular`.

use anyhow::{ensure, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngExt;

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::{take, take_len, CodeLayout, ModelField};
use crate::{math, Params, Primitive, Quantizer};

/// Lloyd steps, the same budget as the `Kmeans` rounder.
const LLOYD_ITERS: usize = 25;
/// Query coordinates are quantized to this many bits (`2^4 - 1` positive steps).
const QUERY_STEPS: f32 = 15.0;
/// A sign alignment below this cannot scale an estimate.
const MIN_DOT: f32 = 1e-3;

/// The `c_rabitq` family. `centroids` is range-checked in `build`; `rotation`
/// defaults to Hadamard.
pub struct CRaBitQ {
    centroids: usize,
    seed: u64,
    dim: usize,
    rot: Box<dyn Primitive>,
}

/// 4-bit grid of one rotated query, plus the scalar facts the estimator folds in.
struct QueryGrid {
    /// `|q - c| < |q|`: the residual is estimated against `q - c`, which also adds
    /// the stored `<residual, centroid>`.
    centred: bool,
    gnorm: f32,
    /// Exact `<centroid, query>`.
    cq: f32,
    v_l: f32,
    delta: f32,
    s_q: f32,
    qbar: Vec<u8>,
}

impl CRaBitQ {
    fn index_bits(&self) -> usize {
        (self.centroids as u32).next_power_of_two().trailing_zeros() as usize
    }

    fn rotated_dim(&self) -> usize {
        self.rot.out_dim(self.dim)
    }

    /// Sign bits of the rotated residual, then `rnorm`, `dot`, `<r, centroid>`.
    fn layout(&self, rotated: usize) -> CodeLayout {
        CodeLayout::new().bits(self.index_bits() + rotated, 1).scalars(3)
    }

    /// Dither in `(0, 1)`, one per rotated coordinate, fixed for a centroid.
    fn offsets(&self, rotated: usize) -> Vec<Vec<f32>> {
        (0..self.centroids)
            .map(|j| {
                let mut rng = math::seed(self.seed.wrapping_add(j as u64).wrapping_add(0x5157));
                (0..rotated)
                    .map(|_| (rng.random::<u8>() as f32 + 0.5) / 256.0)
                    .collect()
            })
            .collect()
    }
}

impl Quantizer for CRaBitQ {
    fn name() -> &'static str {
        "c_rabitq"
    }

    fn display_name() -> &'static str {
        "C-RaBitQ"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "rotation"]
    }

    fn describe() -> &'static str {
        "clustered RaBitQ: 1-bit residual, 4-bit query estimator (not rabitq's CastAngular)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let centroids: usize = get(p, "centroids")?;
        ensure!(
            (2..=1 << CodeLayout::MAX_BITS).contains(&centroids),
            "centroids must be in 2..={}, got {centroids}",
            1u32 << CodeLayout::MAX_BITS
        );
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Ok(Self {
            centroids,
            seed,
            dim,
            rot: rotation.stage(seed),
        })
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        assert_eq!(vectors.ncols(), self.dim, "c_rabitq built for another dim");
        let centroids = math::lloyd_kmeans(vectors, self.centroids, LLOYD_ITERS, self.seed);
        let rot = self.rot.fit(vectors, queries);
        pack_state(&centroids, &rot)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        assert_eq!(vectors.ncols(), self.dim, "c_rabitq built for another dim");
        let (centroids, rot_model) = unpack_state(model);
        let idx = math::nearest_centroid(vectors, centroids.view());
        let n = vectors.nrows();
        let mut residual = vectors.to_owned();
        let mut rc = Array1::<f32>::zeros(n);
        for i in 0..n {
            let c = centroids.row(idx[i] as usize);
            residual.row_mut(i).zip_mut_with(&c, |r, &ci| *r -= ci);
            rc[i] = residual.row(i).dot(&c);
        }
        let rnorm: Array1<f32> = residual
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt())
            .collect();
        math::scale_rows(&mut residual, math::reciprocal(rnorm.view()).view());
        self.rot.apply(rot_model, &mut residual, &[]);
        let rotated = residual.ncols();
        assert_eq!(rotated, self.rotated_dim(), "rotation width changed");

        let ib = self.index_bits();
        let sqrt_d = (rotated as f32).sqrt();
        let mut levels = Array2::<u32>::zeros((n, ib + rotated));
        let mut dot = Array1::<f32>::ones(n);
        for i in 0..n {
            let mut row = levels.row_mut(i);
            for b in 0..ib {
                row[b] = (idx[i] >> b) & 1;
            }
            if rnorm[i] == 0.0 {
                rc[i] = 0.0;
                continue; // code 0, dot 1
            }
            let mut l1 = 0.0f32;
            for j in 0..rotated {
                let coord = residual[[i, j]];
                row[ib + j] = u32::from(coord >= 0.0);
                l1 += coord.abs();
            }
            dot[i] = l1 / sqrt_d;
        }
        self.layout(rotated).pack(levels.view(), &[rnorm.view(), dot.view(), rc.view()])
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        if codes.is_empty() {
            return Array2::zeros((0, self.dim));
        }
        let (centroids, rot_model) = unpack_state(model);
        let (levels, [rnorm, dot, _rc]) = self.layout(self.rotated_dim()).unpack::<3>(codes);
        let ib = self.index_bits();
        let rotated = self.rotated_dim();
        let sqrt_d = (rotated as f32).sqrt();
        let mut oriented = Array2::<f32>::zeros((codes.len(), rotated));
        let mut indexes = Vec::with_capacity(codes.len());
        for (i, row) in levels.rows().into_iter().enumerate() {
            indexes.push(read_index(row, ib));
            if rnorm[i] == 0.0 || dot[i] < MIN_DOT {
                continue;
            }
            let scale = rnorm[i] / (sqrt_d * dot[i]);
            for j in 0..rotated {
                let sign = if row[ib + j] == 0 { -1.0 } else { 1.0 };
                oriented[[i, j]] = sign * scale;
            }
        }
        let mut out = self.rot.reconstruct(rot_model, &[], Some(oriented.view()));
        for (i, &index) in indexes.iter().enumerate() {
            let mut row = out.row_mut(i);
            row += &centroids.row(index);
        }
        out
    }

    fn score(&self, model: &[u8], queries: ArrayView2<f32>, codes: &[&[u8]]) -> Array2<f32> {
        assert_eq!(queries.ncols(), self.dim, "c_rabitq built for another dim");
        let mut out = Array2::<f32>::zeros((queries.nrows(), codes.len()));
        if queries.nrows() == 0 || codes.is_empty() {
            return out;
        }
        let (centroids, rot_model) = unpack_state(model);
        let rotated = self.rotated_dim();
        let (levels, [rnorm, dot, rc]) = self.layout(rotated).unpack::<3>(codes);
        let ib = self.index_bits();
        let indexes: Vec<usize> = levels
            .rows()
            .into_iter()
            .map(|row| read_index(row, ib))
            .collect();

        let mut q_rot = queries.to_owned();
        self.rot.apply_queries(rot_model, &mut q_rot);
        let mut c_rot = centroids.clone();
        self.rot.apply(rot_model, &mut c_rot, &[]);
        let cnorm2: Vec<f32> = centroids.rows().into_iter().map(|c| c.dot(&c)).collect();
        let offsets = self.offsets(rotated);

        for qi in 0..queries.nrows() {
            let q = queries.row(qi);
            let qnorm2 = q.dot(&q);
            let grids: Vec<QueryGrid> = (0..self.centroids)
                .map(|j| {
                    prepare_list(
                        q,
                        qnorm2,
                        centroids.row(j),
                        cnorm2[j],
                        q_rot.row(qi),
                        c_rot.row(j),
                        &offsets[j],
                    )
                })
                .collect();
            for (ci, &index) in indexes.iter().enumerate() {
                let grid = &grids[index];
                let est = if grid.gnorm == 0.0 || dot[ci] < MIN_DOT {
                    0.0
                } else {
                    quantized_ip(levels.row(ci), ib, grid) / dot[ci]
                };
                let mut score = grid.cq + rnorm[ci] * grid.gnorm * est;
                if grid.centred {
                    score += rc[ci];
                }
                out[[qi, ci]] = score;
            }
        }
        out
    }
}

fn read_index(row: ArrayView1<u32>, bits: usize) -> usize {
    let mut index = 0usize;
    for b in 0..bits {
        index |= (row[b] as usize) << b;
    }
    index
}

fn pack_state(centroids: &Array2<f32>, rot: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    centroids.write(&mut buf);
    crate::coding::put_len(&mut buf, rot.len());
    buf.extend_from_slice(rot);
    buf
}

fn unpack_state(model: &[u8]) -> (Array2<f32>, &[u8]) {
    let mut cur = model;
    let centroids = Array2::<f32>::read(&mut cur);
    let n = take_len(&mut cur);
    let rot = take(&mut cur, n);
    assert!(cur.is_empty(), "trailing model bytes");
    (centroids, rot)
}

/// 4-bit levels of `q_prime` on `[min, max]`, dithered by `offsets` in `(0, 1)`.
fn quantize(q_prime: &[f32], offsets: &[f32]) -> (f32, f32, f32, Vec<u8>) {
    let v_l = q_prime.iter().copied().fold(f32::INFINITY, f32::min);
    let v_r = q_prime.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let delta = (v_r - v_l) / QUERY_STEPS;
    if delta <= 0.0 {
        return (v_l, 0.0, 0.0, vec![0; q_prime.len()]);
    }
    let mut s_q = 0.0f32;
    let mut qbar = Vec::with_capacity(q_prime.len());
    for (i, &coord) in q_prime.iter().enumerate() {
        let level = ((coord - v_l) / delta + offsets[i]).floor().clamp(0.0, QUERY_STEPS);
        let level = level as u8;
        qbar.push(level);
        s_q += level as f32;
    }
    (v_l, delta, s_q, qbar)
}

fn prepare_list(
    q: ArrayView1<f32>,
    qnorm2: f32,
    centroid: ArrayView1<f32>,
    cnorm2: f32,
    q_rot: ArrayView1<f32>,
    c_rot: ArrayView1<f32>,
    offsets: &[f32],
) -> QueryGrid {
    let qnorm = qnorm2.sqrt();
    let cq = centroid.dot(&q);
    let gnorm_c = (qnorm2 + cnorm2 - 2.0 * cq).max(0.0).sqrt();
    // A tie stays on the plain query: its scale is no larger.
    let centred = gnorm_c < qnorm;
    let gnorm = if centred { gnorm_c } else { qnorm };
    if gnorm == 0.0 {
        return QueryGrid {
            centred,
            gnorm,
            cq,
            v_l: 0.0,
            delta: 0.0,
            s_q: 0.0,
            qbar: Vec::new(),
        };
    }
    let q_prime: Vec<f32> = if centred {
        (0..q_rot.len()).map(|i| (q_rot[i] - c_rot[i]) / gnorm).collect()
    } else {
        (0..q_rot.len()).map(|i| q_rot[i] / gnorm).collect()
    };
    let (v_l, delta, s_q, qbar) = quantize(&q_prime, offsets);
    QueryGrid { centred, gnorm, cq, v_l, delta, s_q, qbar }
}

/// `<q_hat, sign / sqrt(D)>`, with `q_hat = v_l + delta * level` and `sign = ±1`.
fn quantized_ip(row: ArrayView1<u32>, index_bits: usize, grid: &QueryGrid) -> f32 {
    let d = grid.qbar.len();
    let mut acc = 0.0f32;
    let mut pop = 0.0f32;
    for i in 0..d {
        if row[index_bits + i] != 0 {
            pop += 1.0;
            acc += grid.qbar[i] as f32;
        }
    }
    let sqrt_d = (d as f32).sqrt();
    (grid.delta * (2.0 * acc - grid.s_q) + grid.v_l * (2.0 * pop - d as f32)) / sqrt_d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::params;
    use ndarray::Array2;
    use serde_json::json;

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// The `c_rabitq` quantizer, with params given as configs give them.
    fn c_rabitq(centroids: usize, rotation: &str, seed: u64, dim: usize) -> Result<CRaBitQ> {
        let p = params(&[("centroids", json!(centroids)), ("rotation", json!(rotation))]);
        CRaBitQ::build(&p, seed, dim)
    }

    #[test]
    fn rejects_bad_params() {
        assert!(c_rabitq(1, "hadamard", 1, 64).is_err()); // < 2 centroids
        assert!(c_rabitq(257, "hadamard", 1, 64).is_err()); // > 256 centroids
        assert!(c_rabitq(16, "nope", 1, 64).is_err()); // unknown rotation
        assert!(c_rabitq(2, "hadamard", 1, 64).is_ok());
        assert!(c_rabitq(256, "full", 1, 64).is_ok());
        let bare = params(&[("centroids", json!(16))]);
        assert!(CRaBitQ::build(&bare, 1, 64).is_ok());
    }

    /// The packed levels reproduce `<q_hat, ±1/sqrt(D)>` coordinate-wise.
    #[test]
    fn four_bit_inner_matches_explicit_dot() {
        let q_prime = [0.2f32, -0.5, 0.1, 0.7, -0.3, 0.4, 0.0, -0.2];
        let offsets = [0.1f32, 0.3, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6];
        let (v_l, delta, s_q, qbar) = quantize(&q_prime, &offsets);
        let grid = QueryGrid {
            centred: false,
            gnorm: 1.0,
            cq: 0.0,
            v_l,
            delta,
            s_q,
            qbar,
        };
        let bits = Array1::from(vec![0u32, 1, 0, 1, 1, 0, 0, 1]); // no index bits
        let got = quantized_ip(bits.view(), 0, &grid);
        let sqrt_d = (q_prime.len() as f32).sqrt();
        let mut direct = 0.0f32;
        for i in 0..q_prime.len() {
            let q_hat = v_l + delta * grid.qbar[i] as f32;
            let sign = if bits[i] == 0 { -1.0 } else { 1.0 };
            direct += q_hat * sign / sqrt_d;
        }
        assert!((got - direct).abs() < 1e-5, "{got} vs {direct}");
    }

    /// A constant query has no step: the estimator is the exact constant inner product.
    #[test]
    fn constant_query_uses_the_lower_endpoint() {
        let q_prime = [0.25f32; 4];
        let (v_l, delta, s_q, qbar) = quantize(&q_prime, &[0.5; 4]);
        assert_eq!(delta, 0.0);
        assert!(qbar.iter().all(|&level| level == 0));
        assert_eq!(s_q, 0.0);
        let grid = QueryGrid {
            centred: false,
            gnorm: 1.0,
            cq: 0.0,
            v_l,
            delta,
            s_q,
            qbar,
        };
        let bits = Array1::from(vec![1u32, 1, 1, 0]);
        let got = quantized_ip(bits.view(), 0, &grid);
        // signs +1,+1,+1,-1 sum to 2; <v_l, signs/sqrt(4)> = 0.25 * 2 / 2.
        assert!((got - 0.25).abs() < 1e-5, "{got}");
    }

    /// Build, encode, and score. Dim 20 is not a multiple of 64, so Hadamard pads.
    #[test]
    fn encodes_and_scores() {
        let mut rng = math::seed(3);
        let d = 20;
        let v: Array2<f32> = math::gaussian(&mut rng, (30, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (4, d));
        let codec = c_rabitq(4, "hadamard", 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let implicit = CRaBitQ::build(&params(&[("centroids", json!(4))]), 1, d).unwrap();
        assert_eq!(codes, implicit.encode(&implicit.fit(v.view(), None), v.view()));
        assert!(codes.iter().all(|c| c.len() == codes[0].len() && !c.is_empty()));
        let r = refs(&codes);
        let scored = codec.score(&model, q.view(), &r);
        assert_eq!(scored.dim(), (4, 30));
        assert!(scored.iter().all(|s| s.is_finite()));
        assert_eq!(scored, codec.score(&model, q.view(), &r));
        let recon = codec.reconstruct(&model, &r);
        assert_eq!(recon.dim(), v.dim());
        assert!(recon.iter().all(|s| s.is_finite()));
    }

    /// The 4-bit estimator is not `Kmeans -> Normalize -> Rotate -> CastAngular(1)`.
    #[test]
    fn scores_differ_from_cast_angular() {
        use crate::{AsQuantizer, CastAngular, Kmeans, Normalize, Pipeline};
        let mut rng = math::seed(9);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (40, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));
        let angular = AsQuantizer(
            Pipeline::new(
                d,
                vec![
                    Box::new(Kmeans::new(8, 1)),
                    Box::new(Normalize),
                    super::Rotation::Hadamard.stage(1),
                    Box::new(CastAngular::new(1)),
                ],
            )
            .unwrap(),
        );
        let custom = c_rabitq(8, "hadamard", 1, d).unwrap();
        let ma = angular.fit(v.view(), None);
        let mc = custom.fit(v.view(), None);
        let sa = angular.score(&ma, q.view(), &refs(&angular.encode(&ma, v.view())));
        let sc = custom.score(&mc, q.view(), &refs(&custom.encode(&mc, v.view())));
        let gap: f32 = sa.iter().zip(sc.iter()).map(|(a, c)| (a - c).abs()).sum::<f32>()
            / sa.len() as f32;
        assert!(gap > 1e-3, "mean |CastAngular - estimator| = {gap}");
    }

    /// Centroid dot plus the residual estimate: slope of estimate on true dot is ~ 1.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(5);
        let d = 128;
        let v: Array2<f32> = math::gaussian(&mut rng, (160, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (16, d));
        for rotation in ["full", "hadamard"] {
            let codec = c_rabitq(16, rotation, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let scored = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = scored.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.25, "{rotation} slope {}", se / st);
        }
    }
}
