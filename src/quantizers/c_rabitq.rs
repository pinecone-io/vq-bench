//! `c_rabitq`: clustered RaBitQ scored by the paper's 4-bit query estimator.
//!
//! Encode assigns each vector to its nearest squared-L2 centroid, pads the unit
//! residual to `D = max(256, ceil(d/64)*64)`, and applies four rounds of random
//! signs, an orthonormal 64-point Walsh–Hadamard (`H_64 / 8`, Fino–Algazi), and a
//! permutation. That structured map is the Fastfood / FJLT rotation the RaBitQ
//! estimator is proved against (Gao & Long, SIGMOD 2024; the extended estimator
//! quantizes the rotated query to 4 bits). The centroid id is packed as extra
//! 1-bit fields ahead of the residual signs. The code also stores the residual
//! norm, the sign alignment, and `<residual, centroid>`. Score adds the exact
//! centroid dot to the unbiased residual estimate.
//!
//! This is not catalog `rabitq` with k-means swapped in for `Center`. That family
//! still scores with `CastAngular`.

use anyhow::{ensure, Result};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::catalog::get;
use crate::coding::{CodeLayout, ModelField};
use crate::{math, Params, Quantizer};

/// Ten Lloyd rounds. Assignment stays squared L2; this bench scores inner products.
const LLOYD_ITERS: usize = 10;
/// Query coordinates are quantized to this many bits (`2^4 - 1` positive steps).
const QUERY_STEPS: f32 = 15.0;
/// A sign alignment below this cannot scale an estimate.
const MIN_DOT: f32 = 1e-3;
/// Walsh–Hadamard block. `D` is a multiple of this, and at least [`MIN_PAD`].
const BLOCK: usize = 64;
/// Short vectors are padded out to this width before the orthogonal map.
const MIN_PAD: usize = 256;
/// Rounds of signs, block Hadamard, and a permutation.
const ROUNDS: usize = 4;
/// List ids above this are rejected. The id is extra 1-bit fields, so the cap is
/// not the 8-bit packer width; `2^16` covers a √N codebook on these suites.
const MAX_CENTROIDS: usize = 1 << 16;
/// Separates the per-centroid dither stream from the rotation stream.
const OFFSET_SALT: u64 = 0x5157;

/// The `c_rabitq` family. `centroids` is range-checked in `build`. The orthogonal
/// map is a function of the run seed and `dim`.
pub struct CRaBitQ {
    centroids: usize,
    seed: u64,
    dim: usize,
    width: usize,
    rounds: Vec<Round>,
}

/// One round: diagonal ±1, then `H_64 / 8` on each block, then a permutation.
struct Round {
    signs: Vec<f32>,
    /// `perm[i]` is the destination of coordinate `i`.
    perm: Vec<usize>,
}

/// 4-bit grid of one rotated query, plus the scalar facts the estimator folds in.
struct QueryGrid {
    /// `|q - c| < |q|`: the residual is estimated against `q - c`, which also adds
    /// the stored `<residual, centroid>`.
    centred: bool,
    gnorm: f32,
    /// Exact `<centroid, query>`, accumulated in f64 and stored as f32.
    cq: f32,
    v_l: f32,
    delta: f32,
    s_q: f32,
    qbar: Vec<u8>,
}

/// SplitMix64 (Steele, Lea, and Flood). Local so the map and the dither do not
/// go through `math::seed`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl Round {
    fn draw(rng: &mut SplitMix64, width: usize) -> Self {
        let signs = (0..width)
            .map(|_| if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut perm: Vec<usize> = (0..width).collect();
        for i in (1..width).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        Self { signs, perm }
    }

    /// `perm ∘ (H/8) ∘ signs`. Each factor is an involution, so the inverse is
    /// `signs ∘ (H/8) ∘ perm⁻¹`.
    fn forward(&self, buf: &mut [f32]) {
        for (x, &sign) in buf.iter_mut().zip(&self.signs) {
            *x *= sign;
        }
        hadamard_blocks(buf);
        let src = buf.to_vec();
        for (i, &dest) in self.perm.iter().enumerate() {
            buf[dest] = src[i];
        }
    }

    fn inverse(&self, buf: &mut [f32]) {
        let src = buf.to_vec();
        for (i, &dest) in self.perm.iter().enumerate() {
            buf[i] = src[dest];
        }
        hadamard_blocks(buf);
        for (x, &sign) in buf.iter_mut().zip(&self.signs) {
            *x *= sign;
        }
    }
}

impl CRaBitQ {
    fn index_bits(&self) -> usize {
        (self.centroids as u32).next_power_of_two().trailing_zeros() as usize
    }

    /// Sign bits of the rotated residual, then `rnorm`, `dot`, `<r, centroid>`.
    /// The centroid id occupies the leading 1-bit fields.
    fn layout(&self) -> CodeLayout {
        CodeLayout::new().bits(self.index_bits() + self.width, 1).scalars(3)
    }

    /// Dither in `(0, 1)`, one per rotated coordinate, fixed for a centroid.
    fn offsets(&self) -> Vec<Vec<f32>> {
        (0..self.centroids)
            .map(|j| {
                let mut rng = SplitMix64::new(self.seed.wrapping_add(j as u64).wrapping_add(OFFSET_SALT));
                (0..self.width)
                    .map(|_| (rng.next_u64() as u8 as f32 + 0.5) / 256.0)
                    .collect()
            })
            .collect()
    }

    /// Zero-pad each row from `dim` to `D`.
    fn embed(&self, data: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(data.ncols(), self.dim, "c_rabitq built for another dim");
        let mut padded = Array2::zeros((data.nrows(), self.width));
        padded.slice_mut(s![.., ..self.dim]).assign(&data);
        padded
    }

    fn forward_rows(&self, rows: &mut Array2<f32>) {
        self.map_rows(rows, false);
    }

    fn inverse_rows(&self, rows: &mut Array2<f32>) {
        self.map_rows(rows, true);
    }

    fn map_rows(&self, rows: &mut Array2<f32>, inverse: bool) {
        assert_eq!(rows.ncols(), self.width, "orthogonal map width");
        for mut row in rows.rows_mut() {
            let mut buf: Vec<f32> = row.iter().copied().collect();
            if inverse {
                for round in self.rounds.iter().rev() {
                    round.inverse(&mut buf);
                }
            } else {
                for round in &self.rounds {
                    round.forward(&mut buf);
                }
            }
            for (dst, src) in row.iter_mut().zip(&buf) {
                *dst = *src;
            }
        }
    }

    fn rotate(&self, data: ArrayView2<f32>) -> Array2<f32> {
        let mut padded = self.embed(data);
        self.forward_rows(&mut padded);
        padded
    }

    /// Inverse map, then drop the pad. `data` is `(n, D)`.
    fn unrotate(&self, mut data: Array2<f32>) -> Array2<f32> {
        self.inverse_rows(&mut data);
        data.slice(s![.., ..self.dim]).to_owned()
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
        &["centroids"]
    }

    fn describe() -> &'static str {
        "clustered RaBitQ: 1-bit residual, 4-bit query estimator (not rabitq's CastAngular)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let centroids: usize = get(p, "centroids")?;
        ensure!(
            (2..=MAX_CENTROIDS).contains(&centroids),
            "centroids must be in 2..={MAX_CENTROIDS}, got {centroids}"
        );
        let width = padded_width(dim);
        let mut rng = SplitMix64::new(seed);
        let rounds = (0..ROUNDS).map(|_| Round::draw(&mut rng, width)).collect();
        Ok(Self { centroids, seed, dim, width, rounds })
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        assert_eq!(vectors.ncols(), self.dim, "c_rabitq built for another dim");
        let centroids = math::lloyd_kmeans(vectors, self.centroids, LLOYD_ITERS, self.seed);
        let mut buf = Vec::new();
        centroids.write(&mut buf);
        buf
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        assert_eq!(vectors.ncols(), self.dim, "c_rabitq built for another dim");
        let centroids = unpack_state(model);
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
        let rotated = self.rotate(residual.view());
        let width = self.width;

        let ib = self.index_bits();
        let sqrt_d = (width as f32).sqrt();
        let mut levels = Array2::<u32>::zeros((n, ib + width));
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
            for j in 0..width {
                let coord = rotated[[i, j]];
                row[ib + j] = u32::from(coord >= 0.0);
                l1 += coord.abs();
            }
            dot[i] = l1 / sqrt_d;
        }
        self.layout().pack(levels.view(), &[rnorm.view(), dot.view(), rc.view()])
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        if codes.is_empty() {
            return Array2::zeros((0, self.dim));
        }
        let centroids = unpack_state(model);
        let (levels, [rnorm, dot, _rc]) = self.layout().unpack::<3>(codes);
        let ib = self.index_bits();
        let sqrt_d = (self.width as f32).sqrt();
        let mut oriented = Array2::<f32>::zeros((codes.len(), self.width));
        let mut indexes = Vec::with_capacity(codes.len());
        for (i, row) in levels.rows().into_iter().enumerate() {
            indexes.push(read_index(row, ib));
            if rnorm[i] == 0.0 || dot[i] < MIN_DOT {
                continue;
            }
            let scale = rnorm[i] / (sqrt_d * dot[i]);
            for j in 0..self.width {
                let sign = if row[ib + j] == 0 { -1.0 } else { 1.0 };
                oriented[[i, j]] = sign * scale;
            }
        }
        let mut out = self.unrotate(oriented);
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
        let centroids = unpack_state(model);
        let (levels, [rnorm, dot, rc]) = self.layout().unpack::<3>(codes);
        let ib = self.index_bits();
        let indexes: Vec<usize> = levels.rows().into_iter().map(|row| read_index(row, ib)).collect();

        let q_rot = self.rotate(queries);
        let c_rot = self.rotate(centroids.view());
        let cnorm2: Vec<f32> = centroids.rows().into_iter().map(|c| c.dot(&c)).collect();
        let offsets = self.offsets();

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

/// `D = max(256, ceil(d/64)*64)`.
fn padded_width(dim: usize) -> usize {
    (dim.div_ceil(BLOCK) * BLOCK).max(MIN_PAD)
}

fn read_index(row: ArrayView1<u32>, bits: usize) -> usize {
    let mut index = 0usize;
    for b in 0..bits {
        index |= (row[b] as usize) << b;
    }
    index
}

fn unpack_state(model: &[u8]) -> Array2<f32> {
    let mut cur = model;
    let centroids = Array2::<f32>::read(&mut cur);
    assert!(cur.is_empty(), "trailing model bytes");
    centroids
}

/// Orthonormal Walsh–Hadamard on every 64-block. Unnormalized `H_64` squares to
/// `64 I` and is symmetric, so scaling by `1/8` makes the block map an involution.
fn hadamard_blocks(buf: &mut [f32]) {
    debug_assert_eq!(buf.len() % BLOCK, 0);
    for block in buf.chunks_exact_mut(BLOCK) {
        let mut len = 1;
        while len < BLOCK {
            let mut i = 0;
            while i < BLOCK {
                for j in 0..len {
                    let a = block[i + j];
                    let b = block[i + j + len];
                    block[i + j] = a + b;
                    block[i + j + len] = a - b;
                }
                i += len << 1;
            }
            len <<= 1;
        }
        for x in block.iter_mut() {
            *x *= 1.0 / 8.0;
        }
    }
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
    let cq64 = centroid
        .iter()
        .zip(q.iter())
        .map(|(&c, &qi)| f64::from(c) * f64::from(qi))
        .sum::<f64>();
    let cq = cq64 as f32;
    let gnorm_c = (f64::from(qnorm2) + f64::from(cnorm2) - 2.0 * cq64).max(0.0).sqrt() as f32;
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
    fn c_rabitq(centroids: usize, seed: u64, dim: usize) -> Result<CRaBitQ> {
        let p = params(&[("centroids", json!(centroids))]);
        CRaBitQ::build(&p, seed, dim)
    }

    #[test]
    fn rejects_bad_params() {
        assert!(c_rabitq(1, 1, 64).is_err()); // < 2 centroids
        assert!(c_rabitq(MAX_CENTROIDS + 1, 1, 64).is_err());
        assert!(c_rabitq(2, 1, 64).is_ok());
        assert!(c_rabitq(1024, 1, 64).is_ok()); // past the old 8-bit id cap
        assert!(c_rabitq(MAX_CENTROIDS, 1, 64).is_ok());
        let bare = params(&[("centroids", json!(16))]);
        assert!(CRaBitQ::build(&bare, 1, 64).is_ok());
        let rotation = params(&[("centroids", json!(16)), ("rotation", json!("hadamard"))]);
        let problems = CRaBitQ::verify_params(&rotation);
        assert!(problems.iter().any(|m| m.contains("rotation")), "{problems:?}");
    }

    #[test]
    fn pads_to_at_least_256_and_a_multiple_of_64() {
        assert_eq!(padded_width(20), 256);
        assert_eq!(padded_width(64), 256);
        assert_eq!(padded_width(128), 256);
        assert_eq!(padded_width(256), 256);
        assert_eq!(padded_width(257), 320);
        assert_eq!(padded_width(384), 384);
    }

    /// Signs, `H_64 / 8`, and the permutation are involutions, in reverse order.
    #[test]
    fn rotation_inverse_restores_the_pad() {
        let codec = c_rabitq(2, 1, 20).unwrap();
        assert_eq!(codec.width, 256);
        let mut rng = math::seed(11);
        let x = math::gaussian(&mut rng, (5, 20));
        let padded = codec.embed(x.view());
        let mut y = padded.clone();
        codec.forward_rows(&mut y);
        assert!(y.iter().zip(padded.iter()).any(|(a, b)| (a - b).abs() > 1e-2));
        codec.inverse_rows(&mut y);
        let max = y
            .iter()
            .zip(padded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-3, "round trip max abs {max}");
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

    /// Build, encode, and score. Dim 20 pads to 256.
    #[test]
    fn encodes_and_scores() {
        let mut rng = math::seed(3);
        let d = 20;
        let v: Array2<f32> = math::gaussian(&mut rng, (30, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (4, d));
        let codec = c_rabitq(4, 1, d).unwrap();
        assert_eq!(codec.width, 256);
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let again = CRaBitQ::build(&params(&[("centroids", json!(4))]), 1, d).unwrap();
        assert_eq!(codes, again.encode(&again.fit(v.view(), None), v.view()));
        assert!(codes.iter().all(|c| c.len() == codes[0].len() && !c.is_empty()));
        let (levels, _) = codec.layout().unpack::<3>(&refs(&codes));
        assert_eq!(levels.ncols(), codec.index_bits() + 256);
        let r = refs(&codes);
        let scored = codec.score(&model, q.view(), &r);
        assert_eq!(scored.dim(), (4, 30));
        assert!(scored.iter().all(|s| s.is_finite()));
        assert_eq!(scored, codec.score(&model, q.view(), &r));
        let recon = codec.reconstruct(&model, &r);
        assert_eq!(recon.dim(), v.dim());
        assert!(recon.iter().all(|s| s.is_finite()));
    }

    /// A list id past 256 is extra 1-bit fields, not an 8-bit code.
    #[test]
    fn list_id_packs_past_eight_bits() {
        let mut rng = math::seed(7);
        let d = 8;
        let k = 300;
        let v: Array2<f32> = math::gaussian(&mut rng, (k, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (2, d));
        let codec = c_rabitq(k, 1, d).unwrap();
        assert!(codec.index_bits() > 8);
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let centroids = unpack_state(&model);
        let idx = math::nearest_centroid(v.view(), centroids.view());
        let (levels, _) = codec.layout().unpack::<3>(&refs(&codes));
        assert_eq!(levels.ncols(), codec.index_bits() + codec.width);
        for (i, row) in levels.rows().into_iter().enumerate() {
            assert_eq!(read_index(row, codec.index_bits()), idx[i] as usize);
        }
        let r = refs(&codes);
        let scored = codec.score(&model, q.view(), &r);
        assert!(scored.iter().all(|s| s.is_finite()));
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
                    super::super::rotation::Rotation::Hadamard.stage(1),
                    Box::new(CastAngular::new(1)),
                ],
            )
            .unwrap(),
        );
        let custom = c_rabitq(8, 1, d).unwrap();
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
        let codec = c_rabitq(16, 1, d).unwrap();
        assert_eq!(codec.width, 256);
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let scored = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = scored.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.25, "slope {}", se / st);
    }
}
