//! CAST(NORMAL, b): fixed Lloyd-Max Gaussian codebook + a per-vector dequant scale S
//! -
//! Model: input dim d
//! Code for vector x: b-bit level per coordinate plus the dequant scale S
//! Apply: x --> x - S*codeword
//! Reconstruct: y --> S*codeword + y
//! Score: s --> <q, S*codeword> + s
//!
//! codeword = C[level]/sqrt(d): the codeword on the grid matched to the coordinates of a unit-norm rotated vector

use ndarray::{Array1, Array2, ArrayView2};

use crate::coding::CodeLayout;
use crate::{codebooks, coding, math, Primitive};

/// The per-vector dequant scale S for cast(normal) (the paper's cast(beta)):
/// Plain (S = 1), BiasedMse (S = <x,codeword>/||codeword||^2, least-MSE projection),
/// or Unbiased (S = ||x||^2/<x,codeword>, unbiases the inner product). This is the
/// only thing separating TurboQuant / EDEN / EDEN-unbiased.
#[derive(Clone, Copy)]
pub enum NormalScale {
    Plain,
    BiasedMse,
    Unbiased,
}

pub struct CastNormal {
    bits: u8,
    mode: NormalScale,
    codebook: Vec<f32>, // 2^bits sorted centroids for N(0,1)
}

impl CastNormal {
    /// `bits`-bit Gaussian codebook with dequant scale `mode`; the valid range
    /// (`1..=CodeLayout::MAX_BITS`) is enforced by the quantizer builder.
    pub fn new(bits: u8, mode: NormalScale) -> Self {
        Self {
            bits,
            mode,
            codebook: codebooks::lloyd_max_normal(1usize << bits),
        }
    }

    /// The code layout: `d` levels of `self.bits` bits, then the dequant scale S.
    fn layout(&self, d: usize) -> CodeLayout {
        CodeLayout::new().bits(d, self.bits).scalars(1)
    }

    /// Split each code into its packed levels (`n x d`) and its dequant scales S (`n`).
    fn split(&self, codes: &[&[u8]], d: usize) -> (Array2<u32>, Array1<f32>) {
        let (levels, [scales]) = self.layout(d).unpack::<1>(codes);
        (levels, scales)
    }

    /// Codeword values `S*codeword` per vector (codeword = C[level]/sqrt(d), scaled by S).
    fn dequant(&self, codes: &[&[u8]], d: usize) -> Array2<f32> {
        let (levels, scales) = self.split(codes, d);
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let mut out = levels.mapv(|level| self.codebook[level as usize] * inv_sqrt_d);
        math::scale_rows(&mut out, scales.view());
        out
    }
}

impl Primitive for CastNormal {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (n_v, d) = (vectors.nrows(), vectors.ncols());
        let sqrt_d = (d as f32).sqrt();
        let inv_sqrt_d = 1.0 / sqrt_d;
        // Coordinates are assumed ~ N(0, 1/d), so the fixed N(0,1) codebook is matched
        // by sqrt(d) rather than a per-vector RMS estimate. codeword = C[level]/sqrt(d).
        let mut levels = Array2::<u32>::zeros((n_v, d));
        let mut scales = Array1::<f32>::zeros(n_v);
        for (i, row) in vectors.rows().into_iter().enumerate() {
            let mut codeword = vec![0f32; d];
            for (j, &x) in row.iter().enumerate() {
                let level = codebooks::nearest(&self.codebook, x * sqrt_d); // lift to N(0,1) to pick the level
                levels[[i, j]] = level as u32;
                codeword[j] = self.codebook[level] * inv_sqrt_d;
            }
            scales[i] = match self.mode {
                NormalScale::Plain => 1.0,
                NormalScale::BiasedMse => {
                    let num: f32 = row.iter().zip(&codeword).map(|(&x, &cw)| x * cw).sum();
                    let den: f32 = codeword.iter().map(|&cw| cw * cw).sum();
                    if den > 0.0 {
                        num / den
                    } else {
                        0.0
                    }
                }
                NormalScale::Unbiased => {
                    let num: f32 = row.iter().map(|&x| x * x).sum();
                    let den: f32 = row.iter().zip(&codeword).map(|(&x, &cw)| x * cw).sum();
                    if den.abs() > 1e-12 {
                        num / den
                    } else {
                        0.0
                    }
                }
            };
        }
        self.layout(d).pack(levels.view(), &[scales.view()])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let d = vectors.ncols();
        *vectors -= &self.dequant(codes, d); // residual
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let mut out = self.dequant(codes, d);
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = queries.ncols();
        let mut out = math::matmul(queries, self.dequant(codes, d).t()); // <q, S*codeword>
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, in_dim: usize) -> Option<usize> {
        Some(self.layout(in_dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::refs;
    use ndarray::{Array2, Axis};

    // Gaussian rows scaled to unit norm -- the input cast(normal) assumes (a unit-norm
    // rotated vector), so each coordinate is ~ N(0, 1/d).
    fn unit_rows(n_v: usize, d: usize, seed: u64) -> Array2<f32> {
        let mut v = math::gaussian(&mut math::seed(seed), (n_v, d));
        let norm = v.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
        math::scale_rows(&mut v, math::reciprocal(norm.view()).view());
        v
    }

    #[test]
    fn plain_round_trips_without_per_vector_rms() {
        // Plain (S=1): the matched grid alone reconstructs the input within lattice error.
        let v = unit_rows(40, 128, 3);
        let cast = CastNormal::new(8, NormalScale::Plain);
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let recon = cast.reconstruct(&model, &refs(&codes), None);
        let err: f32 = (&recon - &v).mapv(|x| x * x).sum();
        let energy: f32 = v.mapv(|x| x * x).sum();
        assert!(err / energy < 0.02, "rel recon error {}", err / energy);
        // Plain stores the constant scale 1.0, not a per-vector RMS.
        let stored = f32::from_le_bytes(codes[0][codes[0].len() - 4..].try_into().unwrap());
        assert!((stored - 1.0).abs() < 1e-6, "Plain scale {stored}");
    }

    #[test]
    fn biased_mse_round_trips() {
        // BiasedMse (S = <x,codeword>/||codeword||^2) is the least-MSE projection onto the grid,
        // so it reconstructs at least as well as Plain.
        let v = unit_rows(40, 128, 5);
        let cast = CastNormal::new(8, NormalScale::BiasedMse);
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let recon = cast.reconstruct(&model, &refs(&codes), None);
        let err: f32 = (&recon - &v).mapv(|x| x * x).sum();
        let energy: f32 = v.mapv(|x| x * x).sum();
        assert!(err / energy < 0.02, "rel recon error {}", err / energy);
    }

    #[test]
    fn code_bytes_matches_emitted() {
        let v = unit_rows(3, 6, 1); // d = 6
        for bits in [1u8, 2, 4, 8] {
            let cast = CastNormal::new(bits, NormalScale::Plain);
            let codes = cast.encode(&cast.fit(v.view(), None), v.view());
            assert_eq!(codes[0].len(), cast.code_bytes(6).unwrap());
            assert_eq!(codes[0].len(), (6 * bits as usize).div_ceil(8) + 4);
        }
    }

    #[test]
    fn unbiased_score_slope_is_one() {
        let v = unit_rows(80, 256, 11);
        let q: Array2<f32> = math::gaussian(&mut math::seed(12), (10, 256));
        let cast = CastNormal::new(4, NormalScale::Unbiased);
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let est = cast.score(&model, q.view(), &refs(&codes), None);
        let exact = q.dot(&v.t());
        let sum_est_exact: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let sum_exact_sq: f32 = exact.iter().map(|t| t * t).sum();
        assert!(
            ((sum_est_exact / sum_exact_sq) - 1.0).abs() < 0.15,
            "slope {}",
            sum_est_exact / sum_exact_sq
        );
    }
}
