//! CAST(ANGULAR, b): round to b-bit grid point with minimum angle
//! -
//! Model: input dim d
//! Code for vector x: b-bit level per coordinate plus code_cos = <grid/||grid||, x>
//! Apply: x --> x - grid/||grid||
//! Reconstruct: y --> grid/||grid|| + y
//! Score: s --> <q, grid/||grid||> / code_cos + s

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

pub struct CastAngular {
    bits: u8,
}

impl CastAngular {
    /// `b`-bit angular cast; the valid range (`1..=CodeLayout::MAX_BITS`) is enforced
    /// by the quantizer builder.
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }

    /// `shift = (2^b - 1)/2`: codes are `u = grid + shift`; also the extreme grid level.
    fn shift(&self) -> f32 {
        ((1u32 << self.bits) - 1) as f32 / 2.0
    }

    /// The code layout: `d` levels of `self.bits` bits, then the `code_cos` scalar.
    fn layout(&self, d: usize) -> CodeLayout {
        CodeLayout::new().bits(d, self.bits).scalars(1)
    }

    /// Split each code into its packed levels (`n x d`) and `code_cos` (`n`).
    fn split(&self, codes: &[&[u8]], d: usize) -> (Array2<u32>, Array1<f32>) {
        let (levels, [code_cos]) = self.layout(d).unpack::<1>(codes);
        (levels, code_cos)
    }

    /// The optimal-scale sweep: trace `t*x`, rounding coordinates up as `t` grows and
    /// keeping the best `cos(grid, x)`. Returns the unsigned codes and `code_cos` (best cosine).
    fn quantize(&self, x: ArrayView1<f32>) -> (Vec<u32>, f32) {
        let shift = self.shift();
        let d = x.len();

        // Base state (t -> 0+): each coordinate at its nearest +-0.5 (zeros fold to +0.5).
        let mut grid: Vec<f32> = x.iter().map(|&xi| if xi >= 0.0 { 0.5 } else { -0.5 }).collect();
        let mut dot: f32 = x.iter().zip(&grid).map(|(&xi, &gi)| xi * gi).sum();
        let mut norm_sq = 0.25 * d as f32;

        // Coord crosses integer step (promoting |grid_i| to step+0.5) at t = step/|x_i|;
        // sort the merged crossings. max_steps = 2^(b-1) - 1 per coord (b = 1 => none).
        let max_steps = (shift - 0.5) as u32;
        let mut events: Vec<(f32, usize)> = Vec::with_capacity(d * max_steps as usize);
        for (coord, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let inv = 1.0 / xi.abs();
            for step in 1..=max_steps {
                events.push((step as f32 * inv, coord));
            }
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut best_cos = dot / norm_sq.sqrt();
        let mut best_grid = grid.clone();
        for (e, &(t_cross, coord)) in events.iter().enumerate() {
            let delta = if x[coord] >= 0.0 { 1.0 } else { -1.0 };
            let (old, new) = (grid[coord], grid[coord] + delta);
            dot += delta * x[coord];
            norm_sq += new * new - old * old;
            grid[coord] = new;
            // Evaluate once per distinct t; tied intermediate states aren't roundings of t*x.
            if e + 1 == events.len() || events[e + 1].0 > t_cross {
                let cos = dot / norm_sq.sqrt();
                if cos > best_cos {
                    best_cos = cos;
                    best_grid.copy_from_slice(&grid);
                }
            }
        }

        let unsigned = best_grid.iter().map(|&v| (v + shift).round() as u32).collect();
        (unsigned, best_cos)
    }
}

impl Primitive for CastAngular {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (n_v, d) = (vectors.nrows(), vectors.ncols());
        let mut levels = Array2::<u32>::zeros((n_v, d));
        let mut code_cos = Array1::<f32>::zeros(n_v);
        for (i, row) in vectors.rows().into_iter().enumerate() {
            let (unsigned, cos) = self.quantize(row);
            levels.row_mut(i).assign(&Array1::from(unsigned));
            code_cos[i] = cos;
        }
        self.layout(d).pack(levels.view(), &[code_cos.view()])
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let recon = self.reconstruct(model, codes, None);
        *vectors -= &recon; // residual
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let (levels, _) = self.split(codes, d);
        let mut grid = levels.mapv(|u| u as f32 - self.shift());
        let norms = grid.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f32::sqrt);
        math::scale_rows(&mut grid, math::reciprocal(norms.view()).view());
        if let Some(child) = child_recons {
            grid += &child;
        }
        grid
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = queries.ncols();
        let (levels, code_cos) = self.split(codes, d);
        let shift = self.shift();
        let grid = levels.mapv(|u| u as f32 - shift);
        let grid_norms = grid.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f32::sqrt);
        // <q, grid> = sum_j q_j*u_j - shift*sum(q), then / (||grid||*code_cos) per candidate.
        let sum_q = queries.sum_axis(Axis(1));
        let mut out = math::matmul(queries, levels.mapv(|u| u as f32).t());
        math::offset_rows(&mut out, sum_q.mapv(|sum| -shift * sum).view());
        let denom = &grid_norms * &code_cos;
        // denom can be tiny or negative, so guard on magnitude, not math::reciprocal.
        let inv = denom.mapv(|v| if v.abs() > f32::EPSILON { 1.0 / v } else { 0.0 });
        math::scale_cols(&mut out, inv.view());
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
    use crate::util::testing::{assert_close, refs};
    use ndarray::array;

    #[test]
    fn deterministic_codes() {
        let x = array![[0.6, -0.2, 0.0, -0.8], [0.1, 0.5, -0.5, 0.7]];
        let cast = CastAngular::new(4);
        let model = cast.fit(x.view(), None);
        assert_eq!(cast.encode(&model, x.view()), cast.encode(&model, x.view()));
    }

    #[test]
    fn exact_when_input_on_grid() {
        // x proportional to a b=2 grid point in {+-0.5, +-1.5}: the sweep recovers it
        // with code_cos = 1, so reconstruct = unit(x) and score = q*x^T exactly.
        let grid = array![[1.5, -0.5, 0.5, -1.5]];
        let x = grid.mapv(|v: f32| v / 5f32.sqrt()); // ||grid|| = sqrt(5)
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let cast = CastAngular::new(2);
        let model = cast.fit(x.view(), None);
        let codes = cast.encode(&model, x.view());
        let r = refs(&codes);
        assert_close(&cast.reconstruct(&model, &r, None), &x, 1e-5);
        assert_close(&cast.score(&model, q.view(), &r, None), &q.dot(&x.t()), 1e-4);
    }

    #[test]
    fn code_cos_consistent_with_zero_coord() {
        // With an exact-zero coordinate, the stored code_cos must equal <grid/||grid||, x>
        // recomputed from the decoded levels -- the zero coord counts toward ||grid|| in both.
        let x = array![[0.6, 0.0, 0.8]]; // unit, one zero coord
        let cast = CastAngular::new(2);
        let model = cast.fit(x.view(), None);
        let codes = cast.encode(&model, x.view());
        let r = refs(&codes);
        let (levels, code_cos) = cast.split(&r, 3);
        let grid = levels.row(0).mapv(|u| u as f32 - cast.shift());
        let recomputed = grid.dot(&x.row(0)) / grid.dot(&grid).sqrt();
        assert!((recomputed - code_cos[0]).abs() < 1e-5, "{recomputed} vs {}", code_cos[0]);
    }

    #[test]
    fn b1_is_sign() {
        // b=1: codes are sign bits (x_i >= 0 => 1 => decodes to +0.5), zeros fold to +0.5.
        let x = array![[0.6, -0.2, 0.0, -0.8]];
        let cast = CastAngular::new(1);
        let model = cast.fit(x.view(), None);
        let codes = cast.encode(&model, x.view());
        let r = refs(&codes);
        let (levels, _) = cast.split(&r, 4);
        assert_eq!(levels.row(0).to_vec(), vec![1u32, 0, 1, 0]);
    }

    #[test]
    fn code_bytes_matches_emitted() {
        let o = array![[0.3, 0.4, 0.5, 0.6, 0.7]]; // d = 5
        for bits in [1u8, 2, 4, 8] {
            let cast = CastAngular::new(bits);
            let codes = cast.encode(&cast.fit(o.view(), None), o.view());
            assert_eq!(codes[0].len(), cast.code_bytes(5).unwrap());
            assert_eq!(codes[0].len(), (5 * bits as usize).div_ceil(8) + 4);
        }
    }

    #[test]
    fn composes_after_normalize() {
        use crate::{AsQuantizer, Pipeline, Quantizer};
        let v = array![[0., 1., 2., 3.], [4., -6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(
            Pipeline::new(
                4,
                vec![
                    Box::new(crate::Normalize) as Box<dyn Primitive>,
                    Box::new(CastAngular::new(8)),
                ],
            )
            .unwrap(),
        );
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        let recon = codec.reconstruct(&model, &r);
        // 8-bit angular grid recovers each direction tightly (cos ~ 1 with the input).
        for (rr, vr) in recon.rows().into_iter().zip(v.rows()) {
            let cos = rr.dot(&vr) / (rr.dot(&rr).sqrt() * vr.dot(&vr).sqrt());
            assert!(cos > 0.99, "cos {cos}");
        }
        assert_eq!(codec.score(&model, q.view(), &r).dim(), (2, 3));
    }
}
