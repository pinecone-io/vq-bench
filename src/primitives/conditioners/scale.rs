//! SCALE: applies a fixed affine scale * x + offset to every vector
//! -
//! Model: empty
//! Code for vector x: empty (the constants are config)
//! Apply: x --> scale * x + offset
//! Reconstruct: y --> (y - offset) / scale
//! Score: s --> s / scale - (offset / scale) * sum(q)

use ndarray::{Array2, ArrayView2, Axis};

use crate::{math, Primitive};

pub struct Scale {
    scale: f32,
    offset: f32,
}

impl Scale {
    pub fn new(scale: f32, offset: f32) -> Self {
        debug_assert!(scale != 0.0);
        Self { scale, offset }
    }
}

impl Primitive for Scale {
    fn describe() -> &'static str {
        "apply a fixed affine scaling to every vector"
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        let (scale, offset) = (self.scale, self.offset);
        vectors.mapv_inplace(|x| scale * x + offset);
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let (scale, offset) = (self.scale, self.offset);
        child_recons
            .expect("Scale is not terminal")
            .mapv(|x| (x - offset) / scale)
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let inv = 1.0 / self.scale;
        let mut out = child_scores.expect("Scale is not terminal").mapv(|x| x * inv);
        // the offset term is the same for every candidate, so it's a per-query row offset.
        let sum_q = queries.sum_axis(Axis(1));
        math::offset_rows(&mut out, sum_q.mapv(|sum| -self.offset * inv * sum).view());
        out
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{byte_split, AsQuantizer, CastUint, MinMax, Pipeline, Quantizer};
    use ndarray::array;

    #[test]
    fn round_trip_and_score() {
        let v = array![[1., -2., 3.], [0., 4., -1.]];
        let q = array![[1., 0., -1.], [0.5, 0.5, 0.5]];
        let sc = Scale::new(2.5, -1.0);
        let mut xp = v.clone();
        sc.apply(&[], &mut xp, &[]); // x' = 2.5x - 1
        assert_close(&sc.reconstruct(&[], &[], Some(xp.view())), &v, 1e-4);
        let child = q.dot(&xp.t()); // <q, x'>
        assert_close(
            &sc.score(&[], q.view(), &[], Some(child.view())),
            &q.dot(&v.t()),
            1e-4,
        );
    }

    #[test]
    fn cast_scale_cast_equals_cast4() {
        // cast(2) -> scale(4, 0.5) -> cast(2) reconstructs identically to cast(4): the
        // residual left by the high 2 bits, rescaled to [0,1], supplies the low 2 bits.
        let v = array![[0.05, 0.3, 0.55, 0.95], [0.5, 0.0, 1.0, 0.72]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0.25, -0.5]];
        let nested = AsQuantizer(
            Pipeline::new(
                4,
                vec![
                    Box::new(CastUint::new(2)) as Box<dyn Primitive>,
                    Box::new(Scale::new(4.0, 0.5)),
                    Box::new(CastUint::new(2)),
                ],
            )
            .unwrap(),
        );
        let flat = AsQuantizer(
            Pipeline::new(4, vec![Box::new(CastUint::new(4)) as Box<dyn Primitive>]).unwrap(),
        );

        let (model_nested, model_flat) = (nested.fit(v.view(), None), flat.fit(v.view(), None));
        let (codes_nested, codes_flat) =
            (nested.encode(&model_nested, v.view()), flat.encode(&model_flat, v.view()));
        let (refs_nested, refs_flat) = (refs(&codes_nested), refs(&codes_flat));

        assert_close(
            &nested.reconstruct(&model_nested, &refs_nested),
            &flat.reconstruct(&model_flat, &refs_flat),
            1e-5,
        );
        assert_close(
            &nested.score(&model_nested, q.view(), &refs_nested),
            &flat.score(&model_flat, q.view(), &refs_flat),
            1e-4,
        );
        // 4 bits either way
        assert_eq!(
            byte_split(&model_nested, &codes_nested).1,
            byte_split(&model_flat, &codes_flat).1
        );
    }

    #[test]
    fn minmax_cast4_equals_minmax_cast2_scale_cast2() {
        // The cast(2) -> scale(4, 0.5) -> cast(2) == cast(4) identity, fronted by the same
        // MinMax: MinMax maps each vector into [0,1] identically for both pipelines, so the
        // residual-splitting still rebuilds the exact 4-bit lattice. Realistic un-normalized
        // input (MinMax does the [0,1] mapping the raw cast test assumed).
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0.25, -0.5]];
        let split = AsQuantizer(
            Pipeline::new(
                4,
                vec![
                    Box::new(MinMax::default()) as Box<dyn Primitive>,
                    Box::new(CastUint::new(2)),
                    Box::new(Scale::new(4.0, 0.5)),
                    Box::new(CastUint::new(2)),
                ],
            )
            .unwrap(),
        );
        let flat = AsQuantizer(
            Pipeline::new(
                4,
                vec![
                    Box::new(MinMax::default()) as Box<dyn Primitive>,
                    Box::new(CastUint::new(4)),
                ],
            )
            .unwrap(),
        );

        let (m_split, m_flat) = (split.fit(v.view(), None), flat.fit(v.view(), None));
        let (c_split, c_flat) = (split.encode(&m_split, v.view()), flat.encode(&m_flat, v.view()));
        let (r_split, r_flat) = (refs(&c_split), refs(&c_flat));

        assert_close(
            &split.reconstruct(&m_split, &r_split),
            &flat.reconstruct(&m_flat, &r_flat),
            1e-5,
        );
        assert_close(
            &split.score(&m_split, q.view(), &r_split),
            &flat.score(&m_flat, q.view(), &r_flat),
            1e-4,
        );
        // same 4 bits/dim plus the same MinMax scalars either way
        assert_eq!(
            byte_split(&m_split, &c_split).1,
            byte_split(&m_flat, &c_flat).1
        );
    }
}
