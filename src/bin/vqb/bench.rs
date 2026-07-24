//! Metrics. All score metrics operate on per-query `(true, approx)` score pairs
//! over each query's candidates; recon metrics on sampled reconstructions.
//! Single-stage: the candidates *are* the true top-k, so true ranks come from `true_scores`.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Effective bits per dimension: `(model + code bytes)·8 / (n·d)`.
pub fn bits_per_dim(bytes: usize, n: usize, dim: usize) -> f64 {
    let cells = n * dim;
    if cells == 0 {
        0.0
    } else {
        (bytes * 8) as f64 / cells as f64
    }
}

/// Pool positions ordered by descending score, ties in random order. Without randomization, 
/// the pool is laid out in true-score order, so tiebreaking would reveal true ranking for free.
fn ranks_desc(scores: &[f32], rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.shuffle(rng);
    // The sort is stable, so ties keep the shuffled order.
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Tiebreak rng for query `qi`: a distinct ChaCha stream per query keeps a
/// query's ranking identical however many `k`s or metrics are computed together.
fn query_rng(seed: u64, qi: usize) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(qi as u64);
    rng
}

/// recall@k: overlap of the approx top-k with the true top-k, averaged over queries.
pub fn recalls(
    true_scores: &[Vec<f32>],
    approx_scores: &[Vec<f32>],
    ks: &[usize],
    seed: u64,
) -> BTreeMap<usize, f64> {
    let mut out = BTreeMap::new();
    for &k in ks {
        let (mut hits, mut denom) = (0usize, 0usize);
        for (qi, (ts, as_)) in true_scores.iter().zip(approx_scores).enumerate() {
            let mut rng = query_rng(seed, qi);
            let tr = ranks_desc(ts, &mut rng);
            let ar = ranks_desc(as_, &mut rng);
            let kk = k.min(tr.len());
            let want: std::collections::HashSet<usize> = tr[..kk].iter().copied().collect();
            hits += ar[..k.min(ar.len())]
                .iter()
                .filter(|i| want.contains(i))
                .count();
            denom += kk;
        }
        out.insert(k, hits as f64 / denom.max(1) as f64);
    }
    out
}

/// SOS@k: summed true score of the approx top-k over that of the true top-k.
pub fn sos(
    true_scores: &[Vec<f32>],
    approx_scores: &[Vec<f32>],
    ks: &[usize],
    seed: u64,
) -> BTreeMap<usize, f64> {
    let mut out = BTreeMap::new();
    for &k in ks {
        let (mut numer, mut denom) = (0.0f64, 0.0f64);
        for (qi, (ts, as_)) in true_scores.iter().zip(approx_scores).enumerate() {
            let mut rng = query_rng(seed, qi);
            let tr = ranks_desc(ts, &mut rng);
            let ar = ranks_desc(as_, &mut rng);
            numer += ar[..k.min(ar.len())]
                .iter()
                .map(|&p| ts[p] as f64)
                .sum::<f64>();
            denom += tr[..k.min(tr.len())]
                .iter()
                .map(|&p| ts[p] as f64)
                .sum::<f64>();
        }
        out.insert(
            k,
            if denom != 0.0 {
                numer / denom
            } else {
                f64::NAN
            },
        );
    }
    out
}

/// exp-SOS@k: like [`sos`], but each summed score `s` is replaced by `exp(s / tau)`,
/// keyed by temperature (outer) then k (inner). `exp` is monotonic for `tau > 0`, so the
/// approx/true top-k sets match SOS's — only the summed values change. vq-bench data is
/// unit-normalized (scores in `[-1, 1]`), so `exp(s/tau)` is finite for every swept `tau`
/// and needs no max-shift. (A per-query shift would skew the cross-query micro-average;
/// only a global shift is exact, and it is unnecessary given the bound.)
pub fn exp_sos(
    true_scores: &[Vec<f32>],
    approx_scores: &[Vec<f32>],
    ks: &[usize],
    temps: &[f64],
    seed: u64,
) -> BTreeMap<String, BTreeMap<usize, f64>> {
    // Ranks depend on neither k nor tau, so resolve each query's ordering once.
    let ranked: Vec<(&Vec<f32>, Vec<usize>, Vec<usize>)> = true_scores
        .iter()
        .zip(approx_scores)
        .enumerate()
        .map(|(qi, (ts, as_))| {
            let mut rng = query_rng(seed, qi);
            let tr = ranks_desc(ts, &mut rng);
            let ar = ranks_desc(as_, &mut rng);
            (ts, tr, ar)
        })
        .collect();
    let mut out = BTreeMap::new();
    for &t in temps {
        let mut per_k = BTreeMap::new();
        for &k in ks {
            let (mut numer, mut denom) = (0.0f64, 0.0f64);
            for (ts, tr, ar) in &ranked {
                numer += ar[..k.min(ar.len())]
                    .iter()
                    .map(|&p| (ts[p] as f64 / t).exp())
                    .sum::<f64>();
                denom += tr[..k.min(tr.len())]
                    .iter()
                    .map(|&p| (ts[p] as f64 / t).exp())
                    .sum::<f64>();
            }
            per_k.insert(k, if denom != 0.0 { numer / denom } else { f64::NAN });
        }
        out.insert(temp_key(t), per_k);
    }
    out
}

/// Mean squared and mean signed error of the estimated scores, over all pairs.
pub fn score_mse_bias(true_scores: &[Vec<f32>], approx_scores: &[Vec<f32>]) -> (f64, f64) {
    let (mut sse, mut sum_err, mut n) = (0.0f64, 0.0f64, 0u64);
    for (ts, as_) in true_scores.iter().zip(approx_scores) {
        for (&t, &a) in ts.iter().zip(as_) {
            let e = t as f64 - a as f64;
            sse += e * e;
            sum_err += e;
            n += 1;
        }
    }
    let c = n.max(1) as f64;
    (sse / c, sum_err / c)
}

/// Mean reconstruction error `E_j ‖x_j − x̃_j‖²`.
pub fn recon_mse(references: &[Vec<f32>], recons: &[Vec<f32>]) -> f64 {
    if references.is_empty() {
        return 0.0;
    }
    let mut sse = 0.0f64;
    for (x, xh) in references.iter().zip(recons) {
        sse += x
            .iter()
            .zip(xh)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>();
    }
    sse / references.len() as f64
}

/// Squared norm of the mean residual `‖(1/n)Σ_j (x_j − x̃_j)‖²`.
pub fn recon_bias(references: &[Vec<f32>], recons: &[Vec<f32>]) -> f64 {
    if references.is_empty() {
        return 0.0;
    }
    let dim = references[0].len();
    let mut sum = vec![0.0f64; dim];
    for (x, xh) in references.iter().zip(recons) {
        for (d, s) in sum.iter_mut().enumerate() {
            *s += x[d] as f64 - xh[d] as f64;
        }
    }
    let n = references.len() as f64;
    sum.iter().map(|&s| (s / n).powi(2)).sum()
}

/// Stable log-sum-exp.
fn lse(z: &[f64]) -> f64 {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + z.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

/// Softmax KL `D(p‖p̂)` and TV `½Σ|p−p̂|` per query at inverse-temperature `β`,
/// averaged over queries. `p ∝ exp(β·true)`, `p̂ ∝ exp(β·approx)`.
fn softmax_pair(ts: &[f32], as_: &[f32], beta: f64) -> (f64, f64) {
    let za: Vec<f64> = ts.iter().map(|&s| s as f64 * beta).collect();
    let zb: Vec<f64> = as_.iter().map(|&s| s as f64 * beta).collect();
    let (la, lb) = (lse(&za), lse(&zb));
    let (mut kl, mut tv) = (0.0f64, 0.0f64);
    for (&a, &b) in za.iter().zip(&zb) {
        let (p, ph) = ((a - la).exp(), (b - lb).exp());
        if p > 0.0 {
            kl += p * ((a - la) - (b - lb));
        }
        tv += (p - ph).abs();
    }
    (kl.max(0.0), 0.5 * tv)
}

/// Trim a temperature to a clean map key (`1.0` → `"1"`, `0.5` → `"0.5"`).
fn temp_key(t: f64) -> String {
    format!("{t}")
}

/// Softmax KL and TV, each keyed by temperature.
pub fn softmax_kl_tv(
    true_scores: &[Vec<f32>],
    approx_scores: &[Vec<f32>],
    temps: &[f64],
) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
    let (mut kl_out, mut tv_out) = (BTreeMap::new(), BTreeMap::new());
    let nq = true_scores.len().max(1) as f64;
    for &t in temps {
        let beta = 1.0 / t;
        let (mut kl, mut tv) = (0.0f64, 0.0f64);
        for (ts, as_) in true_scores.iter().zip(approx_scores) {
            let (k, v) = softmax_pair(ts, as_, beta);
            kl += k;
            tv += v;
        }
        kl_out.insert(temp_key(t), kl / nq);
        tv_out.insert(temp_key(t), tv / nq);
    }
    (kl_out, tv_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_per_dim_formula() {
        // 12 bytes over 4 vectors × 3 dims = 96 bits / 12 cells = 8 bits/dim.
        assert_eq!(bits_per_dim(12, 4, 3), 8.0);
        assert_eq!(bits_per_dim(0, 0, 0), 0.0);
    }

    #[test]
    fn perfect_scores_give_full_recall_and_sos() {
        let truth = vec![vec![3.0, 2.0, 1.0], vec![1.0, 5.0, 2.0]];
        let recalls = recalls(&truth, &truth, &[1, 2, 3], 1);
        assert_eq!(recalls[&1], 1.0);
        assert_eq!(recalls[&3], 1.0);
        let sos = sos(&truth, &truth, &[1, 2], 1);
        assert!((sos[&1] - 1.0).abs() < 1e-12);
        assert!((sos[&2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exp_sos_perfect_is_one_and_keys_are_temp_then_k() {
        let truth = vec![vec![3.0, 2.0, 1.0], vec![1.0, 5.0, 2.0]];
        // Perfect approx → ratio 1 at every (temperature, k); keys are temp→k.
        let e = exp_sos(&truth, &truth, &[1, 2], &[0.5, 1.0], 1);
        assert_eq!(e.keys().cloned().collect::<Vec<_>>(), vec!["0.5", "1"]);
        for t in ["0.5", "1"] {
            assert_eq!(e[t].keys().copied().collect::<Vec<_>>(), vec![1, 2]);
            assert!((e[t][&1] - 1.0).abs() < 1e-12);
            assert!((e[t][&2] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn exp_sos_imperfect_is_in_unit_interval() {
        // Reversed approx rankings → the approx top-k picks weaker true scores, so the
        // exp-transformed ratio sits strictly inside (0, 1].
        let truth = vec![vec![3.0, 2.0, 1.0], vec![1.0, 5.0, 2.0]];
        let approx = vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]];
        let v = exp_sos(&truth, &approx, &[1], &[1.0], 1)["1"][&1];
        assert!(v > 0.0 && v <= 1.0, "exp-SOS@1 out of range: {v}");
    }

    #[test]
    fn recall_counts_overlap() {
        // true top-1 is pos 0 (score 3); approx top-1 is pos 2 → miss at k=1, hit at k=2.
        let truth = vec![vec![3.0, 2.0, 1.0]];
        let approx = vec![vec![0.0, 1.0, 2.0]];
        let r = recalls(&truth, &approx, &[1, 2], 1);
        assert_eq!(r[&1], 0.0);
        assert_eq!(r[&2], 0.5); // {pos2} ∩ {pos0,pos1} = ∅? top2 approx={2,1}, true={0,1} → {1}; 1/2
    }

    #[test]
    fn constant_scores_dont_leak_true_ranking() {
        // Pool in true-score order (as the harness lays it out), constant approx
        // scores: recall must sit near chance (k/L), not 1.0, and be seed-stable.
        let l = 100;
        let truth = vec![(0..l).map(|i| (l - i) as f32).collect::<Vec<f32>>(); 50];
        let approx = vec![vec![0.0f32; l]; 50];
        let r = recalls(&truth, &approx, &[10], 1);
        assert!(r[&10] < 0.5, "tie order leaked ground truth: {}", r[&10]);
        assert_eq!(r, recalls(&truth, &approx, &[10], 1));
        let s = sos(&truth, &approx, &[10], 1);
        assert!(s[&10] < 0.9, "sos leaked ground truth: {}", s[&10]);
    }

    #[test]
    fn score_error_and_recon() {
        let truth = vec![vec![1.0, 2.0]];
        let approx = vec![vec![0.0, 2.0]];
        let (mse, bias) = score_mse_bias(&truth, &approx);
        assert!((mse - 0.5).abs() < 1e-12); // (1² + 0²)/2
        assert!((bias - 0.5).abs() < 1e-12); // (1 + 0)/2
        let refs = vec![vec![1.0, 1.0]];
        let rec = vec![vec![1.0, 0.0]];
        assert!((recon_mse(&refs, &rec) - 1.0).abs() < 1e-12); // ‖(0,1)‖² = 1
        assert!((recon_bias(&refs, &rec) - 1.0).abs() < 1e-12); // ‖mean residual (0,1)‖² = 1
    }

    #[test]
    fn identical_softmax_is_zero_divergence() {
        let truth = vec![vec![1.0, 2.0, 3.0]];
        let (kl, tv) = softmax_kl_tv(&truth, &truth, &[1.0, 0.5]);
        assert!(kl["1"] < 1e-12 && tv["1"] < 1e-12);
        assert!(kl["0.5"] < 1e-12);
    }
}
