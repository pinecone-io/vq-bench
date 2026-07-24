//! The catalog of in-tree primitives — the source of truth for `vqb show p`.
//! Primitives are grouped by their `src/primitives/` subdirectory, each with a
//! one-line description. Grows as primitives land.

/// Every primitive subdirectory, with its `(name, description)` entries.
pub fn groups() -> &'static [(&'static str, &'static [(&'static str, &'static str)])] {
    &[
        (
            "conditioners",
            &[
                ("MinMax", "affine scale each vector into desired target range"),
                ("MinMaxDim", "affine scale each dimension into the target range, calibrated over the fit set"),
                ("AbsMax", "scale each vector into [-1,1] by dividing by max absolute value"),
                ("Normalize", "scale each vector to unit L2 norm"),
                ("Center", "subtract the mean over the fit set from every vector"),
                ("Scale", "apply a fixed affine scaling to every vector"),
                ("RandomRotate", "apply a random orthogonal transformation to all vectors"),
                ("RandomHadamard", "fast near-orthogonal random rotation via the randomized Hadamard transform"),
                ("OptimizeSigns", "learn an orthogonal rotation minimizing sign-quantization error"),
            ],
        ),
        (
            "rounders",
            &[
                ("CastUint", "round [0,1] into 2^b uniform bins, reconstructing to bin centers"),
                ("CastNormal", "round unit vector with b-bit Lloyd-Max normal codebook"),
                ("CastAngular", "round unit vector to b-bit grid point of minimum angle"),
                ("CastSign", "one sign bit per coordinate; asymmetric score <q, sign(x)>"),
                ("CastHamming", "one sign bit per coordinate; SimHash angle estimate |q| cos(pi hamming/d)"),
            ],
        ),
    ]
}
