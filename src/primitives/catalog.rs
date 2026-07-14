//! The catalog of in-tree primitives — the source of truth for `vqb show p`.
//! Primitives are grouped by their `src/primitives/` subdirectory, each with a
//! one-line description. Grows as primitives land.

/// Every primitive subdirectory, with its `(name, description)` entries.
pub fn groups() -> &'static [(&'static str, &'static [(&'static str, &'static str)])] {
    &[
        (
            "conditioners",
            &[
                ("minmax", "affine scale each vector into desired target range"),
                ("absmax", "scale each vector into [-1,1] by dividing by max absolute value"),
                ("random_rotate", "apply a random orthogonal transformation to all vectors"),
            ],
        ),
        (
            "rounders",
            &[
                ("cast(uint)", "round [0,1] into 2^b uniform bins, reconstructing to bin centers (1-8 bits)"),
            ],
        ),
    ]
}
