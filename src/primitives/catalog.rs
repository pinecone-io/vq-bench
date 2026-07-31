//! The catalog of in-tree primitives — what `vqb show p` prints. Each group's rows
//! are collected by its `primitives!` invocation from the trait's `name()`/`describe()`.

/// Every primitive subdirectory, with its `(name, description)` entries.
pub fn groups() -> Vec<(&'static str, Vec<(&'static str, &'static str)>)> {
    vec![
        ("conditioners", super::conditioners::catalog()),
        ("rounders", super::rounders::catalog()),
        ("splitters", super::splitters::catalog()),
    ]
}
