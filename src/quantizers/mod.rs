//! Quantizers: the in-tree catalog the harness runs.

pub mod catalog;

/// Register the in-tree quantizer families: declare each `module => Type` and collect
/// it into the registry. Add a family by writing its module (a type implementing
/// [`Quantizer`](crate::Quantizer)) and adding it here — the only registration edit.
macro_rules! quantizers {
    ($($module:ident => $ty:ident),+ $(,)?) => {
        $(mod $module;)+
        /// Every quantizer family the harness can build.
        pub fn quantizers() -> Vec<catalog::QuantizerSpec> {
            vec![$(catalog::QuantizerSpec::of::<$module::$ty>()),+]
        }
    };
}

quantizers! {
    minmax => MinMax,
    scalar => Scalar,
    eden_mse => EdenMse,
    eden_prod => EdenProd,
    turboquant_mse => TurboquantMse,
    rabitq => RaBitQ,
    e_rabitq => ERaBitQ,
    qjl => Qjl,
    simhash => SimHash,
    turboquant_prod => TurboquantProd,
    itq => Itq,
    itq_asym => ItqAsym,
    pq => Pq,
    opq => Opq,
}
