//! Quantizers: the in-tree catalog the harness runs.

pub mod catalog;
mod rotation;

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
    qjl => Qjl,
    itq => Itq,
    itq_asym => ItqAsym,
    simhash => SimHash,
    eden_mse => EdenMse,
    eden_prod => EdenProd,
    turboquant_mse => TurboquantMse,
    turboquant_prod => TurboquantProd,
    rabitq => RaBitQ,
    e_rabitq => ERaBitQ,
    pq => Pq,
    opq => Opq,
    opq_p => OpqP,
}
