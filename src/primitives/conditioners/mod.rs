primitives! { Primitive:
    minmax => MinMax,
    minmax_dim => MinMaxDim,
    absmax => AbsMax,
    normalize => Normalize,
    center => Center,
    scale => Scale,
    random_rotate => RandomRotate,
    random_hadamard => RandomHadamard,
    pca_rotate => PcaRotate,
    balance_parts => BalanceParts,
    resize => Resize,
    optimize_signs => OptimizeSigns,
    optimize_pq => OptimizePq,
}

// Not a cataloged stage: shared plumbing for the conditioners that store a rotation.
mod rotation_model;
