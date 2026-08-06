//! The single home for turning models and codes to and from bytes.
//!
//! A primitive persists bytes on exactly two surfaces:
//!
//! 1. The MODEL is one typed value, self-describing via `ModelField`: `fit` ends
//!    with `pack_model(value)`; every reader starts with `unpack_model::<T>(model)`.
//!    A new field kind means implementing `ModelField` once.
//! 2. A CODE is per-vector bytes described by one `CodeLayout`: `d` bit-packed
//!    unsigned levels of `bits` bits, then `K` trailing `f32` scalars (either block
//!    may be absent). The schema is config, never stored. That single layout value
//!    drives encoding (`pack`/`pack_scalars`), the fixed size (`byte_len`, which
//!    feeds `code_bytes`), and decoding (`unpack::<K>`) -- so the three can never
//!    disagree, and levels read from the front while scalars read from the tail.
//! 3. FRAMING belongs to the pipeline (`put_len`/`take_len`/`take`) -- primitives
//!    never frame, length-prefix, or pack across vectors.
//!
//! So a new primitive answers two questions -- what does the dataset teach (model
//! fields)? what does each vector carry (bits? scalars?) -- and the byte handling
//! writes itself.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// --- length-prefix framing (pipeline code layout) --------------------------

/// Append `n` as a little-endian `u32`.
pub(crate) fn put_len(buf: &mut Vec<u8>, n: usize) {
    buf.extend_from_slice(&(n as u32).to_le_bytes());
}

/// Read a little-endian `u32` length and advance the cursor.
pub(crate) fn take_len(cur: &mut &[u8]) -> usize {
    let (head, rest) = cur.split_at(4);
    *cur = rest;
    u32::from_le_bytes(head.try_into().unwrap()) as usize
}

/// Split off the next `n` bytes and advance the cursor.
pub(crate) fn take<'a>(cur: &mut &'a [u8], n: usize) -> &'a [u8] {
    let (head, rest) = cur.split_at(n);
    *cur = rest;
    head
}

// --- typed model serialization ---------------------------------------------
//
// A model is a sequence of self-delimiting fields. Each ModelField writes its
// own bytes (variable-size ones prefix their length/shape) and reads them back
// by advancing a cursor, so pack_model((a, b)) / unpack_model::<(A, B)>(..)
// round-trip with no external size info.

/// A value that serializes itself into a model buffer and reads itself back,
/// consuming exactly the bytes it wrote.
pub(crate) trait ModelField: Sized {
    fn write(&self, buf: &mut Vec<u8>);
    fn read(cur: &mut &[u8]) -> Self;
}

/// Serialize a model from one field or a tuple of fields.
pub(crate) fn pack_model<T: ModelField>(fields: T) -> Vec<u8> {
    let mut buf = Vec::new();
    fields.write(&mut buf);
    buf
}

/// Read a model back into the requested field type(s). Panics if bytes remain,
/// catching a model/reader layout mismatch -- including a stale cached model.
pub(crate) fn unpack_model<T: ModelField>(model: &[u8]) -> T {
    let mut cur = model;
    let out = T::read(&mut cur);
    assert!(cur.is_empty(), "unpack_model: {} trailing bytes", cur.len());
    out
}

impl ModelField for u32 {
    fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
    fn read(cur: &mut &[u8]) -> Self {
        u32::from_le_bytes(take(cur, 4).try_into().unwrap())
    }
}

impl ModelField for usize {
    fn write(&self, buf: &mut Vec<u8>) {
        (*self as u32).write(buf); // dims/counts fit in u32
    }
    fn read(cur: &mut &[u8]) -> Self {
        u32::read(cur) as usize
    }
}

impl ModelField for f32 {
    fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.to_le_bytes());
    }
    fn read(cur: &mut &[u8]) -> Self {
        f32::from_le_bytes(take(cur, 4).try_into().unwrap())
    }
}

impl ModelField for Array1<f32> {
    fn write(&self, buf: &mut Vec<u8>) {
        self.len().write(buf);
        write_f32s(buf, self.as_standard_layout().as_slice().unwrap());
    }
    fn read(cur: &mut &[u8]) -> Self {
        let n = usize::read(cur);
        Array1::from(read_f32s(take(cur, n * 4)))
    }
}

impl ModelField for Array2<f32> {
    fn write(&self, buf: &mut Vec<u8>) {
        self.nrows().write(buf);
        self.ncols().write(buf);
        write_f32s(buf, self.as_standard_layout().as_slice().unwrap());
    }
    fn read(cur: &mut &[u8]) -> Self {
        let (rows, cols) = (usize::read(cur), usize::read(cur));
        Array2::from_shape_vec((rows, cols), read_f32s(take(cur, rows * cols * 4))).unwrap()
    }
}

impl<T: ModelField> ModelField for Vec<T> {
    fn write(&self, buf: &mut Vec<u8>) {
        self.len().write(buf);
        for item in self {
            item.write(buf);
        }
    }
    fn read(cur: &mut &[u8]) -> Self {
        (0..usize::read(cur)).map(|_| T::read(cur)).collect()
    }
}

/// Compose fields in order: `write` each, then `read` each back in the same order.
macro_rules! tuple_model_field {
    ($($T:ident $idx:tt),+) => {
        impl<$($T: ModelField),+> ModelField for ($($T,)+) {
            fn write(&self, buf: &mut Vec<u8>) {
                $(self.$idx.write(buf);)+
            }
            fn read(cur: &mut &[u8]) -> Self {
                ($($T::read(cur),)+)
            }
        }
    };
}

tuple_model_field!(A 0, B 1);
tuple_model_field!(A 0, B 1, C 2);
tuple_model_field!(A 0, B 1, C 2, D 3);
tuple_model_field!(A 0, B 1, C 2, D 3, E 4);
tuple_model_field!(A 0, B 1, C 2, D 3, E 4, F 5);

// --- f32 (de)serialization -------------------------------------------------

/// Append `f32`s as little-endian bytes.
fn write_f32s(buf: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

/// Read little-endian `f32`s.
fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// --- per-vector codes: the CodeLayout front door ---------------------------
//
// `CodeLayout` is the one way a primitive builds and reads per-vector codes. It
// declares the code shape once -- `dims` bit-packed levels of `bits` bits (leading),
// then `scalars` trailing `f32` columns -- and that single value drives all three
// surfaces: `byte_len` (feeds `Primitive::code_bytes`), `pack`/`pack_scalars`
// (encode), and `unpack` (decode). They can't drift because they read the same
// fields, and `pack` always writes bits-then-scalars, so the columnar decoders
// (levels from the front, scalars from the tail) never need stored offsets. The
// positional byte engine below is private -- callers only reach it through here.

/// The fixed byte layout of one per-vector code: `dims` bit-packed levels of `bits`
/// bits each (leading), then `scalars` trailing `f32` columns.
pub(crate) struct CodeLayout {
    dims: usize,
    bits: u8,
    scalars: usize,
}

impl CodeLayout {
    /// Widest per-level field the bit-packer supports (one byte): the `bitpack_into`
    /// accumulator assumes fields of at most 8 bits.
    pub(crate) const MAX_BITS: u8 = 8;

    /// An empty layout; add blocks with `bits` / `scalars`.
    pub(crate) fn new() -> Self {
        Self { dims: 0, bits: 0, scalars: 0 }
    }

    /// `dims` bit-packed levels of `bits` bits each (`bits` in `1..=MAX_BITS`).
    pub(crate) fn bits(mut self, dims: usize, bits: u8) -> Self {
        debug_assert!(bits <= Self::MAX_BITS, "field width {bits} exceeds MAX_BITS {}", Self::MAX_BITS);
        self.dims = dims;
        self.bits = bits;
        self
    }

    /// `k` trailing per-vector `f32` columns.
    pub(crate) fn scalars(mut self, k: usize) -> Self {
        self.scalars = k;
        self
    }

    /// Fixed byte length of one code -- the single source for `Primitive::code_bytes`.
    pub(crate) fn byte_len(&self) -> usize {
        (self.dims * self.bits as usize).div_ceil(8) + self.scalars * 4
    }

    /// Fresh codes: bit-pack `levels` (`n x dims`), then append the scalar columns.
    pub(crate) fn pack(&self, levels: ArrayView2<u32>, scalars: &[ArrayView1<f32>]) -> Vec<Vec<u8>> {
        debug_assert_eq!(levels.ncols(), self.dims, "levels width != layout dims");
        debug_assert_eq!(scalars.len(), self.scalars, "scalar count != layout scalars");
        let mut codes = vec![Vec::new(); levels.nrows()];
        pack_bits(&mut codes, levels, self.bits);
        pack_scalars(&mut codes, scalars);
        codes
    }

    /// Fresh codes carrying only the scalar columns (a bits-free layout).
    pub(crate) fn pack_scalars(&self, scalars: &[ArrayView1<f32>]) -> Vec<Vec<u8>> {
        debug_assert_eq!(self.dims, 0, "pack_scalars on a layout with bit levels");
        debug_assert_eq!(scalars.len(), self.scalars, "scalar count != layout scalars");
        let mut codes = vec![Vec::new(); scalars.first().map_or(0, |col| col.len())];
        pack_scalars(&mut codes, scalars);
        codes
    }

    /// Split codes back into the `(n x dims)` level matrix and the `K` scalar columns.
    pub(crate) fn unpack<const K: usize>(&self, codes: &[&[u8]]) -> (Array2<u32>, [Array1<f32>; K]) {
        debug_assert_eq!(K, self.scalars, "unpack::<K>: K != layout scalars");
        (unpack_bits(codes, self.dims, self.bits), unpack_scalars::<K>(codes))
    }
}

// --- per-vector code engine (private; reached only via CodeLayout) ---------
//
// Blocks are appended to each per-vector buffer: bit-packed levels then trailing
// f32 scalars, in that order. Both writers work on the whole batch at once. The
// level count `d`, width `bits`, and scalar count `K` are config (not stored):
// `unpack_bits` reads the leading levels, and the scalars sit at the tail so
// `unpack_scalars` needs no offsets at all.

/// Bit-pack each row of `values` (`bits`-bit unsigned, `1..=8`) and append it to
/// the matching code. Called before `pack_scalars` so levels lead each code.
fn pack_bits(codes: &mut [Vec<u8>], values: ArrayView2<u32>, bits: u8) {
    debug_assert_eq!(codes.len(), values.nrows());
    for (code, row) in codes.iter_mut().zip(values.rows()) {
        bitpack_into(code, row.iter().copied(), bits);
    }
}

/// Append each column's per-vector `f32` to the matching code, in column order --
/// the tail layout `unpack_scalars::<K>` reads back. Called after `pack_bits`.
fn pack_scalars(codes: &mut [Vec<u8>], columns: &[ArrayView1<f32>]) {
    for (i, code) in codes.iter_mut().enumerate() {
        for col in columns {
            code.extend_from_slice(&col[i].to_le_bytes());
        }
    }
}

/// Unpack `n` codes of `d` `bits`-bit levels each into an `(n x d)` level matrix.
fn unpack_bits(codes: &[&[u8]], d: usize, bits: u8) -> Array2<u32> {
    let mut flat = Vec::with_capacity(codes.len() * d);
    for code in codes {
        unpack_bits_into(code, d, bits, &mut flat);
    }
    Array2::from_shape_vec((codes.len(), d), flat).unwrap()
}

/// Unpack each code's `K` trailing per-vector `f32` scalars into `K` `n`-length columns.
fn unpack_scalars<const K: usize>(codes: &[&[u8]]) -> [Array1<f32>; K] {
    core::array::from_fn(|j| {
        codes
            .iter()
            .map(|c| {
                let at = c.len() - (K - j) * 4;
                f32::from_le_bytes(c[at..at + 4].try_into().unwrap())
            })
            .collect()
    })
}

// --- bit-packing engine (sub-byte unsigned codes) --------------------------

/// Append `values`, each in its low `bits` (`1..=8`) bits LSB-first, to `out` --
/// `ceil(n*bits/8)` bytes. A byte-at-a-time accumulator: cost is ~flat in `bits`,
/// no per-bit work. The trailing partial byte is flushed so the next appended
/// block starts byte-aligned.
fn bitpack_into(out: &mut Vec<u8>, values: impl Iterator<Item = u32>, bits: u8) {
    let bits = bits as u32;
    let mask = (1u32 << bits) - 1;
    let (mut acc, mut nbits) = (0u64, 0u32);
    for v in values {
        acc |= ((v & mask) as u64) << nbits; // bits <= 8, nbits < 8 => fits in u64
        nbits += bits;
        while nbits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push(acc as u8); // final partial byte (padding bits are zero)
    }
}

/// Append the `n` values of `bits` bits each packed in `bytes` to `out`. Unpacking
/// one code into a reused buffer lets callers stream over codes without an
/// `(n × d)` intermediate.
fn unpack_bits_into(bytes: &[u8], n: usize, bits: u8, out: &mut Vec<u32>) {
    let bits = bits as u32;
    let mask = (1u64 << bits) - 1;
    let (mut acc, mut nbits, mut bi) = (0u64, 0u32, 0usize);
    for _ in 0..n {
        while nbits < bits {
            acc |= (bytes[bi] as u64) << nbits;
            bi += 1;
            nbits += 8;
        }
        out.push((acc & mask) as u32);
        acc >>= bits;
        nbits -= bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn f32s_round_trip() {
        let xs = [1.0, -2.5, 3.25, 0.0];
        let mut buf = Vec::new();
        write_f32s(&mut buf, &xs);
        assert_eq!(read_f32s(&buf), xs);
    }

    #[test]
    fn bits_round_trip_across_widths() {
        // Every supported width, incl. the non-divisors of 8 (3,5,6,7).
        for bits in 1u8..=8 {
            let max = (1u32 << bits) - 1;
            let values: Vec<u32> = (0..13).map(|i| (i * 7 + 3) % (max + 1)).collect();
            let mut packed = Vec::new();
            bitpack_into(&mut packed, values.iter().copied(), bits);
            assert_eq!(packed.len(), (values.len() * bits as usize).div_ceil(8));
            let mut got = Vec::new();
            unpack_bits_into(&packed, values.len(), bits, &mut got);
            assert_eq!(got, values, "bits={bits}");
        }
    }

    #[test]
    fn codes_matrix_round_trips() {
        let levels = array![[0u32, 1, 2, 3], [3, 2, 1, 0]]; // 2-bit values, d = 4
        let mut codes = vec![Vec::new(); 2];
        pack_bits(&mut codes, levels.view(), 2);
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        assert_eq!(unpack_bits(&refs, 4, 2), levels);
    }

    #[test]
    fn model_scalar_round_trips() {
        assert_eq!(unpack_model::<usize>(&pack_model(1536usize)), 1536);
        assert_eq!(unpack_model::<u32>(&pack_model(7u32)), 7);
        assert_eq!(unpack_model::<f32>(&pack_model(-2.5f32)), -2.5);
    }

    #[test]
    fn model_tuple_and_arrays_round_trip() {
        // The kmeans shape: (dim, centroids) with a non-square, self-describing matrix.
        let centroids = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let m = pack_model((3usize, centroids.clone()));
        let (d, c): (usize, Array2<f32>) = unpack_model(&m);
        assert_eq!(d, 3);
        assert_eq!(c, centroids);

        // Array1 on its own.
        let v: Array1<f32> = array![1.5, -2.5, 0.0];
        assert_eq!(unpack_model::<Array1<f32>>(&pack_model(v.clone())), v);
    }

    #[test]
    fn model_vec_of_matrices_round_trips() {
        // Generalizes: a variable-length list of sub-codebooks (e.g. PQ segments).
        let cbs = vec![array![[1.0f32, 2.0]], array![[3.0, 4.0], [5.0, 6.0]]];
        assert_eq!(unpack_model::<Vec<Array2<f32>>>(&pack_model(cbs.clone())), cbs);
    }

    #[test]
    fn code_levels_and_scalar_round_trip() {
        // Two vectors: d=4 of 2-bit levels, then a trailing per-vector f32.
        let levels = array![[0u32, 1, 2, 3], [3, 2, 1, 0]];
        let scalar = array![1.5f32, -0.25];
        let mut codes = vec![Vec::new(); 2];
        pack_bits(&mut codes, levels.view(), 2);
        pack_scalars(&mut codes, &[scalar.view()]);
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        assert_eq!(unpack_bits(&refs, 4, 2), levels);
        let [scalars] = unpack_scalars::<1>(&refs);
        assert_eq!(scalars, scalar);
    }

    #[test]
    fn scalar_only_code_round_trip() {
        // No levels: two f32 columns per vector (the minmax shape).
        let scale = array![2.0f32, 0.5, -1.0];
        let offset = array![1.0f32, 0.0, 3.5];
        let mut codes = vec![Vec::new(); 3];
        pack_scalars(&mut codes, &[scale.view(), offset.view()]);
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        let [got_scale, got_offset] = unpack_scalars::<2>(&refs);
        assert_eq!(got_scale, scale);
        assert_eq!(got_offset, offset);
    }

    #[test]
    fn layout_byte_len_covers_each_shape() {
        assert_eq!(CodeLayout::new().bits(5, 2).byte_len(), 2); // ceil(10/8)
        assert_eq!(CodeLayout::new().scalars(2).byte_len(), 8); // two f32
        assert_eq!(CodeLayout::new().bits(4, 3).scalars(1).byte_len(), 6); // ceil(12/8)+4
    }

    #[test]
    fn layout_pack_unpack_round_trips_bits_and_scalars() {
        let levels = array![[0u32, 1, 2, 3], [3, 2, 1, 0]]; // d = 4, 2-bit
        let cos = array![1.5f32, -0.25];
        let layout = CodeLayout::new().bits(4, 2).scalars(1);
        let codes = layout.pack(levels.view(), &[cos.view()]);
        // byte_len is the single source: it matches what pack actually emits.
        assert_eq!(codes[0].len(), layout.byte_len());
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        let (got_levels, [got_cos]) = layout.unpack::<1>(&refs);
        assert_eq!(got_levels, levels);
        assert_eq!(got_cos, cos);
    }

    #[test]
    fn layout_pack_scalars_round_trips() {
        let scale = array![2.0f32, 0.5, -1.0];
        let offset = array![1.0f32, 0.0, 3.5];
        let layout = CodeLayout::new().scalars(2);
        let codes = layout.pack_scalars(&[scale.view(), offset.view()]);
        assert_eq!(codes[0].len(), layout.byte_len());
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        let (_, [got_scale, got_offset]) = layout.unpack::<2>(&refs);
        assert_eq!(got_scale, scale);
        assert_eq!(got_offset, offset);
    }
}
