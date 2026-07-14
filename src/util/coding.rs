//! Byte/bit coding: length-prefix framing, bit-packing, and `f32` (de)serialization.
//! The single home for turning models and codes to and from bytes.

use ndarray::{Array1, Array2};

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

// --- f32 (de)serialization -------------------------------------------------

/// Append `f32`s as little-endian bytes.
pub(crate) fn write_f32s(buf: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

/// Serialize `f32`s into a fresh little-endian byte buffer (inverse of `read_f32s`).
pub(crate) fn f32s_to_bytes(xs: impl IntoIterator<Item = f32>) -> Vec<u8> {
    xs.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// Read little-endian `f32`s.
pub(crate) fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Pack `K` per-vector `f32` fields (each an `n`-length column) into one code per
/// vector: code `i` holds `[fields[0][i], …, fields[K-1][i]]` as LE `f32`s.
pub(crate) fn pack_f32_fields<const K: usize>(fields: [&Array1<f32>; K]) -> Vec<Vec<u8>> {
    let n = fields.first().map_or(0, |f| f.len());
    (0..n)
        .map(|i| {
            let mut buf = Vec::with_capacity(K * 4);
            write_f32s(&mut buf, &fields.map(|f| f[i]));
            buf
        })
        .collect()
}

/// Unpack each code's `K` `f32` fields back into `K` per-vector columns —
/// `let [scale, offset] = unpack_f32_fields(codes);`.
pub(crate) fn unpack_f32_fields<const K: usize>(codes: &[&[u8]]) -> [Array1<f32>; K] {
    let rows: Vec<Vec<f32>> = codes.iter().map(|c| read_f32s(&c[..K * 4])).collect();
    core::array::from_fn(|j| rows.iter().map(|r| r[j]).collect())
}

// --- bit-packing (sub-byte unsigned codes) ---------------------------------

/// Unpack `n` codes of `d` values each into an `(n × d)` level matrix.
pub(crate) fn unpack_codes(codes: &[&[u8]], d: usize, bits: u8) -> Array2<u32> {
    let mut flat = Vec::with_capacity(codes.len() * d);
    for code in codes {
        unpack_bits_into(code, d, bits, &mut flat);
    }
    Array2::from_shape_vec((codes.len(), d), flat).unwrap()
}

/// Pack each value's low `bits` (`1..=8`) bits, LSB-first, into `ceil(n*bits/8)`
/// bytes. A byte-at-a-time accumulator: cost is ~flat in `bits`, no per-bit work.
pub(crate) fn pack_bits(values: &[u32], bits: u8) -> Vec<u8> {
    let bits = bits as u32;
    let mask = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity((values.len() * bits as usize).div_ceil(8));
    let (mut acc, mut nbits) = (0u64, 0u32);
    for &v in values {
        acc |= ((v & mask) as u64) << nbits; // bits ≤ 8, nbits < 8 ⇒ fits in u64
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
    out
}

/// Append the `n` values of `bits` bits each packed in `bytes` to `out`.
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
            let packed = pack_bits(&values, bits);
            assert_eq!(packed.len(), (values.len() * bits as usize).div_ceil(8));
            let mut got = Vec::new();
            unpack_bits_into(&packed, values.len(), bits, &mut got);
            assert_eq!(got, values, "bits={bits}");
        }
    }

    #[test]
    fn codes_matrix_round_trips() {
        let levels = array![[0u32, 1, 2, 3], [3, 2, 1, 0]]; // 2-bit values, d = 4
        let codes: Vec<Vec<u8>> = levels
            .rows()
            .into_iter()
            .map(|r| pack_bits(r.as_slice().unwrap(), 2))
            .collect();
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        assert_eq!(unpack_codes(&refs, 4, 2), levels);
    }

    #[test]
    fn f32_fields_round_trip() {
        let scale = array![2.0, 0.5, -1.0];
        let offset = array![1.0, 0.0, 3.5];
        let codes = pack_f32_fields([&scale, &offset]);
        let refs: Vec<&[u8]> = codes.iter().map(Vec::as_slice).collect();
        let [s, o] = unpack_f32_fields(&refs);
        assert_eq!((s, o), (scale, offset));
    }
}
