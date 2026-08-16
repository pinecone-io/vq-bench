//! Row-block HDF5 I/O over flat `[f32]` buffers — the whole of this crate's unsafe.
//!
//! hdf5-metno's `read_slice_2d`/`write_slice` are typed in *its* ndarray, and it accepts
//! any of `>=0.15, <=0.17`. That need not be the 0.16 this crate and linfa share: an
//! unlocked resolution puts two incompatible ndarrays in the graph and those calls stop
//! compiling. So row blocks go through `H5Dread`/`H5Dwrite` on plain slices instead,
//! matching `read_raw`/`write_raw` — the only other HDF5 calls in the tree, and Vec-based
//! for the same reason. Reading into a caller-owned buffer also lets a whole streaming
//! pass share one allocation.
//!
//! Both entry points hold `hdf5_sys::LOCK` — the same reentrant mutex hdf5-metno takes
//! around every one of its own calls. The HDF5 C library keeps global state, so skipping it
//! lets a read here corrupt a concurrent `hdf5::` call on another thread, surfacing as a
//! bogus failure from something as basic as `H5Dget_space`.

use std::os::raw::c_void;
use std::ptr;

use anyhow::{bail, ensure, Result};
use hdf5_sys::h5::hsize_t;
use hdf5_sys::LOCK;
use hdf5_sys::h5d::{H5Dget_space, H5Dread, H5Dwrite};
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::{H5Sclose, H5Screate_simple, H5Sselect_hyperslab, H5S_seloper_t};
use hdf5_sys::h5t::H5T_NATIVE_FLOAT;

/// A dataspace id, closed on drop so an early `bail!` cannot leak it.
struct Space(hid_t);

impl Drop for Space {
    fn drop(&mut self) {
        // SAFETY: only ever built from a checked-positive id, and closed exactly once.
        unsafe { H5Sclose(self.0) };
    }
}

/// Rows in a row-major buffer of `len` f32s.
fn rows_of(len: usize, dim: usize) -> Result<usize> {
    ensure!(
        dim > 0 && len.is_multiple_of(dim),
        "{len} f32s is not a whole number of {dim}-column rows"
    );
    Ok(len / dim)
}

/// Whole rows `start..start + rows` of a 2-D dataset, as a selection on the file's
/// dataspace paired with a matching `rows × dim` memory dataspace. The caller holds `LOCK`.
fn row_slab(ds: hid_t, start: usize, rows: usize, dim: usize) -> Result<(Space, Space)> {
    // SAFETY: `ds` is a live dataset id, borrowed from an `hdf5::Dataset`.
    let file = Space(unsafe { H5Dget_space(ds) });
    if file.0 < 0 {
        bail!("reading the dataspace of a {dim}-column dataset");
    }
    let offset = [start as hsize_t, 0];
    let count = [rows as hsize_t, dim as hsize_t];
    // SAFETY: `file.0` is a live 2-D dataspace and both arrays hold one element per
    // dimension, as required. Null stride and block ask for the defaults — contiguous,
    // unit-sized — which is exactly a run of whole rows.
    let rc = unsafe {
        H5Sselect_hyperslab(
            file.0,
            H5S_seloper_t::H5S_SELECT_SET,
            offset.as_ptr(),
            ptr::null(),
            count.as_ptr(),
            ptr::null(),
        )
    };
    if rc < 0 {
        bail!("selecting rows {start}..{} (dataset too short?)", start + rows);
    }
    // SAFETY: rank 2 matches `count`'s length; a null maxdims fixes the extent at `count`.
    let mem = Space(unsafe { H5Screate_simple(2, count.as_ptr(), ptr::null()) });
    if mem.0 < 0 {
        bail!("allocating a {rows}×{dim} memory dataspace");
    }
    Ok((file, mem))
}

/// Read rows `start..start + buf.len() / dim` of `ds` into `buf`, row-major. HDF5 converts
/// the stored type to `f32`, so a dataset written at another precision still reads here.
pub fn read_rows(ds: &hdf5::Dataset, start: usize, dim: usize, buf: &mut [f32]) -> Result<()> {
    let rows = rows_of(buf.len(), dim)?;
    // Declared first so the dataspaces are closed before the lock is released. `ds` came
    // from hdf5-metno, so the library is already initialized.
    let _lock = LOCK.lock();
    let (file, mem) = row_slab(ds.id(), start, rows, dim)?;
    // SAFETY: `buf` holds exactly rows*dim f32s, which is what `mem` describes, so HDF5
    // writes only within it. `H5T_NATIVE_FLOAT` is C `float`, i.e. `f32`, on every
    // platform this builds for — the same assumption `read_raw::<f32>` already makes.
    let rc = unsafe {
        H5Dread(
            ds.id(),
            *H5T_NATIVE_FLOAT,
            mem.0,
            file.0,
            H5P_DEFAULT,
            buf.as_mut_ptr().cast::<c_void>(),
        )
    };
    if rc < 0 {
        bail!("reading rows {start}..{}", start + rows);
    }
    Ok(())
}

/// Write `buf` (row-major) into rows `start..start + buf.len() / dim` of `ds`.
pub fn write_rows(ds: &hdf5::Dataset, start: usize, dim: usize, buf: &[f32]) -> Result<()> {
    let rows = rows_of(buf.len(), dim)?;
    let _lock = LOCK.lock(); // as `read_rows`
    let (file, mem) = row_slab(ds.id(), start, rows, dim)?;
    // SAFETY: as `read_rows`, except HDF5 only reads `buf`.
    let rc = unsafe {
        H5Dwrite(
            ds.id(),
            *H5T_NATIVE_FLOAT,
            mem.0,
            file.0,
            H5P_DEFAULT,
            buf.as_ptr().cast::<c_void>(),
        )
    };
    if rc < 0 {
        bail!("writing rows {start}..{}", start + rows);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a block of rows, and check a partial read lands at the right offset.
    #[test]
    fn round_trips_row_blocks() {
        let dir = std::env::temp_dir().join("vqb-h5-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rows.hdf5");
        let _ = std::fs::remove_file(&path);

        let (rows, dim) = (7usize, 3usize);
        let all: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        {
            let f = hdf5::File::create(&path).unwrap();
            let ds = f.new_dataset::<f32>().shape([rows, dim]).create("x").unwrap();
            // Written in two blocks, so the offset arithmetic is exercised on write too.
            write_rows(&ds, 0, dim, &all[..4 * dim]).unwrap();
            write_rows(&ds, 4, dim, &all[4 * dim..]).unwrap();
        }
        let f = hdf5::File::open(&path).unwrap();
        let ds = f.dataset("x").unwrap();

        let mut got = vec![0f32; rows * dim];
        read_rows(&ds, 0, dim, &mut got).unwrap();
        assert_eq!(got, all);

        // Rows 2..5 only.
        let mut mid = vec![0f32; 3 * dim];
        read_rows(&ds, 2, dim, &mut mid).unwrap();
        assert_eq!(mid, all[2 * dim..5 * dim]);

        // Past the end is an error, not a silent short read.
        let mut over = vec![0f32; 3 * dim];
        assert!(read_rows(&ds, 5, dim, &mut over).is_err());
        // A buffer that isn't a whole number of rows is rejected before any HDF5 call.
        let mut ragged = vec![0f32; dim + 1];
        assert!(read_rows(&ds, 0, dim, &mut ragged).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// These calls must serialise against hdf5-metno's own, which run under the same
    /// `LOCK`. Without it, HDF5's global state is raced and unrelated calls start failing
    /// — `H5Dget_space` on a live dataset was the observed symptom. Interleaving both
    /// kinds of access across threads reproduces it in a single test.
    #[test]
    fn serialises_against_the_safe_api() {
        let dir = std::env::temp_dir().join("vqb-h5-race-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("race.hdf5");
        let _ = std::fs::remove_file(&path);

        let (rows, dim) = (64usize, 8usize);
        let all: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
        {
            let f = hdf5::File::create(&path).unwrap();
            let ds = f.new_dataset::<f32>().shape([rows, dim]).create("x").unwrap();
            write_rows(&ds, 0, dim, &all).unwrap();
        }

        std::thread::scope(|s| {
            for t in 0..8 {
                let (path, all) = (&path, &all);
                s.spawn(move || {
                    for _ in 0..20 {
                        let f = hdf5::File::open(path).unwrap();
                        let ds = f.dataset("x").unwrap();
                        if t % 2 == 0 {
                            // The raw path.
                            let mut buf = vec![0f32; 4 * dim];
                            read_rows(&ds, t * 4, dim, &mut buf).unwrap();
                            assert_eq!(buf, all[t * 4 * dim..(t * 4 + 4) * dim]);
                        } else {
                            // hdf5-metno's own path, which takes the same lock.
                            let got: Vec<f32> = ds.read_raw().unwrap();
                            assert_eq!(got, *all);
                        }
                    }
                });
            }
        });
        std::fs::remove_file(&path).ok();
    }
}
