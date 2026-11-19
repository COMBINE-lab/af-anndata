//! Reads an alevin-fry MatrixMarket file into `X` of a new h5ad file.
//!
//! anndata-rs `MMReader` keeps a COO matrix (24 B per entry), an unsorted CSR
//! (16 B) and a sorted CSR (16 B) at the same time, so it needs 56 B per entry.
//! This reader needs 12 B per entry: pass 1 counts the entries of each row,
//! then one exact allocation holds the column indices (in the type that
//! anndata-rs writes) and the values, and pass 2 puts each entry in its row.
//! X is written with the same datasets, types, chunks and attributes as
//! anndata-rs `CsrMatrix::write`, without the copy that it makes of `indices`.
use anndata::backend::{get_default_write_config, AttributeOp, BackendData, GroupOp, Value};
use anndata::{AnnData, Backend};
use anndata_hdf5::H5;
use anyhow::{anyhow, bail, ensure, Context, Result};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;

/// The same gzip detection as anndata-rs `MMReader`.
fn open(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = || File::open(path).with_context(|| format!("cannot open {}", path.display()));
    Ok(if MultiGzDecoder::new(file()?).header().is_some() {
        Box::new(BufReader::with_capacity(
            1 << 20,
            MultiGzDecoder::new(file()?),
        ))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file()?))
    })
}

#[derive(PartialEq)]
struct Header {
    nrows: usize,
    ncols: usize,
    /// The number of entry lines.
    entries: usize,
    integer: bool,
    /// Each entry `(i, j)` with `i != j` also gives the entry `(j, i)`.
    symmetric: bool,
}

/// Reads the banner and the size line. Only `coordinate {real,integer}
/// {general,symmetric}` is accepted, as in `MMReader`; alevin-fry writes
/// `real general`.
fn read_header(r: &mut dyn BufRead, line: &mut Vec<u8>) -> Result<Header> {
    line.clear();
    r.read_until(b'\n', line)?;
    let banner = String::from_utf8_lossy(line).to_lowercase();
    let toks: Vec<&str> = banner.split_ascii_whitespace().collect();
    let (integer, symmetric) = match toks.as_slice() {
        ["%%matrixmarket", "matrix", "coordinate", f @ ("real" | "integer"), s @ ("general" | "symmetric")] => {
            (*f == "integer", *s == "symmetric")
        }
        _ => bail!("unsupported MatrixMarket banner {banner:?}"),
    };
    loop {
        line.clear();
        ensure!(
            r.read_until(b'\n', line)? > 0,
            "MatrixMarket file has no size line"
        );
        if line[0] != b'%' && !line.iter().all(u8::is_ascii_whitespace) {
            break;
        }
    }
    let v: Vec<usize> = std::str::from_utf8(line)?
        .split_ascii_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .context("bad MatrixMarket size line")?;
    ensure!(
        v.len() == 3,
        "MatrixMarket size line must have 3 fields, not {v:?}"
    );
    Ok(Header {
        nrows: v[0],
        ncols: v[1],
        entries: v[2],
        integer,
        symmetric,
    })
}

/// Skips ASCII whitespace, the one set of separators of this reader.
fn skip_blank(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Parses the 1-based index at `b[i..]`; returns (0-based index, next position).
/// One leading `+` is accepted, as `usize::from_str` in `MMReader` does.
fn parse_index(b: &[u8], i: usize, bound: usize, what: &str) -> Result<(usize, usize)> {
    let mut i = skip_blank(b, i);
    if b.get(i) == Some(&b'+') {
        i += 1;
    }
    let start = i;
    let mut v: usize = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v
            .checked_mul(10)
            .and_then(|v| v.checked_add(usize::from(b[i] - b'0')))
            .ok_or_else(|| anyhow!("{what} index overflows"))?;
        i += 1;
    }
    ensure!(
        i > start,
        "no {what} index in {:?}",
        String::from_utf8_lossy(b)
    );
    ensure!(
        b.get(i).is_none_or(u8::is_ascii_whitespace),
        "no blank after the {what} index in {:?}",
        String::from_utf8_lossy(b)
    );
    ensure!(
        (1..=bound).contains(&v),
        "{what} index {v} is not in 1..={bound}"
    );
    Ok((v - 1, i))
}

/// Parses the row and the column of an entry; returns (row, column, next position).
fn parse_row_col(b: &[u8], h: &Header) -> Result<(usize, usize, usize)> {
    let (row, i) = parse_index(b, 0, h.nrows, "row")?;
    let (col, i) = parse_index(b, i, h.ncols, "column")?;
    ensure!(
        !h.symmetric || row == col || (col < h.nrows && row < h.ncols),
        "the mirror entry ({}, {}) of the symmetric matrix is out of range",
        col + 1,
        row + 1
    );
    Ok((row, col, i))
}

/// Calls `f` for each entry line; fails unless there are exactly `entries` lines.
fn for_each_entry(
    r: &mut dyn BufRead,
    line: &mut Vec<u8>,
    entries: usize,
    mut f: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut seen = 0usize;
    loop {
        line.clear();
        if r.read_until(b'\n', line)? == 0 {
            break;
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        ensure!(
            seen < entries,
            "MatrixMarket file has more than {entries} entries"
        );
        f(line)?;
        seen += 1;
    }
    ensure!(
        seen == entries,
        "MatrixMarket header gives {entries} entries, the file has {seen}"
    );
    Ok(())
}

/// Pass 1: returns the header and the CSR row offsets.
fn count_rows(path: &Path) -> Result<(Header, Vec<usize>)> {
    let mut line = Vec::with_capacity(128);
    let mut r = open(path)?;
    let h = read_header(&mut r, &mut line)?;
    let mut indptr = vec![0usize; h.nrows + 1];
    for_each_entry(&mut r, &mut line, h.entries, |b| {
        if h.symmetric {
            let (row, col, _) = parse_row_col(b, &h)?;
            indptr[row + 1] += 1;
            if row != col {
                indptr[col + 1] += 1;
            }
        } else {
            indptr[parse_index(b, 0, h.nrows, "row")?.0 + 1] += 1;
        }
        Ok(())
    })?;
    for i in 0..h.nrows {
        indptr[i + 1] += indptr[i];
    }
    Ok((h, indptr))
}

/// Pass 2: puts each entry in its row and writes the CSR arrays as `X` of
/// `store`. `I` is the index type on disk, `T` the value type.
fn fill_and_write<I, T>(
    path: &Path,
    h: &Header,
    indptr: Vec<usize>,
    store: &<H5 as Backend>::Store,
) -> Result<()>
where
    I: BackendData + Copy + Default + Ord + TryFrom<usize>,
    T: BackendData + Copy + Default + FromStr,
{
    let nnz = indptr[h.nrows];
    let mut indices = vec![I::default(); nnz];
    let mut data = vec![T::default(); nnz];
    let mut cursor = indptr[..h.nrows].to_vec();
    let mut put = |row: usize, col: usize, val: T| -> Result<()> {
        let p = cursor[row];
        ensure!(
            p < indptr[row + 1],
            "row {row} has more entries than in pass 1"
        );
        indices[p] = I::try_from(col).map_err(|_| anyhow!("column {col} overflows"))?;
        data[p] = val;
        cursor[row] += 1;
        Ok(())
    };

    let mut line = Vec::with_capacity(128);
    let mut r = open(path)?;
    ensure!(
        read_header(&mut r, &mut line)? == *h,
        "{} changed after pass 1",
        path.display()
    );
    for_each_entry(&mut r, &mut line, h.entries, |b| {
        let (row, col, i) = parse_row_col(b, h)?;
        let i = skip_blank(b, i);
        let j = i + b[i..]
            .iter()
            .take_while(|c| !c.is_ascii_whitespace())
            .count();
        ensure!(
            b[j..].iter().all(u8::is_ascii_whitespace),
            "extra field in {:?}",
            String::from_utf8_lossy(b)
        );
        // str::parse, as MMReader does, so the values are bit-identical.
        let val: T = std::str::from_utf8(&b[i..j])?
            .parse()
            .map_err(|_| anyhow!("bad value in {:?}", String::from_utf8_lossy(b)))?;
        put(row, col, val)?;
        if h.symmetric && row != col {
            put(col, row, val)?;
        }
        Ok(())
    })?;
    // A row with fewer entries than in pass 1 would keep slots with no value.
    if let Some(row) = (0..h.nrows).find(|&r| cursor[r] != indptr[r + 1]) {
        bail!(
            "{} changed after pass 1: row {row} has fewer entries than in pass 1",
            path.display()
        );
    }

    // A row whose entries are not in column order is sorted; a duplicate is an error.
    let mut tmp: Vec<(I, T)> = Vec::new();
    for row in 0..h.nrows {
        let (lo, hi) = (indptr[row], indptr[row + 1]);
        if indices[lo..hi].windows(2).all(|w| w[0] < w[1]) {
            continue;
        }
        tmp.clear();
        tmp.extend(
            indices[lo..hi]
                .iter()
                .copied()
                .zip(data[lo..hi].iter().copied()),
        );
        tmp.sort_unstable_by_key(|e| e.0);
        ensure!(
            tmp.windows(2).all(|w| w[0].0 < w[1].0),
            "duplicate entry in row {row}"
        );
        for (k, &(c, v)) in tmp.iter().enumerate() {
            indices[lo + k] = c;
            data[lo + k] = v;
        }
    }

    let indptr: Vec<I> = indptr
        .into_iter()
        .map(|x| I::try_from(x).map_err(|_| anyhow!("indptr value {x} overflows")))
        .collect::<Result<_>>()?;

    // The same group as anndata-rs CsrMatrix::write (data/array/sparse/csr.rs).
    let mut g = store.new_group("X")?;
    g.new_attr("encoding-type", "csr_matrix")?;
    g.new_attr("encoding-version", "0.1.0")?;
    g.new_attr("shape", Value::Array(vec![h.nrows.into(), h.ncols.into()]))?;
    g.new_array_dataset("data", data.as_slice().into(), get_default_write_config())?;
    g.new_array_dataset(
        "indptr",
        indptr.as_slice().into(),
        get_default_write_config(),
    )?;
    g.new_array_dataset(
        "indices",
        indices.as_slice().into(),
        get_default_write_config(),
    )?;
    Ok(())
}

/// Creates the h5ad file `output_path` with `X` read from the MatrixMarket
/// file `mtx_path`, and returns it open.
pub(crate) fn mtx_to_anndata(mtx_path: &Path, output_path: &Path) -> Result<AnnData<H5>> {
    let ctx = || format!("cannot read {}", mtx_path.display());
    let (h, indptr) = count_rows(mtx_path).with_context(ctx)?;
    // The index type rule of CsrMatrix::write: i32 if ncols - 1 and nnz fit, else i64.
    let small =
        i32::try_from(h.ncols.saturating_sub(1)).is_ok() && i32::try_from(indptr[h.nrows]).is_ok();
    let store = H5::new(output_path)?;
    match (small, h.integer) {
        (true, false) => fill_and_write::<i32, f64>(mtx_path, &h, indptr, &store),
        (false, false) => fill_and_write::<i64, f64>(mtx_path, &h, indptr, &store),
        (true, true) => fill_and_write::<i32, i64>(mtx_path, &h, indptr, &store),
        (false, true) => fill_and_write::<i64, i64>(mtx_path, &h, indptr, &store),
    }
    .with_context(ctx)?;
    AnnData::<H5>::open(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::backend::DatasetOp;
    use anndata::data::array::DynCsrMatrix;
    use anndata::{reader::MMReader, AnnDataOp, ArrayData, ArrayElemOp};
    use nalgebra_sparse::CsrMatrix;
    use std::io::Write;
    use std::path::PathBuf;

    fn tmp(test: &str, ext: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "af_anndata_mtx_{test}_{}.{ext}",
            std::process::id()
        ))
    }

    fn text(test: &str, s: &str) -> PathBuf {
        let p = tmp(test, "mtx");
        std::fs::write(&p, s).unwrap();
        p
    }

    /// X as stored in an h5ad file: the values, the on-disk type of each X
    /// dataset, and the layout (the X attributes, and the shape, chunk size and
    /// filters of each X dataset).
    type StoredX = (ArrayData, Vec<String>, Vec<String>);

    /// Reads X of `b` and its layout; removes the file.
    fn x_of(b: AnnData<H5>, path: &Path) -> StoredX {
        let x: ArrayData = b.x().get().unwrap().unwrap();
        b.close().unwrap();
        let g = H5::open(path).unwrap().open_group("X").unwrap();
        let names = ["data", "indptr", "indices"];
        let types = names.map(|d| format!("{:?}", g.open_dataset(d).unwrap().dtype().unwrap()));
        let mut attrs = g.attr_names().unwrap();
        attrs.sort();
        let mut layout: Vec<String> = attrs
            .iter()
            .map(|a| format!("X@{a}={}", g.get_json_attr(a).unwrap()))
            .collect();
        for d in names {
            let ds = g.open_dataset(d).unwrap();
            layout.push(format!(
                "X/{d} shape={:?} chunk={:?} filters={:?}",
                ds.shape(),
                ds.chunk(),
                ds.filters()
            ));
        }
        drop(g);
        std::fs::remove_file(path).unwrap();
        (x, types.to_vec(), layout)
    }

    /// Reads `mtx` with this reader and with the path of af-anndata 8619966
    /// (`MMReader::finish`, which calls `set_x`), checks that X, the on-disk
    /// types and the layout are the same, removes `mtx`, and returns them.
    fn same_as_mmreader(test: &str, mtx: &Path) -> StoredX {
        let (a, b) = (
            tmp(&format!("{test}_new"), "h5ad"),
            tmp(&format!("{test}_old"), "h5ad"),
        );
        let new = x_of(mtx_to_anndata(mtx, &a).unwrap(), &a);
        let old = AnnData::<H5>::new(&b).unwrap();
        MMReader::from_path(mtx).unwrap().finish(&old).unwrap();
        let old = x_of(old, &b);
        std::fs::remove_file(mtx).unwrap();
        assert_eq!(new, old, "this reader and MMReader give a different X");
        new
    }

    fn csr_f64(x: ArrayData) -> CsrMatrix<f64> {
        let ArrayData::CsrMatrix(DynCsrMatrix::F64(m)) = x else {
            panic!("X is not a f64 CSR")
        };
        m
    }

    fn strings<const N: usize>(s: [&str; N]) -> Vec<String> {
        s.map(String::from).to_vec()
    }

    // Row 2 is split in two runs, row 1 is unsorted, row 3 is empty; the rows
    // are not in order; comment lines and trailing blanks as sprs writes them.
    const REAL: &str = "%%MatrixMarket matrix coordinate real general\n% written by sprs\n\
        4 5 7            \n2 1 1.5\n2 3 2\n1 5 3\n1 2 0.25\n4 4 7\n2 4 1e-3\n2 2 9\n";

    #[test]
    fn real_matrix_equals_mmreader() {
        let (x, types, _) = same_as_mmreader("real", &text("real", REAL));
        assert_eq!(types, strings(["F64", "I32", "I32"]));
        let m = csr_f64(x);
        assert_eq!(m.row_offsets(), [0, 2, 6, 6, 7]);
        assert_eq!(m.col_indices(), [1, 4, 0, 1, 2, 3, 3]);
        assert_eq!(m.values(), [0.25, 3.0, 1.5, 9.0, 2.0, 1e-3, 7.0]);
    }

    #[test]
    fn integer_matrix_equals_mmreader() {
        let s = REAL
            .replace("real", "integer")
            .replace("1.5", "1")
            .replace("0.25", "6")
            .replace("1e-3", "8");
        let (x, types, _) = same_as_mmreader("int", &text("int", &s));
        assert!(matches!(x, ArrayData::CsrMatrix(DynCsrMatrix::I64(_))));
        assert_eq!(types, strings(["I64", "I32", "I32"]));
    }

    #[test]
    fn gzip_input_equals_mmreader() {
        let p = tmp("gz", "mtx.gz");
        let mut gz = flate2::write::GzEncoder::new(
            File::create(&p).unwrap(),
            flate2::Compression::default(),
        );
        gz.write_all(REAL.as_bytes()).unwrap();
        gz.finish().unwrap();
        assert_eq!(csr_f64(same_as_mmreader("gz", &p).0).nnz(), 7);
    }

    #[test]
    fn crlf_line_ends_equal_mmreader() {
        let p = text("crlf", &REAL.replace('\n', "\r\n"));
        let m = csr_f64(same_as_mmreader("crlf", &p).0);
        assert_eq!(m.values(), [0.25, 3.0, 1.5, 9.0, 2.0, 1e-3, 7.0]);
    }

    #[test]
    fn plus_sign_on_an_index_equals_mmreader() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 2\n+2 1 4\n1 +2 -3\n";
        let m = csr_f64(same_as_mmreader("plus", &text("plus", s)).0);
        assert_eq!(m.values(), [-3.0, 4.0]);
    }

    #[test]
    fn symmetric_matrix_equals_mmreader() {
        // The lower triangle, as the MatrixMarket format keeps a symmetric matrix.
        let s = "%%MatrixMarket matrix coordinate real symmetric\n3 3 4\n\
                 1 1 1\n3 1 2\n2 2 3\n3 2 4\n";
        let m = csr_f64(same_as_mmreader("sym", &text("sym", s)).0);
        assert_eq!(m.row_offsets(), [0, 2, 4, 6]);
        assert_eq!(m.col_indices(), [0, 2, 1, 2, 0, 1]);
        assert_eq!(m.values(), [1.0, 2.0, 3.0, 4.0, 2.0, 4.0]);
    }

    #[test]
    fn tab_and_form_feed_separators_equal_mmreader() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 2\n1\x0c2 3\n \t2\t1\x0c\t4\n";
        let m = csr_f64(same_as_mmreader("ws", &text("ws", s)).0);
        assert_eq!(m.values(), [3.0, 4.0]);
    }

    #[test]
    fn one_by_one_matrix_equals_mmreader() {
        let s = "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 5\n";
        let m = csr_f64(same_as_mmreader("one", &text("one", s)).0);
        assert_eq!((m.nrows(), m.ncols(), m.values()), (1, 1, &[5.0][..]));
    }

    #[test]
    fn matrix_with_no_entries_equals_mmreader() {
        let s = "%%MatrixMarket matrix coordinate real general\n3 2 0\n";
        let m = csr_f64(same_as_mmreader("empty", &text("empty", s)).0);
        assert_eq!((m.nrows(), m.ncols(), m.nnz()), (3, 2, 0));
    }

    #[test]
    fn shuffled_matrix_larger_than_the_compression_limit_equals_mmreader() {
        // anndata-rs compresses only arrays with more than 128 values.
        let mut seed: u64 = 42;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let (nrows, ncols) = (60u64, 40u64);
        let mut e: Vec<(u64, u64, u64)> = Vec::new();
        for r in 1..=nrows {
            for c in 1..=ncols {
                if next() % 5 == 0 {
                    e.push((r, c, next() % 1000));
                }
            }
        }
        for i in (1..e.len()).rev() {
            e.swap(i, next() as usize % (i + 1));
        }
        assert!(e.len() > 300, "the fixture has only {} entries", e.len());
        let mut s = format!(
            "%%MatrixMarket matrix coordinate real general\n{nrows} {ncols} {}\n",
            e.len()
        );
        for (r, c, v) in &e {
            s += &format!("{r} {c} {}\n", *v as f64 / 7.0);
        }
        let (x, _, layout) = same_as_mmreader("shuffled", &text("shuffled", &s));
        assert_eq!(csr_f64(x).nnz(), e.len());
        let data = layout.iter().find(|l| l.starts_with("X/data ")).unwrap();
        assert!(
            data.contains("chunk=Some(") && !data.contains("filters=[]"),
            "X/data is not chunked and compressed: {data}"
        );
    }

    #[test]
    fn more_than_i32_columns_give_i64_indices_as_mmreader() {
        let s = "%%MatrixMarket matrix coordinate real general\n\
                 1 3000000000 2\n1 3000000000 2\n1 1 1\n";
        let (x, types, _) = same_as_mmreader("wide", &text("wide", s));
        assert_eq!(types, strings(["F64", "I64", "I64"]));
        assert_eq!(csr_f64(x).col_indices(), [0, 2_999_999_999]);
    }

    fn err(test: &str, s: &str) -> String {
        let (mtx, out) = (text(test, s), tmp(test, "h5ad"));
        let e = format!("{:#}", mtx_to_anndata(&mtx, &out).unwrap_err());
        std::fs::remove_file(&mtx).unwrap();
        if out.exists() {
            std::fs::remove_file(&out).unwrap();
        }
        e
    }

    // MMReader adds the values of a duplicate entry; this reader stops.
    #[test]
    fn duplicate_entry_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 2 1\n1 2 1\n";
        assert!(err("dup", s).contains("duplicate entry in row 0"));
    }

    #[test]
    fn fewer_entries_than_the_header_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 3\n1 2 1\n2 1 1\n";
        assert!(err("fewer", s).contains("header gives 3 entries, the file has 2"));
    }

    // MMReader ignores the lines after the header count; this reader stops.
    #[test]
    fn more_entries_than_the_header_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 2 1\n2 1 1\n";
        assert!(err("more", s).contains("has more than 1 entries"));
    }

    #[test]
    fn index_out_of_range_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 3 1\n";
        assert!(err("range", s).contains("column index 3 is not in 1..=2"));
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 1\n";
        assert!(err("zero", s).contains("row index 0 is not in 1..=2"));
    }

    #[test]
    fn extra_field_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 2 1 7\n";
        assert!(err("extra", s).contains("extra field"));
    }

    // MMReader parses each field with str::parse, so "2.5" and "1+2" are not indices.
    #[test]
    fn an_index_without_a_blank_after_it_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 2.5\n";
        assert!(err("nosep_col", s).contains("no blank after the column index"));
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1+2 3\n";
        assert!(err("nosep_row", s).contains("no blank after the row index"));
    }

    // A symmetric file with the same header and line count, but with a
    // diagonal entry in pass 2 in place of an off-diagonal one in pass 1.
    #[test]
    fn a_file_that_changes_between_the_passes_is_an_error() {
        let sym = "%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n1 1 1\n";
        let pass1 = text("pass1", &format!("{sym}2 1 1\n"));
        let pass2 = text("pass2", &format!("{sym}2 2 1\n"));
        let out = tmp("passes", "h5ad");
        let (h, indptr) = count_rows(&pass1).unwrap();
        let e = fill_and_write::<i32, f64>(&pass2, &h, indptr, &H5::new(&out).unwrap());
        for p in [pass1, pass2, out] {
            std::fs::remove_file(p).unwrap();
        }
        let e = format!("{:#}", e.unwrap_err());
        assert!(e.contains("row 0 has fewer entries than in pass 1"), "{e}");
    }

    // MMReader drops the tokens of the size line that are not numbers, and
    // accepts a banner that only contains the words it looks for; this
    // reader stops on both.
    #[test]
    fn extra_tokens_in_the_size_line_or_the_banner_are_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1 junk\n1 1 1\n";
        assert!(err("sizejunk", s).contains("bad MatrixMarket size line"));
        let s = "%%MatrixMarket matrix coordinate real general extra\n2 2 1\n1 1 1\n";
        assert!(err("bannerjunk", s).contains("unsupported MatrixMarket banner"));
    }

    // MMReader splits fields at Unicode whitespace; this reader only at ASCII
    // whitespace.
    #[test]
    fn unicode_whitespace_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real general\n2 2 1\n1\u{3000}2 3\n";
        assert!(err("unicode", s).contains("no blank after the row index"));
    }

    #[test]
    fn symmetric_mirror_out_of_range_is_an_error() {
        let s = "%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 3 1\n";
        assert!(err("symrange", s).contains("mirror entry (3, 1)"));
    }

    // MMReader reads skew-symmetric as symmetric (with no negation) and
    // refuses hermitian, complex and pattern.
    #[test]
    fn other_formats_are_refused() {
        let formats = [
            "real skew-symmetric",
            "real hermitian",
            "complex general",
            "pattern general",
        ];
        for (k, f) in formats.iter().enumerate() {
            let s = format!("%%MatrixMarket matrix coordinate {f}\n2 2 1\n1 1 1\n");
            assert!(err(&format!("fmt{k}"), &s).contains("unsupported MatrixMarket banner"));
        }
    }
}
