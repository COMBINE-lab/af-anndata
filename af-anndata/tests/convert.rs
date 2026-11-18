//! Tests that the h5ad file uses gzip and no Blosc, so that plain h5py and
//! anndata can read it.
//!
//! The fixture is a small multiplex `af_quant` directory with more than 128
//! cells and values, because anndata-rs compresses only arrays with more than
//! 128 values.

use af_anndata::{convert_csr_to_anndata_with_opts, ConvertOpts};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// Each test holds this lock. HDF5 opens files without `O_CLOEXEC` and locks
/// them with `flock` (HDF5 1.x `H5FDsec2.c`, `H5private.h`). A python child,
/// started by one test, gets a copy of the files that other tests have open,
/// and keeps their locks. A later open then fails with errno 11 (EAGAIN).
static HDF5_LOCK: Mutex<()> = Mutex::new(());

fn hdf5_lock() -> MutexGuard<'static, ()> {
    // A test that panics poisons the lock; the other tests can still run.
    HDF5_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Matrix columns, in column order (not sorted).
const GENES: [&str; 4] = ["G2", "G1", "G4", "G3"];
const SAMPLES: [&str; 3] = ["S_1_A", "S_2_B", "S_3"];
const NCELLS: usize = 400;
/// The cell with no counts.
const EMPTY_CELL: usize = 7;

fn barcode(i: usize) -> String {
    // 8 nt, ACGT only.
    (0..8)
        .map(|k| ['A', 'C', 'G', 'T'][(i >> (2 * k)) & 3])
        .collect()
}

/// The fixture counts as (cell, column, value), for `ncols` matrix columns.
fn entries(ncols: usize) -> Vec<(usize, usize, f64)> {
    let mut e = Vec::new();
    for i in (0..NCELLS).filter(|&i| i != EMPTY_CELL) {
        e.push((i, i % ncols, (i % 5 + 1) as f64));
        e.push((i, (i + 1) % ncols, (i % 3 + 1) as f64));
    }
    e
}

/// Writes a synthetic `af_quant` directory and returns its path.
fn make_af_quant(name: &str, usa: bool) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    fs::create_dir_all(root.join("alevin")).unwrap();

    let cols: Vec<String> = if usa {
        GENES
            .iter()
            .map(|g| g.to_string())
            .chain(GENES.iter().map(|g| format!("{g}-U")))
            .chain(GENES.iter().map(|g| format!("{g}-A")))
            .collect()
    } else {
        GENES.iter().map(|g| g.to_string()).collect()
    };
    fs::write(
        root.join("alevin/quants_mat_cols.txt"),
        cols.join("\n") + "\n",
    )
    .unwrap();

    let rows: Vec<String> = (0..NCELLS)
        .map(|i| format!("{}_{}", SAMPLES[i % SAMPLES.len()], barcode(i)))
        .collect();
    fs::write(
        root.join("alevin/quants_mat_rows.txt"),
        rows.join("\n") + "\n",
    )
    .unwrap();

    // The alevin-fry MTX banner, with the nnz field padded to 20 characters.
    let e = entries(cols.len());
    let mut mtx = format!(
        "%%MatrixMarket matrix coordinate real general\n% written by sprs\n{} {} {:<20}\n",
        NCELLS,
        cols.len(),
        e.len()
    );
    for (i, j, v) in &e {
        mtx.push_str(&format!("{} {} {}\n", i + 1, j + 1, v));
    }
    fs::write(root.join("alevin/quants_mat.mtx"), mtx).unwrap();

    let mut fd = String::from(
        "CB\tsample_name\tCorrectedReads\tMappedReads\tDeduplicatedReads\tMappingRate\tDedupRate\tMeanByMax\tNumGenesExpressed\tNumGenesOverMean\n",
    );
    for i in 0..NCELLS {
        let mean_by_max = if i == EMPTY_CELL { "NaN" } else { "0.5" };
        fd.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t1\t0.5\t{}\t2\t1\n",
            barcode(i),
            SAMPLES[i % SAMPLES.len()],
            10 + i,
            10 + i,
            5 + i,
            mean_by_max
        ));
    }
    fs::write(root.join("featureDump.txt"), fd).unwrap();

    let names: String = GENES.iter().map(|g| format!("{g}\tName{g}\n")).collect();
    fs::write(root.join("gene_id_to_name.tsv"), names).unwrap();

    fs::write(
        root.join("quant.json"),
        format!(r#"{{"usa_mode": {usa}, "num_genes": {}}}"#, GENES.len()),
    )
    .unwrap();
    fs::write(root.join("collate.json"), r#"{"multi_barcode": true}"#).unwrap();
    fs::write(root.join("generate_permit_list.json"), "{}").unwrap();
    fs::write(root.join("simpleaf_map_info.json"), r#"{"mapper": "test"}"#).unwrap();
    root
}

fn convert(name: &str, usa: bool, sort_index: bool) -> PathBuf {
    let root = make_af_quant(name, usa);
    let out = root.join("out.h5ad");
    convert_csr_to_anndata_with_opts(&root, &out, ConvertOpts { sort_index })
        .expect("conversion must succeed");
    out
}

/// Returns (path, HDF5 filter ids) for each dataset in the file.
fn dataset_filters(path: &Path) -> Vec<(String, Vec<i32>)> {
    fn walk(g: &hdf5_metno::Group, out: &mut Vec<(String, Vec<i32>)>) {
        for d in g.datasets().unwrap() {
            let ids = d.filters().iter().map(|f| f.id()).collect();
            out.push((d.name(), ids));
        }
        for sub in g.groups().unwrap() {
            walk(&sub, out);
        }
    }
    let file = hdf5_metno::File::open(path).unwrap();
    let mut out = Vec::new();
    walk(&file, &mut out);
    out
}

const H5Z_FILTER_DEFLATE: i32 = 1;
const BLOSC_FILTER: i32 = 32001;

#[test]
fn h5ad_uses_gzip_and_no_blosc() {
    let _lock = hdf5_lock();
    for (name, usa, sort_index) in [
        ("gzip_plain", false, false),
        ("gzip_sorted", false, true),
        ("gzip_usa", true, false),
    ] {
        let before = anndata::backend::get_default_write_config().compression;
        let out = convert(name, usa, sort_index);
        assert_eq!(
            anndata::backend::get_default_write_config().compression,
            before,
            "the caller's write config is put back"
        );
        let filters: HashMap<String, Vec<i32>> = dataset_filters(&out).into_iter().collect();
        for (dset, ids) in &filters {
            assert!(!ids.contains(&BLOSC_FILTER), "{name}: {dset} uses Blosc");
        }
        let mut compressed = vec!["/X/data", "/X/indices", "/obs/corrected_reads"];
        if usa {
            compressed.extend(["/layers/spliced/data", "/layers/unspliced/data"]);
        }
        for dset in compressed {
            assert!(
                filters[dset].contains(&H5Z_FILTER_DEFLATE),
                "{name}: {dset} is not gzip compressed: {:?}",
                filters[dset]
            );
        }
    }
}

/// Reads the h5ad with plain h5py (no `hdf5plugin`) and with anndata: no
/// dataset uses Blosc, `X` uses gzip, and `X` equals the fixture MTX.
///
/// Ignored by default because it needs a python with h5py and anndata. Run it
/// with:
///
/// ```text
/// AF_ANNDATA_TEST_PYTHON=python3 cargo test --release -- --ignored
/// ```
#[test]
#[ignore = "needs python3 with h5py and anndata"]
fn h5ad_reads_with_plain_h5py_and_anndata() {
    let _lock = hdf5_lock();
    const SCRIPT: &str = r#"
import os
import sys
import h5py
import anndata

path, ngenes = sys.argv[1], int(sys.argv[2])

# The expected X: the fixture MTX, with the USA blocks summed onto the genes.
want = {}
mtx = os.path.join(os.path.dirname(path), "alevin", "quants_mat.mtx")
with open(mtx) as m:
    for line in m.readlines()[3:]:
        i, j, v = line.split()
        key = (int(i) - 1, (int(j) - 1) % ngenes)
        want[key] = want.get(key, 0.0) + float(v)

with h5py.File(path, "r") as f:
    filters = {}
    f.visititems(lambda n, o: filters.__setitem__(n, dict(o._filters)) if isinstance(o, h5py.Dataset) else None)
    assert not [n for n, fl in filters.items() if "32001" in fl], filters
    assert "gzip" in filters["X/data"], filters["X/data"]
    # Read every dataset: a Blosc dataset fails here without hdf5plugin.
    f.visititems(lambda name, obj: obj[()] if isinstance(obj, h5py.Dataset) else None)
    data, indices, indptr = (f["X"][k][()] for k in ("data", "indices", "indptr"))
got = {}
for r in range(len(indptr) - 1):
    for k in range(indptr[r], indptr[r + 1]):
        got[(r, int(indices[k]))] = float(data[k])
assert got == want, "X differs from the MTX"
assert "hdf5plugin" not in sys.modules
ad = anndata.read_h5ad(path)
print("ok", ad.shape)
"#;
    let python = std::env::var("AF_ANNDATA_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    for (name, usa) in [("py_plain", false), ("py_usa", true)] {
        let out = convert(name, usa, false);
        let res = std::process::Command::new(&python)
            .env_remove("HDF5_PLUGIN_PATH")
            .args(["-I", "-c", SCRIPT])
            .arg(&out)
            .arg(GENES.len().to_string())
            .output()
            .expect("could not start python");
        assert!(
            res.status.success(),
            "{name}: python failed:\n{}\n{}",
            String::from_utf8_lossy(&res.stdout),
            String::from_utf8_lossy(&res.stderr)
        );
    }
}
