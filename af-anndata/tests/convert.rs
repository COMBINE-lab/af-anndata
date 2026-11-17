//! Tests that convert a small synthetic `af_quant` directory and read the
//! h5ad file back.
//!
//! The fixture is a multiplex (Flex) run:
//! - Sample names contain `_`, and rows of different samples are interleaved.
//! - One cell has no counts, so its `MeanByMax` is `NaN`.
//! - `gene_id_to_name.tsv` is in another order than the matrix columns, has ids
//!   that are not matrix columns, and has no name for one column.
//! - There are more than 128 cells and values, because anndata-rs compresses
//!   only arrays with more than 128 values.

use af_anndata::{convert_csr_to_anndata_with_opts, ConvertOpts};
use anndata::{AnnData, AnnDataOp, ArrayElemOp, Backend};
use anndata_hdf5::H5;
use nalgebra_sparse::CsrMatrix;
use polars::prelude::DataFrame;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Matrix columns, in column order (not sorted).
const GENES: [&str; 4] = ["G2", "G1", "G4", "G3"];
/// Names. `G4` has no name; `GX1` and `GX2` are not matrix columns.
const NAMES: [(&str, &str); 5] = [
    ("GX1", "EXTRA1"),
    ("G3", "Gene3"),
    ("G1", "Gene1"),
    ("GX2", "EXTRA2"),
    ("G2", "Gene2"),
];
const SAMPLES: [&str; 3] = ["S_1_A", "S_2_B", "S_3"];
const NCELLS: usize = 150;
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

    let names: String = NAMES.iter().map(|(id, n)| format!("{id}\t{n}\n")).collect();
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

fn open(path: &Path) -> AnnData<H5> {
    AnnData::<H5>::open(H5::open(path).unwrap()).unwrap()
}

fn str_column(df: &DataFrame, name: &str) -> Vec<Option<String>> {
    let col = df
        .column(name)
        .unwrap_or_else(|_| panic!("var has no column {name}"));
    let col = col.cast(&polars::prelude::DataType::String).unwrap();
    col.str()
        .unwrap()
        .iter()
        .map(|s| s.map(str::to_string))
        .collect()
}

/// The expected name of each gene id: a left join of NAMES onto GENES.
fn expected_names() -> HashMap<&'static str, Option<&'static str>> {
    let names: HashMap<_, _> = NAMES.iter().cloned().collect();
    GENES.iter().map(|g| (*g, names.get(g).copied())).collect()
}

fn check_var(out: &Path) {
    let adata = open(out);
    let var_names = adata.var_names().into_vec();
    let var = adata.read_var().unwrap();
    let ids = str_column(&var, "gene_id");
    let symbols = str_column(&var, "gene_symbol");

    assert_eq!(var.height(), GENES.len(), "one var row per matrix column");
    let expected = expected_names();
    for k in 0..var_names.len() {
        assert_eq!(ids[k].as_deref(), Some(var_names[k].as_str()));
        assert_eq!(
            symbols[k].as_deref(),
            expected[var_names[k].as_str()],
            "name of {}",
            var_names[k]
        );
    }
}

#[test]
fn extra_ids_in_name_file_do_not_panic_and_names_join_on_id() {
    let out = convert("multiplex_plain", false, false);
    let adata = open(&out);
    assert_eq!(adata.var_names().into_vec(), GENES.to_vec());
    check_var(&out);
}

#[test]
fn a_column_with_no_name_is_missing_not_empty() {
    let out = convert("multiplex_missing", false, false);
    let adata = open(&out);
    let var = adata.read_var().unwrap();
    let g4 = GENES.iter().position(|g| *g == "G4").unwrap();
    assert_eq!(str_column(&var, "gene_symbol")[g4], None);
}

#[test]
fn sort_index_keeps_the_names_with_their_ids() {
    let out = convert("multiplex_sorted", false, true);
    let adata = open(&out);
    assert_eq!(adata.var_names().into_vec(), vec!["G1", "G2", "G3", "G4"]);
    check_var(&out);
}

#[test]
fn x_has_the_mtx_values_and_obs_has_the_samples() {
    let out = convert("multiplex_values", false, false);
    let adata = open(&out);
    let x: CsrMatrix<f64> = adata.x().get().unwrap().unwrap();
    assert_eq!((x.nrows(), x.ncols()), (NCELLS, GENES.len()));
    let mut want = vec![vec![0.0; GENES.len()]; NCELLS];
    for (i, j, v) in entries(GENES.len()) {
        want[i][j] += v;
    }
    for (i, j, v) in x.triplet_iter() {
        assert_eq!(*v, want[i][j], "X[{i}, {j}]");
    }
    let x_sum: f64 = x.values().iter().sum();
    let want_sum: f64 = entries(GENES.len()).iter().map(|e| e.2).sum();
    assert_eq!(x_sum, want_sum);

    let obs = adata.read_obs().unwrap();
    let samples = str_column(&obs, "sample_name");
    let cbs = str_column(&obs, "cell_barcode");
    for i in 0..NCELLS {
        assert_eq!(samples[i].as_deref(), Some(SAMPLES[i % SAMPLES.len()]));
        assert_eq!(cbs[i].as_deref(), Some(barcode(i).as_str()));
    }
    let mbm = obs
        .column("mean_by_max")
        .unwrap()
        .f64()
        .unwrap()
        .get(EMPTY_CELL);
    assert!(mbm.unwrap().is_nan(), "MeanByMax of the empty cell is NaN");
}

#[test]
fn usa_mode_var_has_one_row_per_gene_with_names() {
    let out = convert("multiplex_usa", true, false);
    let adata = open(&out);
    assert_eq!(adata.var_names().into_vec(), GENES.to_vec());
    check_var(&out);

    let ncols = 3 * GENES.len();
    let x: CsrMatrix<f64> = adata.x().get().unwrap().unwrap();
    assert_eq!((x.nrows(), x.ncols()), (NCELLS, GENES.len()));
    let x_sum: f64 = x.values().iter().sum();
    let want_sum: f64 = entries(ncols).iter().map(|e| e.2).sum();
    assert_eq!(x_sum, want_sum, "X is the sum of the S, U and A blocks");
}
