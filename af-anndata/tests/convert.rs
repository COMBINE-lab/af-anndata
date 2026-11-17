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
    let ids = str_column(&var, "gene_ids");
    let symbols = str_column(&var, "gene_names");

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
    assert_eq!(str_column(&var, "gene_names")[g4], None);
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

// ---------------------------------------------------------------------------
// Probe matrix (`alevin-fry quant-probe --probe-mtx`)
// ---------------------------------------------------------------------------

use af_anndata::convert_probe_csr_to_anndata;

/// Probe-set CSV rows: (gene_id, probe_id, included, region, gene_name).
/// `GX1` and `GX2` are `included=FALSE` probes that are not matrix columns.
const PROBE_CSV: [(&str, &str, &str, &str, &str); 8] = [
    ("G1", "G1|Gene1|p1", "TRUE", "unspliced", "Gene1"),
    ("G1", "G1|Gene1|p2", "TRUE", "spliced", "Gene1"),
    ("G2", "G2|Gene2|p1", "TRUE", "unspliced", "Gene2"),
    ("GX1", "GX1|EXTRA1|p1", "FALSE", "unspliced", "EXTRA1"),
    ("G3", "G3|Gene3|p1", "TRUE", "spliced", "Gene3"),
    ("G4", "G4|Gene4|p1", "TRUE", "unspliced", "Gene4"),
    ("GX2", "GX2|EXTRA2|p1", "FALSE", "spliced", "EXTRA2"),
    ("G3", "G3|Gene3|p2", "TRUE", "unspliced", "Gene3"),
];
/// Probe matrix columns: another order than the CSV, not grouped by gene.
/// Columns 1 and 4 (the two G1 probes) have counts in the same cells.
const PROBES: [&str; 6] = [
    "G3|Gene3|p2",
    "G1|Gene1|p1",
    "G4|Gene4|p1",
    "G2|Gene2|p1",
    "G1|Gene1|p2",
    "G3|Gene3|p1",
];

fn probe_gene(probe: &str) -> &'static str {
    PROBE_CSV.iter().find(|r| r.1 == probe).unwrap().0
}

/// Probe counts as (cell, probe column, value).
fn probe_entries() -> Vec<(usize, usize, f64)> {
    let mut e = Vec::new();
    for i in (0..NCELLS).filter(|&i| i != EMPTY_CELL) {
        e.push((i, i % PROBES.len(), (i % 5 + 1) as f64));
        e.push((i, (i + 3) % PROBES.len(), (i % 3 + 1) as f64));
    }
    e
}

fn write_mtx(path: &Path, ncols: usize, e: &[(usize, usize, f64)]) {
    let mut mtx = format!(
        "%%MatrixMarket matrix coordinate real general\n% written by sprs\n{} {} {:<20}\n",
        NCELLS,
        ncols,
        e.len()
    );
    for (i, j, v) in e {
        mtx.push_str(&format!("{} {} {}\n", i + 1, j + 1, v));
    }
    fs::write(path, mtx).unwrap();
}

/// Writes the probe-set CSV. With `gene_name` false, the CSV has no
/// `gene_name` column.
fn write_probe_csv(path: &Path, gene_name: bool) {
    let mut csv = String::from("#probe_set_file_format=3.0\n#panel_name=Test Panel v1\n");
    csv.push_str("gene_id,probe_seq,probe_id,included,region");
    csv.push_str(if gene_name { ",gene_name\n" } else { "\n" });
    for (g, p, inc, reg, name) in PROBE_CSV {
        csv.push_str(&format!("{g},ACGTACGTAC,{p},{inc},{reg}"));
        if gene_name {
            csv.push_str(&format!(",{name}"));
        }
        csv.push('\n');
    }
    fs::write(path, csv).unwrap();
}

/// A multiplex `af_quant` of a `quant-probe --probe-mtx` run. The gene matrix
/// is the probe matrix summed by gene. With `csv` true, the probe-set CSV is
/// at `<root>/probe_set.csv`.
fn make_probe_af_quant(name: &str, csv: bool) -> PathBuf {
    let root = make_af_quant(name, false);
    let alevin = root.join("alevin");
    let pe = probe_entries();
    write_mtx(&alevin.join("probe_quants_mat.mtx"), PROBES.len(), &pe);
    fs::write(
        alevin.join("probe_quants_mat_cols.txt"),
        PROBES.join("\n") + "\n",
    )
    .unwrap();
    fs::copy(
        alevin.join("quants_mat_rows.txt"),
        alevin.join("probe_quants_mat_rows.txt"),
    )
    .unwrap();
    let mut gene: HashMap<(usize, usize), f64> = HashMap::new();
    for (i, j, v) in &pe {
        let g = GENES
            .iter()
            .position(|g| *g == probe_gene(PROBES[*j]))
            .unwrap();
        *gene.entry((*i, g)).or_insert(0.0) += v;
    }
    let mut ge: Vec<(usize, usize, f64)> = gene.into_iter().map(|((i, j), v)| (i, j, v)).collect();
    ge.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    write_mtx(&alevin.join("quants_mat.mtx"), GENES.len(), &ge);
    fs::write(
        root.join("quant.json"),
        format!(
            r#"{{"usa_mode": false, "num_genes": {}, "feature_axis": "probe", "probe_mtx": true, "num_probes": {}}}"#,
            GENES.len(),
            PROBES.len()
        ),
    )
    .unwrap();
    if csv {
        write_probe_csv(&root.join("probe_set.csv"), true);
    }
    root
}

fn convert_probe(root: &Path, probe_set: Option<&Path>) -> anyhow::Result<PathBuf> {
    let out = root.join("probe_out.h5ad");
    convert_probe_csr_to_anndata(root, &out, probe_set, ConvertOpts::default())?;
    Ok(out)
}

#[test]
fn probe_var_comes_from_the_probe_set_csv() {
    let root = make_probe_af_quant("probe_var", true);
    let out = convert_probe(&root, None).unwrap();
    let adata = open(&out);
    assert_eq!(adata.var_names().into_vec(), PROBES.to_vec());
    let var = adata.read_var().unwrap();
    let cols: Vec<&str> = var.get_column_names().iter().map(|c| c.as_str()).collect();
    // The CSV header, with no rename and no drop.
    assert_eq!(
        cols,
        [
            "gene_id",
            "probe_seq",
            "probe_id",
            "included",
            "region",
            "gene_name"
        ]
    );
    let genes = str_column(&var, "gene_id");
    let seqs = str_column(&var, "probe_seq");
    let ids = str_column(&var, "probe_id");
    let region = str_column(&var, "region");
    let names = str_column(&var, "gene_name");
    // The CSV reader reads TRUE/FALSE as bool.
    let included = var.column("included").unwrap().bool().unwrap();
    for (k, p) in PROBES.iter().enumerate() {
        let row = PROBE_CSV.iter().find(|r| r.1 == *p).unwrap();
        assert_eq!(genes[k].as_deref(), Some(row.0));
        assert_eq!(seqs[k].as_deref(), Some("ACGTACGTAC"));
        assert_eq!(ids[k].as_deref(), Some(*p));
        assert_eq!(included.get(k), Some(row.2 == "TRUE"));
        assert_eq!(region[k].as_deref(), Some(row.3));
        assert_eq!(names[k].as_deref(), Some(row.4));
    }
}

#[test]
fn probe_x_summed_by_gene_equals_gene_x_and_obs_is_the_same() {
    let root = make_probe_af_quant("probe_sum", true);
    let probe_out = convert_probe(&root, None).unwrap();
    let gene_out = root.join("gene_out.h5ad");
    convert_csr_to_anndata_with_opts(&root, &gene_out, ConvertOpts::default()).unwrap();

    let pa = open(&probe_out);
    let ga = open(&gene_out);
    let px: CsrMatrix<f64> = pa.x().get().unwrap().unwrap();
    let gx: CsrMatrix<f64> = ga.x().get().unwrap().unwrap();
    let probe_genes = str_column(&pa.read_var().unwrap(), "gene_id");
    let gene_names = ga.var_names().into_vec();

    let mut summed: HashMap<(usize, String), f64> = HashMap::new();
    for (i, j, v) in px.triplet_iter() {
        *summed
            .entry((i, probe_genes[j].clone().unwrap()))
            .or_insert(0.0) += v;
    }
    let gene: HashMap<(usize, String), f64> = gx
        .triplet_iter()
        .map(|(i, j, v)| ((i, gene_names[j].clone()), *v))
        .collect();
    assert_eq!(summed, gene, "probe X summed by var gene_id equals gene X");

    assert_eq!(pa.obs_names().into_vec(), ga.obs_names().into_vec());
    let (po, go) = (pa.read_obs().unwrap(), ga.read_obs().unwrap());
    for c in ["sample_name", "cell_barcode"] {
        assert_eq!(str_column(&po, c), str_column(&go, c), "obs {c}");
    }
}

#[test]
fn probe_ref_path_is_used_and_a_csv_without_gene_name_stays_without_it() {
    let root = make_probe_af_quant("probe_noname", false);
    let csv = root.join("other_dir_probe_set.csv");
    write_probe_csv(&csv, false);
    let out = convert_probe(&root, Some(&csv)).unwrap();
    let var = open(&out).read_var().unwrap();
    let cols: Vec<&str> = var.get_column_names().iter().map(|c| c.as_str()).collect();
    // No gene_id_to_name.tsv fallback at probe level.
    assert_eq!(
        cols,
        ["gene_id", "probe_seq", "probe_id", "included", "region"]
    );
}

#[test]
fn probe_conversion_without_a_csv_names_the_path_it_tried() {
    let root = make_probe_af_quant("probe_nocsv", false);
    let err = convert_probe(&root, None).unwrap_err().to_string();
    assert!(
        err.contains(&root.join("probe_set.csv").display().to_string()),
        "{err}"
    );
    let missing = root.join("no_such.csv");
    let err = convert_probe(&root, Some(&missing))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no_such.csv"), "{err}");
}

#[test]
fn a_matrix_probe_that_is_not_in_the_csv_is_an_error() {
    let root = make_probe_af_quant("probe_missing", false);
    let csv = root.join("probe_set.csv");
    write_probe_csv(&csv, true);
    let text = fs::read_to_string(&csv).unwrap();
    let text: String = text
        .lines()
        .filter(|l| !l.contains("G4|Gene4|p1"))
        .map(|l| format!("{l}\n"))
        .collect();
    fs::write(&csv, text).unwrap();
    let err = convert_probe(&root, None).unwrap_err().to_string();
    assert!(
        err.contains("G4|Gene4|p1") && err.contains("1 of 6"),
        "{err}"
    );
}

#[test]
fn probe_rows_must_be_the_gene_rows() {
    let root = make_probe_af_quant("probe_rows", true);
    let rows = root.join("alevin/probe_quants_mat_rows.txt");
    let mut text = fs::read_to_string(&rows).unwrap();
    text = text.replacen("S_1_A_", "S_2_B_", 1);
    fs::write(&rows, text).unwrap();
    let err = convert_probe(&root, None).unwrap_err().to_string();
    assert!(err.contains("probe_quants_mat_rows.txt"), "{err}");
}

#[test]
fn probe_sort_index_keeps_the_annotations_with_their_probe() {
    let root = make_probe_af_quant("probe_sorted", true);
    let out = root.join("probe_sorted.h5ad");
    convert_probe_csr_to_anndata(&root, &out, None, ConvertOpts { sort_index: true }).unwrap();
    let adata = open(&out);
    let names = adata.var_names().into_vec();
    let mut sorted = PROBES.map(String::from).to_vec();
    sorted.sort();
    assert_eq!(names, sorted);
    let var = adata.read_var().unwrap();
    let genes = str_column(&var, "gene_id");
    let region = str_column(&var, "region");
    for (k, p) in names.iter().enumerate() {
        let row = PROBE_CSV.iter().find(|r| r.1 == p.as_str()).unwrap();
        assert_eq!(genes[k].as_deref(), Some(row.0), "gene of {p}");
        assert_eq!(region[k].as_deref(), Some(row.3), "region of {p}");
    }
}

#[test]
fn probe_conversion_rejects_usa_mode() {
    let root = make_probe_af_quant("probe_usa", true);
    fs::write(
        root.join("quant.json"),
        r#"{"usa_mode": true, "num_genes": 4, "probe_mtx": true}"#,
    )
    .unwrap();
    let err = convert_probe(&root, None).unwrap_err().to_string();
    assert!(
        err.contains("usa_mode") && err.contains("no USA mode"),
        "{err}"
    );
}
