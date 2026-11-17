//! Converts [alevin-fry] / [simpleaf] quantification output into the [AnnData]
//! (`.h5ad`) format.
//!
//! Point [`convert_csr_to_anndata`] at an `af_quant` directory and it writes a
//! single HDF5-backed AnnData file: the matrix-market counts become `X`, the cell
//! barcodes and gene ids become the `obs` and `var` indices, per-cell metrics from
//! `featureDump.txt` become `obs` columns, and the run's JSON metadata becomes
//! `uns` entries.
//!
//! For a 10x Flex run of `alevin-fry quant-probe --probe-mtx`,
//! [`convert_probe_csr_to_anndata`] writes the probe matrix to a second file,
//! with the same `obs` and a `var` from the 10x probe-set CSV.
//!
//! ```no_run
//! use af_anndata::{convert_csr_to_anndata_with_opts, ConvertOpts};
//! use std::path::Path;
//!
//! # fn main() -> anyhow::Result<()> {
//! convert_csr_to_anndata_with_opts(
//!     Path::new("af_quant"),
//!     Path::new("out.h5ad"),
//!     ConvertOpts { sort_index: true },
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # USA mode
//!
//! In USA mode the feature axis carries unspliced, spliced and ambiguous counts
//! for every gene, so there are three times as many columns as genes. These are
//! split into the `unspliced`, `spliced` and `ambiguous` layers and `X` is set to
//! their sum, over a `var` axis of just the genes.
//!
//! # Dependency version coupling
//!
//! `anndata` does not re-export `polars` or `nalgebra-sparse`, yet both cross its
//! API boundary. This crate must declare exactly the versions `anndata` requires;
//! anything else resolves two copies and fails with `cannot add
//! &CsrMatrix<i32> to CsrMatrix<i32>`, where both types look identical. See the
//! note in `Cargo.toml` and the README.
//!
//! [alevin-fry]: https://github.com/COMBINE-lab/alevin-fry
//! [simpleaf]: https://github.com/COMBINE-lab/simpleaf
//! [AnnData]: https://anndata.readthedocs.io/en/stable/

use anndata::data::array::dataframe::DataFrameIndex;
use anndata::{reader::MMReader, s, AnnData, AnnDataOp, ArrayData, ArrayElemOp};
use anndata_hdf5::H5;
use anyhow::{bail, Context};
use polars::io::prelude::*;
use polars::prelude::{Column, CsvReadOptions, DataFrame, DataType, Field, PolarsError, Schema};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{error, info, trace, warn};

/// Tags the actual type of matrix that has
/// been populated in a `CSRMatPaack`
enum PopulatedMatType {
    I32,
    I64,
    F32,
    F64,
}

/// Holds the different representable types of
/// matrices. It is expected that only
/// one of these should be populated.
struct CSRMatPack {
    pub mat_i32: nalgebra_sparse::CsrMatrix<i32>,
    pub mat_i64: nalgebra_sparse::CsrMatrix<i64>,
    pub mat_f32: nalgebra_sparse::CsrMatrix<f32>,
    pub mat_f64: nalgebra_sparse::CsrMatrix<f64>,
}

impl CSRMatPack {
    /// Create a new CSRMatPack with empty matrices for all supported types
    pub fn new(nr: usize, ngenes: usize) -> Self {
        Self {
            mat_i32: nalgebra_sparse::CsrMatrix::<i32>::zeros(nr, ngenes),
            mat_i64: nalgebra_sparse::CsrMatrix::<i64>::zeros(nr, ngenes),
            mat_f32: nalgebra_sparse::CsrMatrix::<f32>::zeros(nr, ngenes),
            mat_f64: nalgebra_sparse::CsrMatrix::<f64>::zeros(nr, ngenes),
        }
    }

    pub fn ncells(&self) -> usize {
        self.mat_f64.nrows()
    }

    pub fn ngenes(&self) -> usize {
        self.mat_f64.ncols()
    }

    /// Returns a `PopulatedMatType` enum specifying the type of matrix in this
    /// pack that is populated.  If more than one type is populated, it returns
    /// an error.
    pub fn populated_type(&self) -> anyhow::Result<Option<PopulatedMatType>> {
        let mut t: Option<PopulatedMatType> = None;
        let mut nset = 0_usize;
        if self.mat_i32.nnz() > 0 {
            nset += 1;
            t = Some(PopulatedMatType::I32);
        } else if self.mat_i64.nnz() > 0 {
            nset += 1;
            t = Some(PopulatedMatType::I64);
        } else if self.mat_f32.nnz() > 0 {
            nset += 1;
            t = Some(PopulatedMatType::F32);
        } else if self.mat_f64.nnz() > 0 {
            nset += 1;
            t = Some(PopulatedMatType::F64);
        }
        if nset > 1 {
            bail!("The CSRMatPack has > 1 set matrix type. This should not happen");
        }
        if t.is_some() {
            Ok(t)
        } else {
            Ok(None)
        }
    }
}

/// accumulates the content of `slice` into the apropriately typed member of `csr_accum` and
/// returns `OK(CSRMatPack)` with the accumulated result in the appropriately typed member upon
/// success. If an unsupported matrix type is provided in the `slice`, an error is raised.
fn accumulate_layer(slice: &ArrayData, mut csr_accum: CSRMatPack) -> anyhow::Result<CSRMatPack> {
    match slice {
        anndata::data::array::ArrayData::CsrMatrix(a) => {
            trace!("confirmed ArrayData slice is in CSR format");
            match a {
                anndata::data::array::DynCsrMatrix::I8(_l) => {
                    bail!("I8 matrix type is not supported")
                }
                anndata::data::array::DynCsrMatrix::I16(_l) => {
                    bail!("I16 matrix type is not supported")
                }
                anndata::data::array::DynCsrMatrix::I32(l) => {
                    csr_accum.mat_i32 = csr_accum.mat_i32 + l;
                }
                anndata::data::array::DynCsrMatrix::I64(l) => {
                    csr_accum.mat_i64 = csr_accum.mat_i64 + l;
                }
                anndata::data::array::DynCsrMatrix::U8(_l) => {
                    bail!("U8 matrix type is not supported")
                }
                anndata::data::array::DynCsrMatrix::U16(_l) => {
                    bail!("U16 matrix type is not supported")
                }
                anndata::data::array::DynCsrMatrix::U32(_l) => {
                    bail!(
                        "Addition is not supported for U32 CSR matrices because they do not satisfy NEG<Output=T>; storing result in an i64"
                    );
                }
                anndata::data::array::DynCsrMatrix::U64(_l) => {
                    bail!(
                        "Addition is not supported for U32 CSR matrices because they do not satisfy NEG<Output=T>; storing result in an i64"
                    );
                }
                anndata::data::array::DynCsrMatrix::F32(l) => {
                    csr_accum.mat_f32 = csr_accum.mat_f32 + l;
                }
                anndata::data::array::DynCsrMatrix::F64(l) => {
                    csr_accum.mat_f64 = csr_accum.mat_f64 + l;
                }
                anndata::data::array::DynCsrMatrix::Bool(_l) => {
                    bail!("Bool matrix type is not supported")
                }
                anndata::data::array::DynCsrMatrix::String(_l) => {
                    bail!("String matrix type is not supported")
                }
            };
        }
        _ => warn!("expected underlying CSR matrix; cannot populate X layer with sum counts!"),
    }
    Ok(csr_accum)
}

/// Sets the X layer of the `AnnData` object, as well as the `n_obs` and `n_vars` values.
/// Any existing X layer will be deleted.  The contents of the new `X` layer will depend on
/// which matrix type in the `csr_in` `CSRMatPack` is populated.
fn set_x_layer<B: anndata::Backend>(b: &mut AnnData<B>, csr_in: CSRMatPack) -> anyhow::Result<()> {
    // get rid of the old X
    b.del_x().context("unable to delete X")?;
    b.del_obs()?;
    b.set_n_obs(csr_in.ncells())
        .context("unable to set n_obs")?;
    //b.del_var()?;
    b.set_n_vars(csr_in.ngenes())
        .context("unable to set n_vars")?;
    // set the new X
    match csr_in
        .populated_type()
        .context("error getting populated type of the X matrix")?
    {
        Some(PopulatedMatType::I32) => {
            let csr_mat = anndata::data::array::DynCsrMatrix::I32(csr_in.mat_i32);
            b.set_x(csr_mat).context("unable to set all 0s X")?;
        }
        Some(PopulatedMatType::I64) => {
            let csr_mat = anndata::data::array::DynCsrMatrix::I64(csr_in.mat_i64);
            b.set_x(csr_mat).context("unable to set all 0s X")?;
        }
        Some(PopulatedMatType::F32) => {
            let csr_mat = anndata::data::array::DynCsrMatrix::F32(csr_in.mat_f32);
            b.set_x(csr_mat).context("unable to set all 0s X")?;
        }
        Some(PopulatedMatType::F64) => {
            let csr_mat = anndata::data::array::DynCsrMatrix::F64(csr_in.mat_f64);
            b.set_x(csr_mat).context("unable to set all 0s X")?;
        }
        None => {
            warn!(
                "None of the underlying matrices for the layers had counts; setting the output to the trivial empty matrix (of type f64)"
            );
            let csr_mat = anndata::data::array::DynCsrMatrix::F64(
                nalgebra_sparse::CsrMatrix::<f64>::zeros(csr_in.ncells(), csr_in.ngenes()),
            );
            b.set_x(csr_mat).context("unable to set all 0s X")?;
        }
    }
    Ok(())
}

fn separate_usa_layers<B: anndata::Backend>(
    b: &mut AnnData<B>,
    row_df: &DataFrame,
    var_df: DataFrame,
) -> anyhow::Result<()> {
    let mut sw = libsw::Sw::new();
    sw.start()?;

    let nr = b.n_obs();
    let nc = b.n_vars();
    // if USA mode then the number of genes is
    // 1/3 of the number of features
    let ngenes = nc / 3;

    let mut csr_zero = CSRMatPack::new(nr, ngenes);

    // Get the unspliced, spliced and ambiguous slices
    let slice1: ArrayData = b.get_x().slice(s![.., 0..ngenes])?.unwrap();
    csr_zero = accumulate_layer(&slice1, csr_zero)?;

    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice2: ArrayData = b.get_x().slice(s![.., ngenes..2 * ngenes])?.unwrap();
    csr_zero = accumulate_layer(&slice2, csr_zero)?;
    //let var2 = vars.slice(ngenes as i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice3: ArrayData = b.get_x().slice(s![.., 2 * ngenes..3 * ngenes])?.unwrap();
    csr_zero = accumulate_layer(&slice3, csr_zero)?;
    //let var3 = vars.slice(2_i64 * ngenes as i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    set_x_layer(b, csr_zero)?;
    b.set_var(var_df)?;

    let layers = vec![
        ("spliced".to_owned(), slice1),
        ("unspliced".to_owned(), slice2),
        ("ambiguous".to_owned(), slice3),
    ];
    b.set_layers(layers)
        .context("unable to set layers for AnnData object")?;
    info!("setting layers took {:#?}", sw.elapsed());
    b.set_obs(row_df.clone())?;

    Ok(())
}

fn parse_multiplex_barcodes(obs_names: &[String]) -> Option<(Vec<String>, Vec<String>)> {
    let mut sample_names = Vec::with_capacity(obs_names.len());
    let mut cell_barcodes = Vec::with_capacity(obs_names.len());

    for obs_name in obs_names {
        let (sample_name, cell_barcode) = obs_name.rsplit_once('_')?;
        sample_names.push(sample_name.to_string());
        cell_barcodes.push(cell_barcode.to_string());
    }

    Some((sample_names, cell_barcodes))
}

/// Reads `gene_id_to_name.tsv`: two tab-separated columns (gene id, gene name),
/// no header. Returns an error for a line that does not have two fields, or for
/// an id that occurs more than one time.
fn read_gene_id_to_name(path: &Path) -> anyhow::Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read {}", path.display()))?;
    let mut names = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let Some((id, name)) = line.split_once('\t') else {
            bail!(
                "{} line {}: expected 2 tab-separated fields (gene id, gene name), found {:?}",
                path.display(),
                i + 1,
                line
            );
        };
        if name.contains('\t') {
            bail!(
                "{} line {}: expected 2 tab-separated fields (gene id, gene name), found more",
                path.display(),
                i + 1
            );
        }
        if names.insert(id.to_string(), name.to_string()).is_some() {
            bail!(
                "{} line {}: the gene id {:?} occurs more than one time",
                path.display(),
                i + 1,
                id
            );
        }
    }
    Ok(names)
}

/// Builds the `var` frame from the matrix column ids (`gene_ids`, in column
/// order), then left-joins the names of `gene_id_to_name` onto them.
///
/// Ids in the name file that are not matrix columns are ignored. A Flex index
/// has such ids: the genes of the probes that are not in the index. A column id
/// that has no name gets a missing value, as a left join in pyroe `load_fry`.
fn build_var_df(gene_ids: &[String], gene_id_to_name: Option<&Path>) -> anyhow::Result<DataFrame> {
    let mut columns = vec![Column::new("gene_ids".into(), gene_ids)];
    if let Some(path) = gene_id_to_name {
        let names = read_gene_id_to_name(path)?;
        let symbols: Vec<Option<&str>> = gene_ids
            .iter()
            .map(|id| names.get(id).map(String::as_str))
            .collect();
        let n_missing = symbols.iter().filter(|s| s.is_none()).count();
        if n_missing > 0 {
            warn!(
                "{} of {} matrix columns have no name in {}; their gene_names value is missing",
                n_missing,
                gene_ids.len(),
                path.display()
            );
        }
        columns.push(Column::new("gene_names".into(), symbols));
    }
    Ok(DataFrame::new(gene_ids.len(), columns)?)
}

/// Options controlling how an `af_quant` directory is converted to AnnData.
///
/// This is a struct rather than extra parameters so that new knobs can be added
/// without breaking callers; construct it with `..Default::default()`.
#[derive(Debug, Clone, Default)]
pub struct ConvertOpts {
    /// Sort `obs` and `var` by their index (barcode and gene id, respectively)
    /// before writing. The result is a pure permutation of the unsorted output --
    /// `X`, every layer, and the `obs`/`var` annotations are all permuted
    /// together. Useful for producing byte-comparable output across runs, e.g.
    /// in CI. See COMBINE-lab/af-anndata#1.
    pub sort_index: bool,
}

/// Convert an `af_quant` directory to an HDF5-backed AnnData file using default options.
///
/// Equivalent to [`convert_csr_to_anndata_with_opts`] with [`ConvertOpts::default`].
pub fn convert_csr_to_anndata<P: AsRef<Path>>(root_path: P, output_path: P) -> anyhow::Result<()> {
    convert_csr_to_anndata_with_opts(root_path, output_path, ConvertOpts::default())
}

/// Convert an `af_quant` directory to an HDF5-backed AnnData file.
pub fn convert_csr_to_anndata_with_opts<P: AsRef<Path>>(
    root_path: P,
    output_path: P,
    opts: ConvertOpts,
) -> anyhow::Result<()> {
    write_anndata(root_path.as_ref(), output_path.as_ref(), opts, None)
}

/// The file name of the 10x probe-set CSV in an `af_quant` directory.
pub const PROBE_SET_FILE: &str = "probe_set.csv";

/// Convert the probe matrix of an `af_quant` directory to an HDF5-backed
/// AnnData file.
///
/// The input is the output of `alevin-fry quant-probe --probe-mtx`:
/// `alevin/probe_quants_mat.mtx` (cells x probes) and its rows and cols files.
/// `obs` is the same as in the gene file of [`convert_csr_to_anndata_with_opts`].
/// `var` comes from the 10x probe-set CSV, joined on `probe_id`: the columns
/// `probe_id`, `gene_ids`, `gene_names` and `region` (`gene_names` and `region`
/// only if the CSV has them). If the CSV has no `gene_name` column, the names
/// come from `gene_id_to_name.tsv`.
///
/// The CSV is `probe_set`, or else `<root_path>/probe_set.csv`. It is an error
/// if neither exists, or if a matrix probe is not in the CSV.
pub fn convert_probe_csr_to_anndata<P: AsRef<Path>>(
    root_path: P,
    output_path: P,
    probe_set: Option<&Path>,
    opts: ConvertOpts,
) -> anyhow::Result<()> {
    let root_path = root_path.as_ref();
    let probe_set = find_probe_set(root_path, probe_set)?;
    write_anndata(root_path, output_path.as_ref(), opts, Some(&probe_set))
}

/// Returns the probe-set CSV: `given`, or else `<root_path>/probe_set.csv`.
fn find_probe_set(root_path: &Path, given: Option<&Path>) -> anyhow::Result<PathBuf> {
    if let Some(p) = given {
        if !p.is_file() {
            bail!("the probe-set CSV {} does not exist", p.display());
        }
        return Ok(p.to_path_buf());
    }
    let p = root_path.join(PROBE_SET_FILE);
    if !p.is_file() {
        bail!(
            "no probe-set CSV was given, and {} does not exist; give the 10x probe-set CSV of the run",
            p.display()
        );
    }
    Ok(p)
}

/// Reads the table of a 10x probe-set CSV, with all columns as strings. The
/// `#` lines before the header are skipped.
fn read_probe_set(path: &Path) -> anyhow::Result<DataFrame> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read {}", path.display()))?;
    let n_hash_lines = text.lines().take_while(|l| l.starts_with('#')).count();
    let table = CsvReadOptions::default()
        .with_has_header(true)
        .with_skip_lines(n_hash_lines)
        // 0 rows for the schema: every column is read as a string.
        .with_infer_schema_length(Some(0))
        .with_raise_if_empty(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()
        .with_context(|| format!("could not read the table of {}", path.display()))?;
    for required in ["probe_id", "gene_id"] {
        if table.column(required).is_err() {
            bail!(
                "{} has no `{}` column; its columns are {:?}",
                path.display(),
                required,
                table.get_column_names()
            );
        }
    }
    Ok(table)
}

/// Returns the values of the string column `name` of the probe-set table, in
/// the order of `rows`, or `None` if the table has no such column.
fn probe_set_column<'a>(
    table: &'a DataFrame,
    name: &str,
    rows: &[usize],
) -> anyhow::Result<Option<Vec<Option<&'a str>>>> {
    let Ok(col) = table.column(name) else {
        return Ok(None);
    };
    let col = col.str()?;
    Ok(Some(rows.iter().map(|&r| col.get(r)).collect()))
}

/// Makes the probe `var`: the probe ids are the matrix columns, and the
/// annotations are joined from the probe-set CSV on `probe_id`.
fn probe_var(
    probe_ids: &[String],
    probe_set: &Path,
    gene_id_to_name: Option<&Path>,
) -> anyhow::Result<DataFrame> {
    let table = read_probe_set(probe_set)?;

    let csv_ids = table.column("probe_id")?.str()?;
    let mut row_of = HashMap::with_capacity(csv_ids.len());
    for (r, id) in csv_ids.iter().enumerate() {
        let id = id.with_context(|| {
            format!("{} data row {} has no probe_id", probe_set.display(), r + 1)
        })?;
        if row_of.insert(id, r).is_some() {
            bail!(
                "{}: the probe_id {:?} occurs more than one time",
                probe_set.display(),
                id
            );
        }
    }
    let missing: Vec<&str> = probe_ids
        .iter()
        .filter(|id| !row_of.contains_key(id.as_str()))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        bail!(
            "{} of {} matrix probes are not in {} (the first ones: {:?}); give the probe-set CSV that the index was made from",
            missing.len(),
            probe_ids.len(),
            probe_set.display(),
            &missing[..missing.len().min(5)]
        );
    }
    let rows: Vec<usize> = probe_ids.iter().map(|id| row_of[id.as_str()]).collect();

    let gene_ids: Vec<&str> = probe_set_column(&table, "gene_id", &rows)?
        .expect("gene_id is a required column")
        .into_iter()
        .zip(probe_ids)
        .map(|(g, p)| {
            g.with_context(|| format!("{}: probe {} has no gene_id", probe_set.display(), p))
        })
        .collect::<anyhow::Result<_>>()?;

    let mut columns = vec![
        Column::new("probe_id".into(), probe_ids),
        Column::new("gene_ids".into(), &gene_ids),
    ];
    if let Some(names) = probe_set_column(&table, "gene_name", &rows)? {
        columns.push(Column::new("gene_names".into(), names));
    } else if let Some(path) = gene_id_to_name {
        let names = read_gene_id_to_name(path)?;
        let names: Vec<Option<&str>> = gene_ids
            .iter()
            .map(|g| names.get(*g).map(String::as_str))
            .collect();
        columns.push(Column::new("gene_names".into(), names));
    } else {
        warn!(
            "{} has no gene_name column and there is no gene_id_to_name file, so `var` has no gene_names",
            probe_set.display()
        );
    }
    if let Some(region) = probe_set_column(&table, "region", &rows)? {
        columns.push(Column::new("region".into(), region));
    }
    Ok(DataFrame::new(probe_ids.len(), columns)?)
}

/// Writes the gene matrix (`probe_set` is `None`) or the probe matrix of
/// `alevin-fry quant-probe --probe-mtx` (`probe_set` is the 10x probe-set CSV).
fn write_anndata(
    root_path: &Path,
    output_path: &Path,
    opts: ConvertOpts,
    probe_set: Option<&Path>,
) -> anyhow::Result<()> {
    let (mtx_file, rows_file, cols_file) = if probe_set.is_some() {
        (
            "probe_quants_mat.mtx",
            "probe_quants_mat_rows.txt",
            "probe_quants_mat_cols.txt",
        )
    } else {
        (
            "quants_mat.mtx",
            "quants_mat_rows.txt",
            "quants_mat_cols.txt",
        )
    };
    let json_path = PathBuf::from(&root_path);

    let mut gpl_path = json_path.clone();
    gpl_path.push("generate_permit_list.json");

    let mut collate_path = json_path.clone();
    collate_path.push("collate.json");

    let mut quant_path = json_path.clone();
    quant_path.push("quant.json");

    let mut map_log_path = json_path.clone();
    map_log_path.push("simpleaf_map_info.json");

    let mut multiplex_info_path = json_path.clone();
    multiplex_info_path.push("simpleaf_multiplex_quant_info.json");

    let mut sample_info_path = json_path.clone();
    sample_info_path.push("sample_info.json");

    let alevin_path = root_path.join("alevin");
    let mut p = PathBuf::from(&alevin_path);
    p.push(mtx_file);

    let mut colpath = PathBuf::from(&alevin_path);
    colpath.push(cols_file);

    let mut rowpath = PathBuf::from(&alevin_path);
    rowpath.push(rows_file);

    let mut gene_id_to_name_path = PathBuf::from(&root_path);
    gene_id_to_name_path.push("gene_id_to_name.tsv");

    if !p.is_file() {
        anyhow::bail!(
            "the count file was expected at {} but could not be found",
            p.display()
        );
    }
    if !colpath.is_file() {
        anyhow::bail!(
            "the column annotation file was expected at {} but could not be found",
            colpath.display()
        );
    }
    if !rowpath.is_file() {
        anyhow::bail!(
            "the row annotation file was expected at {} but could not be found",
            rowpath.display()
        );
    }
    if !gpl_path.is_file() {
        anyhow::bail!(
            "the generate_permit_list json file was expected at {} but could not be found",
            gpl_path.display()
        );
    }
    if !collate_path.is_file() {
        anyhow::bail!(
            "the collate json file was expected at {} but could not be found",
            collate_path.display()
        );
    }
    if !quant_path.is_file() {
        anyhow::bail!(
            "the quant json file was expected at {} but could not be found",
            quant_path.display()
        );
    }

    // see if we have a valid gene id to name file
    let gene_id_to_name_path = gene_id_to_name_path
        .is_file()
        .then_some(gene_id_to_name_path);
    // otherwise, wan the user (a probe run can take the names from its CSV)
    if gene_id_to_name_path.is_none() && probe_set.is_none() {
        warn!(
            "Could not find the `gene_id_to_name` file, so only gene IDs and not symbols will be present in `var`"
        );
    }

    // read in the relevant JSON files
    let qf = std::fs::File::open(&quant_path)?;
    let quant_json: Value = serde_json::from_reader(qf)
        .with_context(|| format!("could not parse {} as valid JSON.", quant_path.display()))?;

    let cf = std::fs::File::open(&collate_path)?;
    let collate_json: Value = serde_json::from_reader(cf)
        .with_context(|| format!("could not parse {} as valid JSON.", collate_path.display()))?;

    let gplf = std::fs::File::open(&gpl_path)?;
    let gpl_json: Value = serde_json::from_reader(gplf)
        .with_context(|| format!("could not parse {} as valid JSON.", gpl_path.display()))?;

    let sample_info_json: Option<Value> =
        if let Ok(sample_info_file) = std::fs::File::open(&sample_info_path) {
            Some(serde_json::from_reader(sample_info_file).with_context(|| {
                format!(
                    "could not parse {} as valid JSON.",
                    sample_info_path.display()
                )
            })?)
        } else {
            None
        };

    let multiplex_info_json: Option<Value> =
        if let Ok(multiplex_info_file) = std::fs::File::open(&multiplex_info_path) {
            Some(
                serde_json::from_reader(multiplex_info_file).with_context(|| {
                    format!(
                        "could not parse {} as valid JSON.",
                        multiplex_info_path.display()
                    )
                })?,
            )
        } else {
            None
        };

    let map_json: Value = if let Ok(mapf) = std::fs::File::open(&map_log_path) {
        serde_json::from_reader(mapf)
            .with_context(|| format!("could not parse {} as valid JSON.", gpl_path.display()))?
    } else {
        warn!("Could not find a simpleaf_map_info.json in the provided directory; please upgrade to the latest version of simpleaf when possible!");
        serde_json::json!({
            "mapper" : "file_not_found",
            "num_mapped": 0,
            "num_poisoned": 0,
            "num_reads": 0,
            "percent_mapped": 0.
        })
    };

    let usa_mode = if let Some(Value::Bool(v)) = quant_json.get("usa_mode") {
        *v
    } else {
        false
    };
    let is_multi_barcode = collate_json
        .get("multi_barcode")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || sample_info_json.is_some();

    info!("USA mode : {}", usa_mode);

    let mut sw = libsw::Sw::new();
    sw.start()?;

    let r = MMReader::from_path(&p)?;

    let parse_opts = CsvParseOptions::default().with_separator(b'\t');
    // read the gene ids
    let mut col_df = match CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_opts.clone())
        .with_raise_if_empty(true)
        .try_into_reader_with_file_path(Some(colpath))?
        .finish()
    {
        Ok(dframe) => dframe,
        Err(PolarsError::NoData(estr)) => {
            error!("error reading column labels : {:?};", estr);
            bail!("failed to construct the column data frame.");
        }
        Err(err) => {
            bail!(err);
        }
    };
    col_df.set_column_names(&["gene_id"])?;

    // read the barcodes
    let mut row_df = match CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_opts.clone())
        .with_raise_if_empty(true)
        .try_into_reader_with_file_path(Some(rowpath.clone()))?
        .finish()
    {
        Ok(dframe) => dframe,
        Err(PolarsError::NoData(estr)) => {
            error!("error reading row labels : {:?};", estr);
            error!(
                "this likely indicates the row labels (barcode list) was empty --- please ensure the barcode list is properly matched to the chemistry being processed"
            );
            bail!("failed to construct the row data frame.");
        }
        Err(err) => {
            bail!(err);
        }
    };

    let nobs_cols = row_df.columns().len();
    match nobs_cols {
        1 => row_df.set_column_names(&["barcodes"])?,
        3 => row_df.set_column_names(&["barcodes", "spot_x", "spot_y"])?,
        x => {
            error!(
                "quants_mat_rows.txt file should have 1 (sc/sn-RNA) or 3 columns (spatial); the provided file has {}",
                x
            );
            bail!(
                "quants_mat_rows.txt file should have 1 (sc/sn-RNA) or 3 columns (spatial); the provided file has {}",
                x
            );
        }
    }
    let obs_barcodes: Vec<String> = row_df
        .column("barcodes")?
        .str()
        .unwrap()
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect();

    // Build `var` from the matrix columns. In USA mode the columns are the
    // spliced, unspliced and ambiguous blocks of the same genes, so the gene
    // ids are the first third of the columns.
    let ncols = col_df.height();
    let ngenes = if usa_mode {
        if ncols % 3 != 0 {
            bail!(
                "USA mode needs a multiple of 3 columns in quants_mat_cols.txt, found {}",
                ncols
            );
        }
        ncols / 3
    } else {
        ncols
    };
    let gene_ids: Vec<String> = col_df
        .column("gene_id")?
        .str()?
        .iter()
        .take(ngenes)
        .enumerate()
        .map(|(i, s)| {
            s.map(str::to_string)
                .with_context(|| format!("quants_mat_cols.txt line {} is empty", i + 1))
        })
        .collect::<anyhow::Result<_>>()?;
    // In a probe run, usa_mode is false, so `gene_ids` are all the probe ids.
    let var_df = if let Some(probe_set) = probe_set {
        if usa_mode {
            bail!(
                "{} has usa_mode true, but a probe matrix has no USA mode",
                quant_path.display()
            );
        }
        // obs comes from featureDump.txt, which follows the gene rows.
        let gene_rows = alevin_path.join("quants_mat_rows.txt");
        let same_rows = std::fs::read(&rowpath)
            .with_context(|| format!("could not read {}", rowpath.display()))?
            == std::fs::read(&gene_rows)
                .with_context(|| format!("could not read {}", gene_rows.display()))?;
        if !same_rows {
            bail!(
                "{} is not the same as {}, so featureDump.txt does not describe its rows",
                rowpath.display(),
                gene_rows.display()
            );
        }
        probe_var(&gene_ids, probe_set, gene_id_to_name_path.as_deref())?
    } else {
        build_var_df(&gene_ids, gene_id_to_name_path.as_deref())?
    };

    // make the AnnData object and populate it from the MMReader
    let mut b = AnnData::<H5>::new(output_path)?;
    r.finish(&b)?;
    info!("Reading MM into AnnData took {:#?}", sw.elapsed());

    // read in the feature dump data
    let mut feat_dump_path = PathBuf::from(&root_path);
    feat_dump_path.push("featureDump.txt");
    let feat_parse_options =
        polars::io::csv::read::CsvParseOptions::default().with_separator(b'\t');
    // Defense-in-depth: pin `CB` and `sample_name` to String explicitly.
    // Without these, Polars auto-infers their dtype from the first N rows
    // — if a multi-sample run ever produces malformed mixed-width rows
    // (see COMBINE-lab/simpleaf#195), `sample_name` would infer as Int64
    // from the integers leaking up from the next column and crash on
    // the first real well name like "A-A01".  `with_schema_overwrite`
    // only applies entries that match real columns, so the `sample_name`
    // entry is a no-op in single-sample runs.
    let feat_dump_schema = Arc::new(Schema::from_iter([
        Field::new("CB".into(), DataType::String),
        Field::new("sample_name".into(), DataType::String),
        Field::new("CorrectedReads".into(), DataType::Int64),
        Field::new("MappedReads".into(), DataType::Int64),
        Field::new("DeduplicatedReads".into(), DataType::Float64),
        Field::new("MappingRate".into(), DataType::Float64),
        Field::new("DedupRate".into(), DataType::Float64),
        Field::new("MeanByMax".into(), DataType::Float64),
        Field::new("NumGenesExpressed".into(), DataType::Int64),
        Field::new("NumGenesOverMean".into(), DataType::Int64),
    ]));
    let mut feat_dump_frame = match polars_io::csv::read::CsvReadOptions::default()
        .with_parse_options(feat_parse_options)
        .with_has_header(true)
        .with_schema_overwrite(Some(feat_dump_schema))
        .with_raise_if_empty(true)
        .try_into_reader_with_file_path(Some(feat_dump_path.clone()))
        .context("could not create TSV file reader")?
        .finish()
    {
        Ok(dframe) => dframe,
        Err(PolarsError::NoData(estr)) => {
            error!(
                "error reading the feature file ({}): {:?};",
                feat_dump_path.display(),
                estr
            );
            error!(
                "this likely indicates no barcodes were processed and written to the output --- please ensure the barcode list is properly matched to the chemistry being processed"
            );
            bail!("failed to construct the feature data frame.");
        }
        Err(err) => {
            bail!(err);
        }
    };
    // add the features to the row df
    // skip the first column since it is `CB` (the cell barcode) and is
    // redundant with the cell barcode we already have in this dataframe
    // CB      CorrectedReads  MappedReads     DeduplicatedReads       MappingRate     DedupRate       MeanByMax       NumGenesExpressed       NumGenesOverMean
    let col_rename = vec![
        ("CorrectedReads", "corrected_reads"),
        ("MappedReads", "mapped_reads"),
        ("DeduplicatedReads", "deduplicated_reads"),
        ("MappingRate", "mapping_rate"),
        ("DedupRate", "dedup_rate"),
        ("MeanByMax", "mean_by_max"),
        ("NumGenesExpressed", "num_genes_expressed"),
        ("NumGenesOverMean", "num_genes_over_mean"),
    ];
    for (old_name, new_name) in col_rename {
        feat_dump_frame.rename(old_name, new_name.into())?;
    }
    let has_sample_name = feat_dump_frame
        .columns()
        .iter()
        .any(|c| c.name().as_str() == "sample_name");
    let has_cb = feat_dump_frame
        .columns()
        .iter()
        .any(|c| c.name().as_str() == "CB");

    if is_multi_barcode {
        if has_cb {
            feat_dump_frame.rename("CB", "cell_barcode".into())?;
        }
        if let Some((sample_names, cell_barcodes)) = parse_multiplex_barcodes(&obs_barcodes) {
            if !has_sample_name {
                feat_dump_frame.with_column(Column::new("sample_name".into(), sample_names))?;
            }
            if !has_cb {
                feat_dump_frame.with_column(Column::new("cell_barcode".into(), cell_barcodes))?;
            }
        } else if !has_sample_name {
            warn!(
                "Could not derive multiplex sample_name/cell_barcode columns from obs names; expected values like SAMPLE_BARCODE."
            );
        }
    }
    let row_df = if is_multi_barcode {
        row_df.hstack(&feat_dump_frame.into_columns())?
    } else {
        row_df.hstack(&feat_dump_frame.into_columns()[1..])?
    };

    // read in the quant JSON file
    let gpl_json_str = serde_json::to_string(&gpl_json).context(
        "could not convert generate_permit_list.json to string succesfully to place in uns data.",
    )?;
    let collate_json_str = serde_json::to_string(&collate_json)
        .context("could not convert collate.json to string succesfully to place in uns data.")?;
    let quant_json_str = serde_json::to_string(&quant_json)
        .context("could not convert quant.json to string succesfully to place in uns data.")?;
    let map_log_json_str = serde_json::to_string(&map_json).context(
        "could not convert simpleaf_map_info.json to string succesfully to place in uns data.",
    )?;
    let sample_info_json_str = sample_info_json
        .as_ref()
        .map(|v| {
            serde_json::to_string(v).context(
                "could not convert sample_info.json to string succesfully to place in uns data.",
            )
        })
        .transpose()?;
    let multiplex_info_json_str = multiplex_info_json
        .as_ref()
        .map(|v| {
            serde_json::to_string(v).context(
                "could not convert simpleaf_multiplex_quant_info.json to string succesfully to place in uns data.",
            )
        })
        .transpose()?;

    // set unstructured metadata
    let mut uns: Vec<(String, anndata::Data)> = vec![
        ("gpl_info".to_owned(), anndata::Data::from(gpl_json_str)),
        (
            "collate_info".to_owned(),
            anndata::Data::from(collate_json_str),
        ),
        ("quant_info".to_owned(), anndata::Data::from(quant_json_str)),
        (
            "simpleaf_map_info".to_owned(),
            anndata::Data::from(map_log_json_str),
        ),
    ];
    if let Some(sample_info_json_str) = sample_info_json_str {
        uns.push((
            "sample_info".to_owned(),
            anndata::Data::from(sample_info_json_str),
        ));
    }
    if let Some(multiplex_info_json_str) = multiplex_info_json_str {
        uns.push((
            "simpleaf_multiplex_quant_info".to_owned(),
            anndata::Data::from(multiplex_info_json_str),
        ));
    }
    b.set_uns(uns).context("failed to set \"uns\" data")?;

    if usa_mode {
        separate_usa_layers(&mut b, &row_df, var_df)?;
    } else {
        b.set_var(var_df)?;
        b.set_obs(row_df.clone())?;
    }

    let var_index = DataFrameIndex::from(gene_ids.clone());
    b.set_var_names(var_index)?;

    let obs_index = DataFrameIndex::from(obs_barcodes.clone());
    b.set_obs_names(obs_index)?;

    if opts.sort_index {
        sort_by_index(&b, &obs_barcodes, &gene_ids)?;
    }

    Ok(())
}

/// Returns the permutation that sorts `keys` ascending, i.e. `argsort(keys)`.
fn argsort(keys: &[String]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..keys.len()).collect();
    perm.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    perm
}

/// Reorders `b` so that `obs` is sorted by barcode and `var` by gene id.
///
/// `AnnData::subset` permutes `X`, `layers`, `obs`/`var` and the `obsm`/`obsp`/
/// `varm`/`varp` collections together, so the result is a pure permutation: the
/// same values, reachable under the same labels, in sorted order.
fn sort_by_index<B: anndata::Backend>(
    b: &AnnData<B>,
    obs_barcodes: &[String],
    gene_ids: &[String],
) -> anyhow::Result<()> {
    use anndata::data::SelectInfoElem;

    let obs_perm = argsort(obs_barcodes);
    let var_perm = argsort(gene_ids);

    // Nothing to do if both axes are already ordered; `subset` is not free.
    let sorted = |p: &[usize]| p.windows(2).all(|w| w[0] < w[1]);
    if sorted(&obs_perm) && sorted(&var_perm) {
        info!("obs and var are already sorted by index; skipping the sort");
        return Ok(());
    }

    info!("sorting obs and var by index");
    let selection: Vec<SelectInfoElem> = vec![obs_perm.into(), var_perm.into()];
    b.subset(selection.as_slice())
        .context("unable to sort the AnnData object by obs/var index")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{argsort, parse_multiplex_barcodes, probe_var, read_gene_id_to_name};

    const PROBE_HEADER: &str = "#probe_set_file_format=3.0\ngene_id,probe_id,included\n";

    fn probe_var_err(test: &str, csv: &str) -> String {
        let p = name_file(test, csv);
        let ids = vec!["P1".to_string(), "P2".to_string()];
        let err = probe_var(&ids, &p, None).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        err
    }

    #[test]
    fn probe_set_needs_probe_id_and_gene_id_columns() {
        let err = probe_var_err("cols", "gene,probe_id\nG1,P1\nG2,P2\n");
        assert!(err.contains("no `gene_id` column"), "{err}");
    }

    #[test]
    fn probe_set_rejects_a_duplicate_probe_id() {
        let csv = format!("{PROBE_HEADER}G1,P1,TRUE\nG2,P2,TRUE\nG3,P1,TRUE\n");
        let err = probe_var_err("dupprobe", &csv);
        assert!(err.contains("\"P1\" occurs more than one time"), "{err}");
    }

    #[test]
    fn probe_set_without_optional_columns_gives_probe_id_and_gene_ids_only() {
        let p = name_file("minimal", "gene_id,probe_id\nG2,P2\nG1,P1\n");
        let ids = vec!["P1".to_string(), "P2".to_string()];
        let var = probe_var(&ids, &p, None).unwrap();
        std::fs::remove_file(&p).unwrap();
        let cols: Vec<&str> = var.get_column_names().iter().map(|c| c.as_str()).collect();
        assert_eq!(cols, ["probe_id", "gene_ids"]);
        let genes: Vec<Option<&str>> = var
            .column("gene_ids")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .collect();
        assert_eq!(genes, [Some("G1"), Some("G2")]);
    }

    fn name_file(test: &str, text: &str) -> std::path::PathBuf {
        let p =
            std::env::temp_dir().join(format!("af_anndata_{}_{}.tsv", test, std::process::id()));
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn gene_id_to_name_reads_two_columns() {
        let p = name_file("ok", "G1\tA\nG2\tB\n");
        let names = read_gene_id_to_name(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(names.len(), 2);
        assert_eq!(names["G2"], "B");
    }

    #[test]
    fn gene_id_to_name_rejects_a_duplicate_id() {
        let p = name_file("dup", "G1\tA\nG1\tB\n");
        let err = read_gene_id_to_name(&p).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        assert!(
            err.contains("line 2") && err.contains("more than one time"),
            "{err}"
        );
    }

    #[test]
    fn gene_id_to_name_rejects_a_line_without_two_fields() {
        let p = name_file("one", "G1\tA\nG2\n");
        let err = read_gene_id_to_name(&p).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        assert!(err.contains("line 2"), "{err}");

        let p = name_file("three", "G1\tA\tX\n");
        let err = read_gene_id_to_name(&p).unwrap_err().to_string();
        std::fs::remove_file(&p).unwrap();
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn argsort_orders_by_key_not_position() {
        let keys: Vec<String> = ["TTT", "AAA", "GGG", "CCC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let perm = argsort(&keys);
        assert_eq!(perm, vec![1, 3, 2, 0]);
        let applied: Vec<&String> = perm.iter().map(|&i| &keys[i]).collect();
        assert_eq!(applied, vec!["AAA", "CCC", "GGG", "TTT"]);
    }

    #[test]
    fn argsort_of_sorted_input_is_the_identity() {
        let keys: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(argsort(&keys), vec![0, 1, 2]);
    }

    #[test]
    fn multiplex_barcodes_split_from_right() {
        let obs = vec![
            "sample_one_AAACCTGAGT".to_string(),
            "sample_two_TTGGAACCGA".to_string(),
        ];
        let (sample_names, cell_barcodes) =
            parse_multiplex_barcodes(&obs).expect("should parse multiplex barcodes");
        assert_eq!(sample_names, vec!["sample_one", "sample_two"]);
        assert_eq!(cell_barcodes, vec!["AAACCTGAGT", "TTGGAACCGA"]);
    }
}
