use anndata::{reader::MMReader, s, AnnData, AnnDataOp, ArrayData, ArrayElemOp};
use anndata_hdf5::H5;
use anyhow::{bail, Context};
use polars::io::prelude::*;
use polars::prelude::{CsvReadOptions, DataFrame, PolarsError, Series, SortMultipleOptions};
use serde_json::Value;
use std::path::{Path, PathBuf};
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
    mut b: AnnData<B>,
    row_df: DataFrame,
    col_df: DataFrame,
    var_df: Option<DataFrame>,
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
    let vars = col_df;

    let slice1: ArrayData = b.get_x().slice(s![.., 0..ngenes])?.unwrap();
    csr_zero = accumulate_layer(&slice1, csr_zero)?;

    let var1 = vars.slice(0_i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice2: ArrayData = b.get_x().slice(s![.., ngenes..2 * ngenes])?.unwrap();
    csr_zero = accumulate_layer(&slice2, csr_zero)?;

    let var2 = vars.slice(ngenes as i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice3: ArrayData = b.get_x().slice(s![.., 2 * ngenes..3 * ngenes])?.unwrap();
    csr_zero = accumulate_layer(&slice3, csr_zero)?;

    let var3 = vars.slice(2_i64 * ngenes as i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    set_x_layer(&mut b, csr_zero)?;

    // populate with the gene id and gene symbol if we have it
    // otherwise just set the gene name
    if let Some(var_info) = var_df {
        b.set_var(var_info)?;
    } else {
        let mut temp_var = var1.clone();
        temp_var.set_column_names(["gene_id"])?;
        b.set_var(temp_var)?;
    }

    let layers = vec![
        ("spliced".to_owned(), slice1),
        ("unspliced".to_owned(), slice2),
        ("ambiguous".to_owned(), slice3),
    ];

    let varm = vec![
        ("spliced".to_owned(), var1),
        ("unspliced".to_owned(), var2),
        ("ambiguous".to_owned(), var3),
    ];
    b.set_layers(layers)
        .context("unable to set layers for AnnData object")?;
    info!("setting layers took {:#?}", sw.elapsed());
    b.set_varm(varm)?;
    b.set_obs(row_df)?;

    Ok(())
}

pub fn convert_csr_to_anndata<P: AsRef<Path>>(root_path: P, output_path: P) -> anyhow::Result<()> {
    let root_path = root_path.as_ref();
    let json_path = PathBuf::from(&root_path);

    let mut gpl_path = json_path.clone();
    gpl_path.push("generate_permit_list.json");

    let mut collate_path = json_path.clone();
    collate_path.push("collate.json");

    let mut quant_path = json_path.clone();
    quant_path.push("quant.json");

    let mut map_log_path = json_path.clone();
    map_log_path.push("simpleaf_map_info.json");

    let alevin_path = root_path.join("alevin");
    let mut p = PathBuf::from(&alevin_path);
    p.push("quants_mat.mtx");

    let mut colpath = PathBuf::from(&alevin_path);
    colpath.push("quants_mat_cols.txt");

    let mut rowpath = PathBuf::from(&alevin_path);
    rowpath.push("quants_mat_rows.txt");

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
    // otherwise, wan the user
    if gene_id_to_name_path.is_none() {
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
    col_df.set_column_names(["gene_id"])?;

    // read the barcodes
    let mut row_df = match CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_opts.clone())
        .with_raise_if_empty(true)
        .try_into_reader_with_file_path(Some(rowpath))?
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

    let nobs_cols = row_df.get_columns().len();
    match nobs_cols {
        1 => row_df.set_column_names(["barcodes"])?,
        3 => row_df.set_column_names(["barcodes", "spot_x", "spot_y"])?,
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

    // read the gene_id_to_name file
    let var_df = if let Some(id_to_name) = gene_id_to_name_path {
        // if we had the gene id to name file name, then we want to read it in, but
        // we also want to re-order the rows to match with the row-labels given in
        // the input file.
        let ordered_ids = col_df.column("gene_id")?.as_materialized_series().str()?;
        // read through the gene names in order, and build a hash map from gene_id to
        // rank (corresponding column) of the count matrix
        let gene_rank_hash = std::collections::HashMap::<&str, u64>::from_iter(
            ordered_ids.iter().enumerate().map(|(i, s)| {
                (
                    s.expect("should not be missing a gene id in the `quants_mat_cols` file"),
                    i as u64,
                )
            }),
        );

        // read the gene id to name file
        let mut vd = CsvReadOptions::default()
            .with_has_header(false)
            .with_parse_options(parse_opts)
            .with_raise_if_empty(true)
            .try_into_reader_with_file_path(Some(id_to_name))?
            .finish()?;

        // create a column that lists the rank for each gene
        // *in the order it appears in the gene id to name list*.
        // we will use this to re-order the rows of the gene id to name
        // dataframe.
        let rank_vec: Series = Series::from_iter(
            vd.select_at_idx(0)
                .expect("0th column of gene id to gene name DataFrame should exist")
                .as_materialized_series()
                .str()?
                .iter()
                .map(|s| {
                    gene_rank_hash
                        [s.expect("should not be a missing gene id in the `gene_id_to_name` file")]
                }),
        );
        // add the rank column to the Data Frame
        vd.with_column(rank_vec)?;
        // set the column names
        vd.set_column_names(["gene_id", "gene_symbol", "gene_rank"])?;
        // reorder the rows to put the ranks in order, bringing the ids and names
        // with them.
        vd.sort_in_place(["gene_rank"], SortMultipleOptions::default())?;
        // we no longer need the rank column
        vd.drop_in_place("gene_rank")?;
        Some(vd)
    } else {
        None
    };

    // make the AnnData object and populate it from the MMReader
    let b = AnnData::<H5>::new(output_path.as_ref())?;
    r.finish(&b)?;
    info!("Reading MM into AnnData took {:#?}", sw.elapsed());

    // read in the feature dump data
    let mut feat_dump_path = PathBuf::from(&root_path);
    feat_dump_path.push("featureDump.txt");
    let feat_parse_options =
        polars::io::csv::read::CsvParseOptions::default().with_separator(b'\t');
    let mut feat_dump_frame = match polars_io::csv::read::CsvReadOptions::default()
        .with_parse_options(feat_parse_options)
        .with_has_header(true)
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
    let row_df = row_df.hstack(&feat_dump_frame.take_columns()[1..])?;

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

    // set unstructured metadata
    let uns: Vec<(String, anndata::Data)> = vec![
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
    b.set_uns(uns).context("failed to set \"uns\" data")?;

    if usa_mode {
        separate_usa_layers(b, row_df, col_df, var_df)?;
    } else {
        if let Some(var_info) = var_df {
            b.set_var(var_info)?;
        } else {
            b.set_var(col_df)?;
        }
        b.set_obs(row_df)?;
    }
    Ok(())
}
