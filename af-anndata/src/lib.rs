use anndata::{reader::MMReader, s, AnnData, AnnDataOp, ArrayData, ArrayElemOp};
use anndata_hdf5::H5;
use anyhow::Context;
use polars::io::prelude::*;
use polars::prelude::{CsvReadOptions, DataFrame};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tracing::info;

fn separate_usa_layers<B: anndata::Backend>(
    b: AnnData<B>,
    row_df: DataFrame,
    col_df: DataFrame,
) -> anyhow::Result<()> {
    let mut sw = libsw::Sw::new();
    sw.start()?;

    let nr = b.n_obs();
    let nc = b.n_vars();
    // if USA mode then the number of genes is
    // 1/3 of the number of features
    let ngenes = nc / 3;

    // Get the unspliced, spliced and ambiguous slices
    let vars = col_df;

    let slice1: ArrayData = b.get_x().slice(s![.., 0..ngenes])?.unwrap();

    match slice1 {
        anndata::data::array::ArrayData::Array(ref a) => info!("Array"),
        anndata::data::array::ArrayData::CsrMatrix(ref a) => info!("CSR"),
        anndata::data::array::ArrayData::CscMatrix(ref a) => info!("CSC"),
        anndata::data::array::ArrayData::CsrNonCanonical(ref a) => info!("CSR Non-canonical"),
        anndata::data::array::ArrayData::DataFrame(ref a) => info!("DataFrame"),
    }

    let var1 = vars.slice(0_i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice2: ArrayData = b.get_x().slice(s![.., ngenes..2 * ngenes])?.unwrap();
    let var2 = vars.slice(ngenes as i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    let slice3: ArrayData = b.get_x().slice(s![.., 2 * ngenes..3 * ngenes])?.unwrap();
    let var3 = vars.slice(2_i64 * ngenes as i64, ngenes);
    info!("getting slice took {:#?}", sw.elapsed());
    sw.reset();
    sw.start()?;

    // We must have an X, but don't want to waste space on it
    // so set it as and empty matrix
    let csr_zero = nalgebra_sparse::CsrMatrix::<f64>::zeros(nr, ngenes);
    let csr_zero = anndata::data::array::DynCsrMatrix::F64(csr_zero);
    // get rid of the old X
    b.del_x().context("unable to delete X")?;
    b.del_obs()?;
    b.set_n_obs(nr).context("unable to set n_obs")?;
    b.set_n_vars(ngenes).context("unable to set n_vars")?;
    // set the new X
    b.set_x(csr_zero).context("unable to set all 0s X")?;

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
    let mut json_path = PathBuf::from(&root_path);
    json_path.push("quant.json");

    let alevin_path = root_path.join("alevin");
    let mut p = PathBuf::from(&alevin_path);
    p.push("quants_mat.mtx");

    let mut colpath = PathBuf::from(&alevin_path);
    colpath.push("quants_mat_cols.txt");

    let mut rowpath = PathBuf::from(&alevin_path);
    rowpath.push("quants_mat_rows.txt");

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
    if !json_path.is_file() {
        anyhow::bail!(
            "the json file was expected at {} but could not be found",
            json_path.display()
        );
    }

    let jf = std::fs::File::open(&json_path)?;
    let quant_json: Value = serde_json::from_reader(jf)
        .with_context(|| format!("could not parse {} as valid JSON.", json_path.display()))?;

    let usa_mode = if let Some(Value::Bool(v)) = quant_json.get("usa_mode") {
        *v
    } else {
        false
    };

    info!("USA mode : {}", usa_mode);

    let mut sw = libsw::Sw::new();
    sw.start()?;

    let r = MMReader::from_path(&p)?;

    // read the gene symbols
    let mut col_df = CsvReadOptions::default()
        .with_has_header(false)
        .try_into_reader_with_file_path(Some(colpath))?
        .finish()?;
    col_df.set_column_names(["gene_symbols"])?;

    // read the barcodes
    let mut row_df = CsvReadOptions::default()
        .with_has_header(false)
        .try_into_reader_with_file_path(Some(rowpath))?
        .finish()?;
    row_df.set_column_names(["barcodes"])?;

    // make the AnnData object and populate it from the MMReader
    let b = AnnData::<H5>::new(output_path.as_ref())?;
    r.finish(&b)?;
    info!("Reading MM into AnnData took {:#?}", sw.elapsed());

    // read in the feature dump data
    let mut feat_dump_path = PathBuf::from(&root_path);
    feat_dump_path.push("featureDump.txt");
    let feat_parse_options =
        polars::io::csv::read::CsvParseOptions::default().with_separator(b'\t');
    let feat_dump_frame = polars_io::csv::read::CsvReadOptions::default()
        .with_parse_options(feat_parse_options)
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(feat_dump_path))
        .context("could not create TSV file reader")?
        .finish()
        .context("could not parse feature TSV file")?;
    // add the features to the row df
    // skip the first column since it is `CB` (the cell barcode) and is
    // redundant with the cell barcode we already have in this dataframe
    let row_df = row_df.hstack(&feat_dump_frame.take_columns()[1..])?;

    // read in the generate_permit_list JSON file
    let mut gpl_path = PathBuf::from(&root_path);
    gpl_path.push("generate_permit_list.json");
    let jf = std::fs::File::open(&gpl_path)?;
    let gpl_json: Value = serde_json::from_reader(jf)
        .with_context(|| format!("could not parse {} as valid JSON.", gpl_path.display()))?;

    // set unstructured metadata
    let quant_json_str = serde_json::to_string(&quant_json)
        .context("could not convert quant.json to string succesfully to place in uns data.")?;
    let gpl_json_str = serde_json::to_string(&gpl_json).context(
        "could not convert generate_permit_list.json to string succesfully to place in uns data.",
    )?;

    let uns: Vec<(String, anndata::Data)> = vec![
        ("quant_info".to_owned(), anndata::Data::from(quant_json_str)),
        ("gpl_info".to_owned(), anndata::Data::from(gpl_json_str)),
    ];
    b.set_uns(uns).context("failed to set \"uns\" data")?;

    if usa_mode {
        separate_usa_layers(b, row_df, col_df)?;
    } else {
        b.set_var(col_df)?;
        b.set_obs(row_df)?;
    }
    Ok(())
}
