//! `convert-af` converts a [simpleaf]/[alevin-fry] `af_quant` output directory
//! into a single HDF5-backed [AnnData] (`.h5ad`) file.
//!
//! [simpleaf]: https://github.com/COMBINE-lab/simpleaf
//! [alevin-fry]: https://github.com/COMBINE-lab/alevin-fry
//! [AnnData]: https://anndata.readthedocs.io/en/stable/

use af_anndata::{convert_csr_to_anndata_with_opts, ConvertOpts};
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::fmt;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The input `af_quant` directory produced by simpleaf / alevin-fry
    input: PathBuf,

    /// The output AnnData (`.h5ad`) file to write
    output: PathBuf,

    /// Sort `obs` and `var` by their index (cell barcode and gene id) before
    /// writing. The output is a pure permutation of the unsorted result, which
    /// makes it directly comparable across runs.
    #[arg(long)]
    sort_index: bool,
}

fn main() -> anyhow::Result<()> {
    fmt::fmt().init();
    let cli = Cli::parse();
    convert_csr_to_anndata_with_opts(
        cli.input.as_path(),
        cli.output.as_path(),
        ConvertOpts {
            sort_index: cli.sort_index,
        },
    )
}
