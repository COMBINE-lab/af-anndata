//! `convert-af` converts a [simpleaf]/[alevin-fry] `af_quant` output directory
//! into a single HDF5-backed [AnnData] (`.h5ad`) file.
//!
//! [simpleaf]: https://github.com/COMBINE-lab/simpleaf
//! [alevin-fry]: https://github.com/COMBINE-lab/alevin-fry
//! [AnnData]: https://anndata.readthedocs.io/en/stable/

use af_anndata::{convert_csr_to_anndata_with_opts, convert_probe_csr_to_anndata, ConvertOpts};
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

    /// Convert the probe matrix (`alevin/probe_quants_mat.mtx`, from
    /// `alevin-fry quant-probe --probe-mtx`) in place of the gene matrix
    #[arg(long)]
    probe: bool,

    /// The 10x probe-set CSV for `--probe`. The default is
    /// `<input>/probe_set.csv`
    #[arg(long, requires = "probe", value_name = "CSV")]
    probe_ref: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    fmt::fmt().init();
    let cli = Cli::parse();
    let opts = ConvertOpts {
        sort_index: cli.sort_index,
    };
    if cli.probe {
        convert_probe_csr_to_anndata(
            cli.input.as_path(),
            cli.output.as_path(),
            cli.probe_ref.as_deref(),
            opts,
        )
    } else {
        convert_csr_to_anndata_with_opts(cli.input.as_path(), cli.output.as_path(), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn the_output_is_required_in_both_modes() {
        assert!(Cli::try_parse_from(["convert-af", "in"]).is_err());
        assert!(Cli::try_parse_from(["convert-af", "--probe", "in"]).is_err());
        assert!(Cli::try_parse_from(["convert-af", "--probe", "in", "out.h5ad"]).is_ok());
    }

    #[test]
    fn probe_ref_needs_probe() {
        assert!(Cli::try_parse_from(["convert-af", "--probe-ref", "p.csv", "in", "out"]).is_err());
        let cli =
            Cli::try_parse_from(["convert-af", "--probe", "--probe-ref", "p.csv", "in", "out"])
                .unwrap();
        assert_eq!(cli.probe_ref.unwrap().to_str(), Some("p.csv"));
    }
}
