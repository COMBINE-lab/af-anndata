# af-anndata

[![crates.io](https://img.shields.io/crates/v/af-anndata.svg)](https://crates.io/crates/af-anndata)

A Rust library for converting [alevin-fry] / [simpleaf] single-cell and
single-nucleus quantification output into the [AnnData] (`.h5ad`) format.

For the command-line tool built on this library, see [`convert-af`](../convert-af).

## What it does

Point it at an `af_quant` directory and it produces one HDF5-backed AnnData file:

| Input | Becomes |
|---|---|
| `alevin/quants_mat.mtx` | `X` (and, in USA mode, the layers below) |
| `alevin/quants_mat_rows.txt` | the `obs` index (cell barcodes) |
| `alevin/quants_mat_cols.txt` | the `var` index (gene ids) |
| `gene_id_to_name.tsv` | `var["gene_symbol"]`, if present |
| `featureDump.txt` | per-cell QC metrics in `obs` |
| `*.json` run metadata | `uns` entries |

In USA mode the feature axis holds unspliced, spliced and ambiguous counts for
each gene. These are split into the `unspliced`, `spliced` and `ambiguous`
layers, and `X` is set to their sum, over a `var` axis of just the genes.

Multiplexed runs, whose barcodes look like `<sample_name>_<CB>`, are split into
separate `sample_name` and `cell_barcode` columns in `obs`.

## Installing

```toml
[dependencies]
af-anndata = "0.5"
```

## Usage

```rust
use af_anndata::{convert_csr_to_anndata, convert_csr_to_anndata_with_opts, ConvertOpts};
use std::path::Path;

# fn main() -> anyhow::Result<()> {
convert_csr_to_anndata(Path::new("af_quant"), Path::new("out.h5ad"))?;

// Or, to get output that is directly comparable across runs:
convert_csr_to_anndata_with_opts(
    Path::new("af_quant"),
    Path::new("out.h5ad"),
    ConvertOpts { sort_index: true },
)?;
# Ok(())
# }
```

`sort_index` sorts `obs` by barcode and `var` by gene id. It is a pure
permutation: `X`, every layer, and the `obs`/`var` annotations move together.

## Keeping the dependency versions aligned

`anndata` does **not** re-export `polars` or `nalgebra-sparse`, but both cross its
API boundary — `DataFrame`s go into `set_obs`/`set_var`, and `CsrMatrix` comes out
of `DynCsrMatrix`. This crate must therefore declare *exactly* the versions
`anndata` itself requires, currently `polars 0.53` and `nalgebra-sparse 0.11`.

Upgrading either past what `anndata` wants resolves two copies of the crate and
fails with the memorable ``cannot add &CsrMatrix<i32> to CsrMatrix<i32>``, where
the two types are nominally identical (this was [#2]). CI builds without a
lockfile specifically to catch it. If that job fails, realign the pins — do not
relax them.

### A known upstream landmine

`hdf5-metno-src` 0.10.3 vendors HDF5 2.2.0, but `hdf5-metno-sys` 0.11.3's build
script only accepts HDF5 minor versions `{0,1,8,10,12,14}` and panics with
`Invalid H5_VERSION: "2.2.0"`. The committed `Cargo.lock` therefore holds
`hdf5-metno-src` at 0.10.2. This only bites builds that compile HDF5 from source;
if you have a system HDF5, `hdf5-metno-sys` links it and never gets there.

## Related projects

- [convert-af](../convert-af) — the CLI wrapper.
- [simpleaf] — calls this library at the end of its quantification workflow.

## License

BSD 3-Clause; see [LICENSE](LICENSE).

[alevin-fry]: https://github.com/COMBINE-lab/alevin-fry
[simpleaf]: https://github.com/COMBINE-lab/simpleaf
[AnnData]: https://anndata.readthedocs.io/en/stable/
[#2]: https://github.com/COMBINE-lab/af-anndata/issues/2
