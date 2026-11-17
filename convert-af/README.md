# convert-af

[![crates.io](https://img.shields.io/crates/v/convert-af.svg)](https://crates.io/crates/convert-af)

A command-line tool that converts [simpleaf] / [alevin-fry] single-cell,
single-nucleus and spatial quantification output directly into the [AnnData]
(`.h5ad`) format.

It is a thin wrapper over the [`af-anndata`](../af-anndata) library, which
documents the conversion in detail.

## Installing

```bash
cargo install convert-af
```

## Usage

```bash
convert-af <input af_quant directory> <output .h5ad file>
convert-af --probe [--probe-ref <probe-set CSV>] <input af_quant directory> <output .h5ad file>
```

Options:

| Flag | Effect |
|---|---|
| `--sort-index` | Sort `obs` by cell barcode and `var` by gene id before writing. A pure permutation — `X`, every layer and the annotations move together — which makes output directly comparable across runs. |
| `--probe` | Convert the probe matrix (`alevin/probe_quants_mat.mtx`, from `alevin-fry quant-probe --probe-mtx`) in place of the gene matrix. |
| `--probe-ref <CSV>` | The 10x probe-set CSV for `--probe`. The default is `<input>/probe_set.csv`; if neither exists, `convert-af` stops with an error. |
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |

Example:

```bash
convert-af --sort-index results/af_quant sample.h5ad
convert-af --probe --probe-ref Chromium_Mouse_Transcriptome_Probe_Set_v2.0.0_GRCm39-2024-A.csv results/af_quant probe_quants.h5ad
```

Set `RUST_LOG` to control logging verbosity (`RUST_LOG=debug` for more detail).

## What you get

A single HDF5-backed AnnData file with cell barcodes as the `obs` index and gene
ids as the `var` index. For USA-mode quantification, `X` holds the summed counts
and the `unspliced`, `spliced` and `ambiguous` layers hold the components. Per-cell
QC metrics from `featureDump.txt` land in `obs`, and the run's JSON metadata in
`uns`.

With `--probe`, `X` holds the probe counts and `var` is indexed by probe id,
with all the columns of the probe-set CSV, as they are.
`obs` is the same as in the gene file. See the af-anndata README.

## License

BSD 3-Clause; see [LICENSE](LICENSE).

[simpleaf]: https://github.com/COMBINE-lab/simpleaf
[alevin-fry]: https://github.com/COMBINE-lab/alevin-fry
[AnnData]: https://anndata.readthedocs.io/en/stable/
