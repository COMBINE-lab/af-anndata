use af_anndata::convert_csr_to_anndata;
use std::env;
use tracing_subscriber::fmt;

fn main() -> anyhow::Result<()> {
    fmt::fmt().init();
    let d = env::args().nth(1).expect("input expected");
    let opath = env::args().nth(2).expect("input expected");
    let p = std::path::Path::new(&d);
    let o = std::path::Path::new(&opath);
    convert_csr_to_anndata(p, o)
}
