// build.rs
// :PROPERTIES:
// :header-args: :tangle build.rs
// :END:

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*build.rs][build.rs:1]]
use bindgen;
use cc;

use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new()
        .cpp(false)
        .file("lib/lbfgs.c")
        .include("lib")
        .compile("liblbfgs.a");

    // println!("cargo:rustc-link-lib=lbfgs");

    let bindings = bindgen::Builder::default()
        .header("lib/lbfgs.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
// build.rs:1 ends here
