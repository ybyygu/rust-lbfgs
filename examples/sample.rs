// sample.rs
// :PROPERTIES:
// :header-args: :tangle examples/sample.rs
// :END:
// Adopted from sample.c in original source.

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*sample.rs][sample.rs:1]]
use lbfgs::{LBFGS, Progress};
use quicli::prelude::*;

fn main() {
    const N: usize = 100;

    // Initialize the variables
    let mut x = [0.0 as f64; N];
    for i in (0..N).step_by(2) {
        x[i] = -1.2;
        x[i+1] = 1.0;
    }

    // Default evaluator adopted from liblbfgs sample.c
    //
    // # Parameters
    // - arr_x: The current values of variables.
    // - gx   : The gradient vector. The callback function must compute the gradient values for the current variables.
    // # Return
    // - fx: evaluated value
    let evaluate = |arr_x: &[f64], gx: &mut [f64]| {
        let n = arr_x.len();

        let mut fx = 0.0;
        for i in (0..n).step_by(2) {
            let t1 = 1.0 - arr_x[i];
            let t2 = 10.0 * (arr_x[i+1] - arr_x[i] * arr_x[i]);
            gx[i+1] = 20.0 * t2;
            gx[i] = -2.0 * (arr_x[i] * gx[i+1] + t1);
            fx += t1 * t1 + t2 * t2;
        }

        Ok(fx)
    };

    // Default progress monitor adopted from liblbfgs sample.c
    //
    // # Parameters
    // - prgr: holding all progressive data
    // # Return
    // - false to continue the optimization process. Returning true will cancel the optimization process.
    let progress = |prgr: &Progress| {
        let x = &prgr.arr_x;

        println!("Iteration {}:", &prgr.niter);
        println!("  fx = {}, x[0] = {}, x[1] = {}", &prgr.fx, x[0], x[1]);
        println!("  xnorm = {}, gnorm = {}, step = {}", &prgr.xnorm, &prgr.gnorm, &prgr.step);
        println!("");

        false
    };

    let mut lbfgs = LBFGS::default();
    let fx = lbfgs.run(&mut x, evaluate, progress).expect("lbfgs run");
    println!("  fx = {:}, x[0] = {}, x[1] = {}\n", fx, x[0], x[1]);
}
// sample.rs:1 ends here
