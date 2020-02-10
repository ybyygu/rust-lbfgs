// sample.rs
// :PROPERTIES:
// :header-args: :tangle examples/sample.rs
// :END:
// Adopted from sample.c in original source.

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*sample.rs][sample.rs:1]]
use liblbfgs::{lbfgs, Progress};

fn main() {
    const N: usize = 100;

    // Initialize the variables
    let mut x = [0.0 as f64; N];
    for i in (0..N).step_by(2) {
        x[i] = -1.2;
        x[i + 1] = 1.0;
    }

    // Default evaluator adopted from liblbfgs sample.c
    //
    // # Parameters
    // - x: The current values of variables.
    // - gx   : The gradient vector. The callback function must compute the gradient values for the current variables.
    // # Return
    // - fx: evaluated value
    let evaluate = |x: &[f64], gx: &mut [f64]| {
        let n = x.len();

        let mut fx = 0.0;
        for i in (0..n).step_by(2) {
            let t1 = 1.0 - x[i];
            let t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
            gx[i + 1] = 20.0 * t2;
            gx[i] = -2.0 * (x[i] * gx[i + 1] + t1);
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
        let x = &prgr.x;

        println!("Iteration {}:", &prgr.niter);
        println!("  fx = {}, x[0] = {}, x[1] = {}", &prgr.fx, x[0], x[1]);
        println!(
            "  xnorm = {}, gnorm = {}, step = {}",
            &prgr.xnorm, &prgr.gnorm, &prgr.step
        );
        println!("");

        false
    };

    let prb = lbfgs()
        .minimize(&mut x, evaluate, progress)
        .expect("lbfgs minimize");

    println!("  fx = {:}, x[0] = {}, x[1] = {}\n", prb.fx, x[0], x[1]);
}
// sample.rs:1 ends here
