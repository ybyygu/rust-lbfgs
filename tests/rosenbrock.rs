// rosenbrock
// :PROPERTIES:
// :header-args: :tangle tests/rosenbrock.rs
// :END:
// # The Rosenbrock function

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*rosenbrock][rosenbrock:1]]
//! Multidimensional Rosenbrock test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
//!
//! where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
//
//! # Reference
//! - https://en.wikipedia.org/wiki/Rosenbrock_function

use approx::*;

#[test]
fn test_lbfgs() {
    use lbfgs::{Progress, LBFGS};

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
    // - arr_x: The current values of variables.
    // - gx   : The gradient vector. The callback function must compute the gradient values for the current variables.
    // # Return
    // - fx: evaluated value
    let evaluate = |arr_x: &[f64], gx: &mut [f64]| {
        let n = arr_x.len();

        let mut fx = 0.0;
        for i in (0..n).step_by(2) {
            let t1 = 1.0 - arr_x[i];
            let t2 = 10.0 * (arr_x[i + 1] - arr_x[i] * arr_x[i]);
            gx[i + 1] = 20.0 * t2;
            gx[i] = -2.0 * (arr_x[i] * gx[i + 1] + t1);
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

    let mut lbfgs = LBFGS::default();
    let fx = lbfgs.run(&mut x, evaluate, progress).expect("lbfgs run");
    // Iteration 37:
    // fx = 0.0000000000000012832127771605377, x[0] = 0.9999999960382451, x[1] = 0.9999999917607568
    // xnorm = 9.999999938995018, gnorm = 0.0000009486547293218877, step = 1

    assert_relative_eq!(0.0, fx, epsilon=1e-4);
    for i in 0..N {
        assert_relative_eq!(1.0, x[i], epsilon=1e-4);
    }

    // OWL-QN
    lbfgs.param.orthantwise = true;
    lbfgs.param.orthantwise_c = 1.0;
    lbfgs.param.orthantwise_start = 0;
    lbfgs.param.orthantwise_end = 99;
    let fx = lbfgs.run(&mut x, evaluate, progress).expect("lbfgs run");
    // Iteration 171:
    // fx = 43.50249999999999, x[0] = 0.2500000069348678, x[1] = 0.057500004213084016
    // xnorm = 1.8806931246657475, gnorm = 0.00000112236896804755, step = 1

    assert_relative_eq!(43.5025, fx, epsilon=1e-4);
    assert_relative_eq!(0.2500, x[0], epsilon=1e-4);
    assert_relative_eq!(0.0575, x[1], epsilon=1e-4);
}
// rosenbrock:1 ends here
