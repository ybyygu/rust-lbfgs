// rosenbrock
// :PROPERTIES:
// :header-args: :tangle benches/rosenbrock.rs
// :END:

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*rosenbrock][rosenbrock:1]]
use quicli::prelude::*;
type Result<T> = ::std::result::Result<T, Error>;

#[macro_use]
extern crate criterion;

use criterion::Criterion;
use lbfgs::LBFGS;

// Default evaluator adopted from liblbfgs sample.c
//
// # Parameters
// - arr_x: The current values of variables.
// - gx   : The gradient vector. The callback function must compute the gradient values for the current variables.
// # Return
// - fx: evaluated value
fn evaluate(arr_x: &[f64], gx: &mut [f64]) -> Result<f64> {
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
}

// Initialize the variables
fn init_variables(n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).step_by(2) {
        x[i] = -1.2;
        x[i+1] = 1.0;
    }

    x
}

fn rosenbrock() {
    const N: usize = 100;

    let mut x = init_variables(N);

    let mut lbfgs = LBFGS::default();
    lbfgs.run(&mut x, evaluate, |_| false).expect("lbfgs run");
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("rosenbrock lbfgs", |b| b.iter(|| rosenbrock()));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
// rosenbrock:1 ends here
