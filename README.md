
# LBFGS

[![Build Status](https://travis-ci.org/ybyygu/rust-lbfgs.svg?branch=master)](https://travis-ci.org/ybyygu/rust-lbfgs)
[![GPL3 licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Fast and safe Rust implementation of LBFGS and OWL-QN algorithms ported from
Naoaki Okazaki's C library [libLBFGS](http://chokkan.org/software/liblbfgs/).

Check [rust-liblbfgs](https://github.com/ybyygu/rust-liblbfgs) for a working wrapper around the original C codes.


# Motivation

-   Bring native LBFGS implementation to Rust community.
-   Learn how a great optimization algorithm is implemented in real world.
-   Learn how to "replace the jet engine while still flying" [URL](http://jensimmons.com/post/jan-4-2017/replacing-jet-engine-while-still-flying)
-   Make it more maintainable with Rust high level abstraction.
-   Improve it to meet my needs for computational chemistry.


# Todo

-   [ ] Parallel with rayon
-   [ ] SIMD support
-   [X] add option to disable line search for gradient only optimization
-   [X] Fix issues inherited from liblbfgs [URL](https://github.com/chokkan/liblbfgs/pulls)


# Features

-   Clean and safe Rust implementation.
-   OWL-QN algorithm.
-   Closure based callback interfaces.
-   Damped L-BFGS algorithm.


# Usage

```rust
// 0. Import the lib
use liblbfgs::lbfgs;

const N: usize = 100;

// 1. Initialize data
let mut x = [0.0 as f64; N];
for i in (0..N).step_by(2) {
    x[i] = -1.2;
    x[i + 1] = 1.0;
}

// 2. Defining how to evaluate function and gradient
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

let prb = lbfgs()
    .with_max_iterations(5)
    // .with_orthantwise(1.0, 0, 99) // enable OWL-QN algorithm
    // .with_orthantwise(1.0, 0, None) // with end parameter auto determined
    .minimize(
        &mut x,                   // input variables
        evaluate,                 // define how to evaluate function
        |prgr| {                  // define progress monitor
            println!("iter: {:}", prgr.niter);
            false                 // returning true will cancel optimization
        }
    )
    .expect("lbfgs owlqn minimize");

println!("fx = {:}", prb.fx);
```

The callback functions are native Rust FnMut closures, possible to
capture/change variables in the environment.

Full codes with comments are available in examples/sample.rs.

Run the example:

    cargo run --example sample
