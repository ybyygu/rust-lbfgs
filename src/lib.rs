// docs

//
//  Copyright (c) 1990, Jorge Nocedal
//  Copyright (c) 2007-2010 Naoaki Okazaki
//  Copyright (c) 2018-2019 Wenping Guo
//  All rights reserved.
//
//! Limited memory BFGS (L-BFGS) algorithm
//!
//! # Example
//! ```
//! // 0. Import the lib
//! use lbfgs::lbfgs;
//! 
//! const N: usize = 100;
//! 
//! // 1. Initialize data
//! let mut x = [0.0 as f64; N];
//! for i in (0..N).step_by(2) {
//!     x[i] = -1.2;
//!     x[i+1] = 1.0;
//! }
//! 
//! // 2. Defining how to evaluate function and gradient
//! let evaluate = |x: &[f64], gx: &mut [f64]| {
//!     let n = x.len();
//! 
//!     let mut fx = 0.0;
//!     for i in (0..n).step_by(2) {
//!         let t1 = 1.0 - x[i];
//!         let t2 = 10.0 * (x[i+1] - x[i] * x[i]);
//!         gx[i+1] = 20.0 * t2;
//!         gx[i] = -2.0 * (x[i] * gx[i+1] + t1);
//!         fx += t1 * t1 + t2 * t2;
//!     }
//! 
//!     Ok(fx)
//! };
//! 
//! // 3. Carry out LBFGS optimization
//! let prb = lbfgs()
//!     .with_max_iterations(5)
//!     .with_orthantwise(1.0, 0, 99) // enable OWL-QN
//!     .minimize(
//!         &mut x,                   // input variables
//!         evaluate,                 // define how to evaluate function
//!         |prgr| {                  // define progress monitor
//!             println!("iter: {:}", prgr.niter);
//!             false                 // returning true will cancel optimization
//!         }
//!     )
//!     .expect("lbfgs owlqn minimize");
//! 
//! println!("fx = {:}", prb.fx);
//! ```

// base

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*base][base:1]]
use quicli::prelude::*;
type Result<T> = ::std::result::Result<T, Error>;

mod lbfgs;
pub mod math;
pub mod line;
pub use crate::lbfgs::*;
// base:1 ends here

// lbfgs

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*lbfgs][lbfgs:1]]
pub fn lbfgs<F>() -> LBFGS<F>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    LBFGS::default()
}
// lbfgs:1 ends here
