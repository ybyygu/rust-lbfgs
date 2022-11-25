//  Copyright (c) 1990, Jorge Nocedal
//  Copyright (c) 2007-2010 Naoaki Okazaki
//  Copyright (c) 2018-2022 Wenping Guo
//  All rights reserved.
//
//! Limited memory BFGS (L-BFGS) algorithm ported from liblbfgs
//!
//! # Example
//! ```
//! // 0. Import the lib
//! use liblbfgs::lbfgs;
//!
//! const N: usize = 100;
//!
//! // 1. Initialize data
//! let mut x = [0.0 as f64; N];
//! for i in (0..N).step_by(2) {
//!     x[i] = -1.2;
//!     x[i + 1] = 1.0;
//! }
//!
//! // 2. Defining how to evaluate function and gradient
//! let evaluate = |x: &[f64], gx: &mut [f64]| {
//!     let n = x.len();
//!
//!     let mut fx = 0.0;
//!     for i in (0..n).step_by(2) {
//!         let t1 = 1.0 - x[i];
//!         let t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
//!         gx[i + 1] = 20.0 * t2;
//!         gx[i] = -2.0 * (x[i] * gx[i + 1] + t1);
//!         fx += t1 * t1 + t2 * t2;
//!     }
//!
//!     Ok(fx)
//! };
//!
//! let prb = lbfgs()
//!     .with_max_iterations(5)
//!     //.with_orthantwise(1.0, 0, 99) // enable OWL-QN algorithm
//!     //.with_orthantwise(1.0, 0, None) // with end parameter auto determined
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

mod core;
mod lbfgs;
mod orthantwise;

mod common {
    pub use anyhow::*;
    pub use log::{debug, error, info, trace, warn};
}

use crate::common::*;
use crate::core::*;

pub mod line;
pub mod math;
pub use crate::core::{Problem, Progress, Report};
pub use crate::lbfgs::Lbfgs;
pub use crate::orthantwise::*;

/// Create a default LBFGS optimizer.
pub fn lbfgs() -> Lbfgs {
    Lbfgs::default()
}

/// Default test function (rosenbrock) adopted from liblbfgs sample.c
pub fn default_evaluate() -> impl FnMut(&[f64], &mut [f64]) -> Result<f64> {
    move |arr_x: &[f64], gx: &mut [f64]| {
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
    }
}

/// Default progress monitor adopted from liblbfgs sample.c
///
/// # Notes
///
/// * Returning true will cancel the optimization process.
///
pub fn default_progress() -> impl FnMut(&Progress) -> bool {
    move |prgr| {
        println!("Iteration {}, Evaluation {}:", prgr.niter, prgr.neval);
        println!(
            " fx = {:-12.6} xnorm = {:-12.6}, gnorm = {:-12.6}, ls = {}, step = {}",
            prgr.fx, prgr.xnorm, prgr.gnorm, prgr.ncall, prgr.step
        );

        false
    }
}
