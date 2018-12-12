// base

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*base][base:1]]
#![allow(nonstandard_style)]

use quicli::prelude::*;
type Result<T> = ::std::result::Result<T, Error>;

pub mod lbfgs;
pub mod math;
pub mod line;
pub use crate::lbfgs::*;
// base:1 ends here

// lbfgs

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*lbfgs][lbfgs:1]]
#[derive(Debug, Clone)]
pub struct LBFGS<F, G>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    G: FnMut(&Progress) -> bool,
{
    pub param: LbfgsParam,
    evaluate: Option<F>,
    progress: Option<G>,
}

impl<F, G> Default for LBFGS<F, G>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    G: FnMut(&Progress) -> bool,
{
    fn default() -> Self {
        LBFGS {
            param: LbfgsParam::default(),
            evaluate: None,
            progress: None,
        }
    }
}

/// Create lbfgs optimizer with epsilon convergence
impl<F, G> LBFGS<F, G>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    G: FnMut(&Progress) -> bool,
{
    pub fn new(epsilon: f64) -> Self {
        assert!(epsilon.is_sign_positive());

        let mut lbfgs = LBFGS::default();

        lbfgs.param.epsilon = epsilon;

        lbfgs
    }
}

impl<F, G> LBFGS<F, G>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    G: FnMut(&Progress) -> bool,
{
    /// Start the L-BFGS optimization; this will invoke the callback functions
    /// evaluate() and progress() when necessary.
    ///
    /// # Parameters
    /// - x      : The array of variables, which will be updated during optimization.
    /// - eval_fn: A closure to evaluate x
    /// - prgr_fn: A closure to monitor progress
    pub fn run(&mut self, x: &mut [f64], eval_fn: F, prgr_fn: G) -> Result<f64> {
        let progress = Some(prgr_fn);

        // call lbfgs function
        let mut fx = 0.0;
        lbfgs(
            x,
            &mut fx,
            eval_fn,
            progress,
            &self.param,
        )?;

        Ok(fx)
    }
}
// lbfgs:1 ends here
