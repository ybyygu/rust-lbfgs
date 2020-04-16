// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*imports][imports:1]]
use crate::core::*;

use crate::lbfgs::*;
// imports:1 ends here

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*traits][traits:1]]
pub trait EvaluateLbfgs {
    /// Evaluate gradient `gx` at positions `x`, returns function value on success.
    fn evaluate<'a, 'b>(&mut self, x: &'a [f64], gx: &'b mut [f64]) -> Result<f64>;
}

impl<T> EvaluateLbfgs for T
where
    T: Send + Sync + for<'a, 'b> FnMut(&'a [f64], &'b mut [f64]) -> Result<f64>
{
    /// Evaluate gradient `gx` at positions `x`, returns function value on success.
    fn evaluate<'a, 'b>(&mut self, x: &'a [f64], gx: &'b mut [f64]) -> Result<f64> {
        self(x, gx)
    }
}
// traits:1 ends here

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*base][base:1]]
use crate::lbfgs::IterationData;
use crate::lbfgs::Problem;

#[derive(Default)]
pub struct LbfgsState<'a> {
    /// LBFGS parameters
    pub(crate) vars: LbfgsParam,

    /// Define how to evaluate gradient and value
    pub(crate) prbl: Option<Problem<'a>>,
    pub(crate) end: usize,
    pub(crate) step: f64,
    pub(crate) k: usize,
    pub(crate) lm_arr: Vec<IterationData>,
    pub(crate) pf: Vec<f64>,
    pub(crate) ncall: usize,
}
// base:1 ends here
