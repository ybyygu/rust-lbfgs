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
pub fn lbfgs<F, G>() -> LBFGS<F, G>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    G: FnMut(&Progress) -> bool,
{
    LBFGS::default()
}
// lbfgs:1 ends here
