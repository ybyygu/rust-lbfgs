//! core data structures for L-BFGS algorithm

use crate::common::*;
use crate::math::*;
use crate::orthantwise::*;

/// Represents an optimization problem.
///
/// `Problem` holds input variables `x`, gradient `gx` arrays, and function value `fx`.
pub struct Problem<'a, E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// x is an array of length n. on input it must contain the base point for
    /// the line search.
    pub(crate) x: &'a mut [f64],

    /// `fx` is a variable. It must contain the value of problem `f` at
    /// x.
    pub(crate) fx: f64,

    /// `gx` is an array of length n. It must contain the gradient of `f` at
    /// x.
    pub(crate) gx: Vec<f64>,

    /// Cached position vector of previous step.
    pub(crate) xp: Vec<f64>,

    /// Cached gradient vector of previous step.
    pub(crate) gp: Vec<f64>,

    /// Pseudo gradient for OrthantWise Limited-memory Quasi-Newton (owlqn) algorithm.
    pg: Vec<f64>,

    /// For owlqn projection
    wp: Vec<f64>,

    /// Search direction
    d: Vec<f64>,

    /// Store callback function for evaluating objective function.
    eval_fn: E,

    /// Orthantwise operations
    owlqn: Option<Orthantwise>,

    /// Evaluated or not
    evaluated: bool,

    /// The number of evaluation.
    neval: usize,
}

impl<'a, E> Problem<'a, E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// Initialize problem with array length n
    pub fn new(x: &'a mut [f64], eval: E, owlqn: Option<Orthantwise>) -> Self {
        let n = x.len();
        Problem {
            fx: 0.0,
            gx: vec![0.0; n],
            xp: vec![0.0; n],
            gp: vec![0.0; n],
            pg: vec![0.0; n],
            wp: vec![0.0; n],
            d: vec![0.0; n],
            evaluated: false,
            neval: 0,
            x,
            eval_fn: eval,
            owlqn,
        }
    }

    /// Compute the initial gradient in the search direction.
    pub fn dginit(&self) -> Result<f64> {
        if self.owlqn.is_none() {
            let dginit = self.gx.vecdot(&self.d);
            if dginit > 0.0 {
                warn!(
                    "The current search direction increases the objective function value. dginit = {:-0.4}",
                    dginit
                );
            }

            Ok(dginit)
        } else {
            Ok(self.pg.vecdot(&self.d))
        }
    }

    /// Update search direction using evaluated gradient.
    pub fn update_search_direction(&mut self) {
        if self.owlqn.is_some() {
            self.d.vecncpy(&self.pg);
        } else {
            self.d.vecncpy(&self.gx);
        }
    }

    /// Return a reference to current search direction vector
    pub fn search_direction(&self) -> &[f64] {
        &self.d
    }

    /// Return a mutable reference to current search direction vector
    pub fn search_direction_mut(&mut self) -> &mut [f64] {
        &mut self.d
    }

    /// Compute the gradient in the search direction without sign checking.
    pub fn dg_unchecked(&self) -> f64 {
        self.gx.vecdot(&self.d)
    }

    // FIXME: improve
    pub fn evaluate(&mut self) -> Result<()> {
        self.fx = (self.eval_fn)(&self.x, &mut self.gx)?;

        // Compute the L1 norm of the variables and add it to the object value.
        if let Some(owlqn) = self.owlqn {
            self.fx += owlqn.x1norm(&self.x)
        }

        self.evaluated = true;
        self.neval += 1;

        Ok(())
    }

    /// Return total number of evaluations.
    pub fn number_of_evaluation(&self) -> usize {
        self.neval
    }

    /// Test if `Problem` has been evaluated or not
    pub fn evaluated(&self) -> bool {
        self.evaluated
    }

    /// Copies all elements from src into self.
    pub fn clone_from(&mut self, src: &Problem<E>) {
        self.x.clone_from_slice(&src.x);
        self.gx.clone_from_slice(&src.gx);
        self.fx = src.fx;
    }

    /// Take a line step along search direction.
    ///
    /// Compute the current value of x: x <- x + (*step) * d.
    ///
    pub fn take_line_step(&mut self, step: f64) {
        self.x.veccpy(&self.xp);
        self.x.vecadd(&self.d, step);

        // Choose the orthant for the new point.
        // The current point is projected onto the orthant.
        if let Some(owlqn) = self.owlqn {
            owlqn.constraint_line_search(&mut self.x, &self.wp);
        }
    }

    pub fn fix_orthant_new_point(&mut self) {
        // follow the mathematical definition
        fn signum(x: f64) -> f64 {
            if x.is_nan() || x == 0.0 {
                0.0
            } else {
                x.signum()
            }
        }
        let n = self.x.len();
        // wp[i] = (xp[i] == 0.) ? -gp[i] : xp[i];
        for i in 0..n {
            let epsilon = if self.xp[i] == 0.0 { signum(-self.gp[i]) } else { signum(self.xp[i]) };
            self.wp[i] = epsilon;
        }
    }

    /// Return gradient vector norm: ||gx||
    pub fn gnorm(&self) -> f64 {
        if self.owlqn.is_some() {
            self.pg.vec2norm()
        } else {
            self.gx.vec2norm()
        }
    }

    /// Return position vector norm: ||x||
    pub fn xnorm(&self) -> f64 {
        self.x.vec2norm()
    }

    pub fn orthantwise(&self) -> bool {
        self.owlqn.is_some()
    }

    /// Revert to previous step
    pub fn revert(&mut self) {
        self.x.veccpy(&self.xp);
        self.gx.veccpy(&self.gp);
    }

    /// Store the current position and gradient vectors.
    pub fn save_state(&mut self) {
        self.xp.veccpy(&self.x);
        self.gp.veccpy(&self.gx);
    }

    /// Constrain the search direction for orthant-wise updates.
    pub fn constrain_search_direction(&mut self) {
        if let Some(owlqn) = self.owlqn {
            owlqn.constrain_search_direction(&mut self.d, &self.pg);
        }
    }

    // FIXME
    pub fn update_owlqn_gradient(&mut self) {
        if let Some(owlqn) = self.owlqn {
            owlqn.compute_pseudo_gradient(&mut self.pg, &self.x, &self.gx);
        }
    }
}

/// Store optimization progress data, for progress monitor
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Progress<'a> {
    /// The current values of variables
    pub x: &'a [f64],

    /// The current gradient values of variables.
    pub gx: &'a [f64],

    /// The current value of the objective function.
    pub fx: f64,

    /// The Euclidean norm of the variables
    pub xnorm: f64,

    /// The Euclidean norm of the gradients.
    pub gnorm: f64,

    /// The line-search step used for this iteration.
    pub step: f64,

    /// The iteration count.
    pub niter: usize,

    /// The total number of evaluations.
    pub neval: usize,

    /// The number of function evaluation calls in line search procedure
    pub ncall: usize,
}

impl<'a> Progress<'a> {
    pub fn new<E>(prb: &'a Problem<E>, niter: usize, ncall: usize, step: f64) -> Self
    where
        E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    {
        Progress {
            x: &prb.x,
            gx: &prb.gx,
            fx: prb.fx,
            xnorm: prb.xnorm(),
            gnorm: prb.gnorm(),
            neval: prb.number_of_evaluation(),
            ncall,
            step,
            niter,
        }
    }
}

#[derive(Debug, Clone)]
/// Represents the final optimization outcome
pub struct Report {
    /// The current value of the objective function.
    pub fx: f64,

    /// The Euclidean norm of the variables
    pub xnorm: f64,

    /// The Euclidean norm of the gradients.
    pub gnorm: f64,

    /// The total number of evaluations.
    pub neval: usize,
}

impl Report {
    pub(crate) fn new<E>(prb: &Problem<E>) -> Self
    where
        E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    {
        Self {
            fx: prb.fx,
            xnorm: prb.xnorm(),
            gnorm: prb.gnorm(),
            neval: prb.number_of_evaluation(),
        }
    }
}
