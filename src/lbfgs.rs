// header

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*header][header:1]]
//!       Limited memory BFGS (L-BFGS).
//
//  Copyright (c) 1990, Jorge Nocedal
//  Copyright (c) 2007-2010 Naoaki Okazaki
//  Copyright (c) 2018-2019 Wenping Guo
//  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//
// This library is a C port of the FORTRAN implementation of Limited-memory
// Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
// The original FORTRAN source code is available at:
// http://www.ece.northwestern.edu/~nocedal/lbfgs.html
//
// The L-BFGS algorithm is described in:
//     - Jorge Nocedal.
//       Updating Quasi-Newton Matrices with Limited Storage.
//       <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
//     - Dong C. Liu and Jorge Nocedal.
//       On the limited memory BFGS method for large scale optimization.
//       <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.
//
// The line search algorithms used in this implementation are described in:
//     - John E. Dennis and Robert B. Schnabel.
//       <i>Numerical Methods for Unconstrained Optimization and Nonlinear
//       Equations</i>, Englewood Cliffs, 1983.
//     - Jorge J. More and David J. Thuente.
//       Line search algorithm with guaranteed sufficient decrease.
//       <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
//       pp. 286-307, 1994.
//
// This library also implements Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
// method presented in:
//     - Galen Andrew and Jianfeng Gao.
//       Scalable training of L1-regularized log-linear models.
//       In <i>Proceedings of the 24th International Conference on Machine
//       Learning (ICML 2007)</i>, pp. 33-40, 2007.

// I would like to thank the original author, Jorge Nocedal, who has been
// distributing the effieicnt and explanatory implementation in an open source
// licence.
// header:1 ends here

// base

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*base][base:1]]
use quicli::prelude::*;
type Result<T> = ::std::result::Result<T, Error>;

use crate::math::LbfgsMath;
use crate::line::*;
// base:1 ends here

// parameters

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*parameters][parameters:1]]
/// L-BFGS optimization parameters.
///
/// Call lbfgs_parameter_t::default() function to initialize parameters to the
/// default values.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct LbfgsParam {
    /// The number of corrections to approximate the inverse hessian matrix.
    ///
    /// The L-BFGS routine stores the computation results of previous \ref m
    /// iterations to approximate the inverse hessian matrix of the current
    /// iteration. This parameter controls the size of the limited memories
    /// (corrections). The default value is 6. Values less than 3 are not
    /// recommended. Large values will result in excessive computing time.
    pub m: usize,

    /// Epsilon for convergence test.
    ///
    /// This parameter determines the accuracy with which the solution is to be
    /// found. A minimization terminates when
    ///
    /// ||g|| < epsilon * max(1, ||x||),
    ///
    /// where ||.|| denotes the Euclidean (L2) norm. The default value is \c
    /// 1e-5.
    pub epsilon: f64,

    /// Distance for delta-based convergence test.
    ///
    /// This parameter determines the distance, in iterations, to compute the
    /// rate of decrease of the objective function. If the value of this
    /// parameter is zero, the library does not perform the delta-based
    /// convergence test.
    ///
    /// The default value is 0.
    pub past: usize,

    /// Delta for convergence test.
    ///
    /// This parameter determines the minimum rate of decrease of the objective
    /// function. The library stops iterations when the following condition is
    /// met: |f' - f| / f < delta, where f' is the objective value of past
    /// iterations ago, and f is the objective value of the current iteration.
    /// The default value is 1e-5.
    ///
    pub delta: f64,

    /// The maximum number of LBFGS iterations.
    ///
    /// The lbfgs optimization terminates when the iteration count exceedes this
    /// parameter.
    ///
    /// Setting this parameter to zero continues an optimization process until a
    /// convergence or error. The default value is 0.
    pub max_iterations: usize,

    /// The maximum allowed number of evaluations of function value and
    /// gradients. This number could be larger than max_iterations since line
    /// search procedure may involve one or more evaluations.
    ///
    /// Setting this parameter to zero continues an optimization process until a
    /// convergence or error. The default value is 0.
    pub max_evaluations: usize,

    /// The line search options.
    ///
    ///  This parameter specifies a line search algorithm to be used by the
    ///  L-BFGS routine.
    ///
    pub linesearch: LineSearch,

    /// Enable OWL-QN regulation or not
    pub orthantwise: bool,

    // FIXME: better name
    pub owlqn: Orthantwise,
}

impl Default for LbfgsParam {
    /// Initialize L-BFGS parameters to the default values.
    ///
    /// Call this function to fill a parameter structure with the default values
    /// and overwrite parameter values if necessary.
    fn default() -> Self {
        LbfgsParam {
            m: 6,
            epsilon: 1e-5,
            past: 0,
            delta: 1e-5,
            max_iterations: 0,
            max_evaluations: 0,
            orthantwise: false,
            owlqn: Orthantwise::default(),
            linesearch: LineSearch::default(),
        }
    }
}
// parameters:1 ends here

// problem

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*problem][problem:1]]
/// Represents an optimization problem.
///
/// `Problem` holds input variables `x`, gradient `gx` arrays, and function value `fx`.
#[derive(Debug)]
pub struct Problem<'a, E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// x is an array of length n. on input it must contain the base point for
    /// the line search.
    pub x: &'a mut [f64],

    /// `fx` is a variable. It must contain the value of problem `f` at
    /// x.
    pub fx: f64,

    /// `gx` is an array of length n. It must contain the gradient of `f` at
    /// x.
    pub gx: Vec<f64>,

    /// Cached position vector of previous step.
    xp: Vec<f64>,

    /// Cached gradient vector of previous step.
    gp: Vec<f64>,

    /// Pseudo gradient for OrthantWise Limited-memory Quasi-Newton (owlqn) algorithm.
    pg: Vec<f64>,

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
    pub fn new(x: &'a mut [f64], eval_fn: E, owlqn: Option<Orthantwise>) -> Self {
        let n = x.len();
        Problem {
            fx: 0.0,
            gx: vec![0.0; n],
            xp: vec![0.0; n],
            gp: vec![0.0; n],
            pg: vec![0.0; n],
            d: vec![0.0; n],
            evaluated: false,
            neval: 0,
            x,
            eval_fn,
            owlqn,
        }
    }

    /// Compute the initial gradient in the search direction.
    pub fn dginit(&self) -> Result<f64> {
        if self.owlqn.is_none() {
            let dginit = self.gx.vecdot(&self.d);
            if dginit > 0.0 {
                warn!("The current search direction increases the objective function value. dginit = {:-0.4}", dginit);
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

        // FIXME: to be better
        // if self.orthantwise {
        // Compute the L1 norm of the variable and add it to the object value.
        // fx += self.owlqn.x1norm(x);
        // self.owlqn.pseudo_gradient(&mut pg, &x, &g);
        // }

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
            owlqn.project(&mut self.x, &self.xp, &self.gp);
        }
    }

    /// Compute the initial step
    pub fn initial_step(&self) -> f64 {
        self.d.vec2norminv()
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
    fn save_state(&mut self) {
        self.xp.veccpy(&self.x);
        self.gp.veccpy(&self.gx);
    }

    /// Constrain the search direction for orthant-wise updates.
    fn constrain_search_direction(&mut self) {
        if let Some(owlqn) = self.owlqn {
            owlqn.constrain(&mut self.d, &self.pg);
        }
    }

    // FIXME
    fn update_owlqn_gradient(&mut self) {
        if let Some(owlqn) = self.owlqn {
            owlqn.pseudo_gradient(&mut self.pg, &self.x, &self.gx);
        }
    }
}
// problem:1 ends here

// progress

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*progress][progress:1]]
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
    fn new<E>(prb: &'a Problem<E>, niter: usize, ncall: usize, step: f64) -> Self
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
// progress:1 ends here

// orthantwise

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*orthantwise][orthantwise:1]]
/// Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) algorithm
#[derive(Copy, Clone, Debug)]
pub struct Orthantwise {
    /// Coeefficient for the L1 norm of variables.
    ///
    ///  Setting this parameter to a positive value activates Orthant-Wise
    ///  Limited-memory Quasi-Newton (OWL-QN) method, which minimizes the
    ///  objective function F(x) combined with the L1 norm |x| of the variables,
    ///  {F(x) + C |x|}. This parameter is the coeefficient for the |x|, i.e.,
    ///  C. As the L1 norm |x| is not differentiable at zero, the library
    ///  modifies function and gradient evaluations from a client program
    ///  suitably; a client program thus have only to return the function value
    ///  F(x) and gradients G(x) as usual. The default value is 1.
    pub c: f64,

    /// Start index for computing L1 norm of the variables.
    ///
    /// This parameter is valid only for OWL-QN method (i.e., orthantwise_c !=
    /// 0). This parameter b (0 <= b < N) specifies the index number from which
    /// the library computes the L1 norm of the variables x,
    ///
    /// |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
    ///
    /// In other words, variables x_1, ..., x_{b-1} are not used for computing
    /// the L1 norm. Setting b (0 < b < N), one can protect variables, x_1, ...,
    /// x_{b-1} (e.g., a bias term of logistic regression) from being
    /// regularized. The default value is zero.
    pub start: i32,

    /// End index for computing L1 norm of the variables.
    ///
    /// This parameter is valid only for OWL-QN method (i.e., \ref orthantwise_c
    /// != 0). This parameter e (0 < e <= N) specifies the index number at which
    /// the library stops computing the L1 norm of the variables x,
    pub end: i32,
}

impl Default for Orthantwise {
    fn default() -> Self {
        Orthantwise {
            c: 1.0,
            start: 0,
            end: -1,
        }
    }
}

impl Orthantwise {
    // FIXME: remove
    // a dirty wrapper for start and end
    fn start_end(&self, x: &[f64]) -> (usize, usize) {
        let start = self.start as usize;
        let end = if self.end < 0 {
            x.len()
        } else {
            self.end as usize
        };

        (start, end)
    }

    /// Compute the L1 norm of the variables.
    fn x1norm(&self, x: &[f64]) -> f64 {
        let (start, end) = self.start_end(x);

        let mut s = 0.0;
        for i in start..end {
            s += self.c * x[i].abs();
        }

        s
    }

    /// Compute the psuedo-gradients.
    fn pseudo_gradient(&self, pg: &mut [f64], x: &[f64], g: &[f64]) {
        let (start, end) = self.start_end(x);
        let c = self.c;

        // Compute the negative of gradients.
        for i in 0..start {
            pg[i] = g[i];
        }

        // Compute the psuedo-gradients.
        for i in start..end {
            if x[i] < 0.0 {
                // Differentiable.
                pg[i] = g[i] - c;
            } else if 0.0 < x[i] {
                pg[i] = g[i] + c;
            } else {
                if g[i] < -c {
                    // Take the right partial derivative.
                    pg[i] = g[i] + c;
                } else if c < g[i] {
                    // Take the left partial derivative.
                    pg[i] = g[i] - c;
                } else {
                    pg[i] = 0.;
                }
            }
        }

        for i in end..g.len() {
            pg[i] = g[i];
        }
    }

    /// Choose the orthant for the new point.
    ///
    /// During the line search, each search point is projected onto the orthant
    /// of the previous point.
    fn project(&self, x: &mut [f64], xp: &[f64], gp: &[f64]) {
        let (start, end) = self.start_end(xp);

        for i in start..end {
            let sign = if xp[i] == 0.0 { -gp[i] } else { xp[i] };
            if x[i] * sign <= 0.0 {
                x[i] = 0.0
            }
        }
    }

    fn constrain(&self, d: &mut [f64], pg: &[f64]) {
        let (start, end) = self.start_end(pg);

        for i in start..end {
            if d[i] * pg[i] >= 0.0 {
                d[i] = 0.0;
            }
        }
    }
}
// orthantwise:1 ends here

// builder

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*builder][builder:1]]
#[derive(Debug, Clone)]
pub struct LBFGS<F>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    // FIXME: make it private
    pub param: LbfgsParam,
    evaluate: Option<F>,
}

impl<F> Default for LBFGS<F>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    fn default() -> Self {
        LBFGS {
            param: LbfgsParam::default(),
            evaluate: None,
        }
    }
}

/// Create lbfgs optimizer with epsilon convergence
impl<F> LBFGS<F>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// Set scaled gradient norm for converence test
    ///
    /// This parameter determines the accuracy with which the solution is to be
    /// found. A minimization terminates when
    ///
    /// ||g|| < epsilon * max(1, ||x||),
    ///
    /// where ||.|| denotes the Euclidean (L2) norm. The default value is 1e-5.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(
            epsilon.is_sign_positive(),
            "Invalid parameter epsilon specified."
        );

        self.param.epsilon = epsilon;

        self
    }

    /// Set orthantwise parameters
    pub fn with_orthantwise(mut self, c: f64, start: usize, end: usize) -> Self {
        assert!(
            c.is_sign_positive(),
            "Invalid parameter orthantwise c parameter specified."
        );
        warn!("Only the backtracking line search is available for OWL-QN algorithm.");

        self.param.orthantwise = true;
        self.param.owlqn.c = c;
        self.param.owlqn.start = start as i32;
        self.param.owlqn.end = end as i32;

        self
    }

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 1e-4. This parameter should be greater
    /// than zero and smaller than 0.5.
    pub fn with_linesearch_ftol(mut self, ftol: f64) -> Self {
        assert!(ftol >= 0.0, "Invalid parameter ftol specified.");
        self.param.linesearch.ftol = ftol;

        self
    }

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 0.9. If the function and gradient evaluations are
    /// inexpensive with respect to the cost of the iteration (which is
    /// sometimes the case when solving very large problems) it may be
    /// advantageous to set this parameter to a small value. A typical small
    /// value is 0.1. This parameter shuold be greater than the ftol parameter
    /// (1e-4) and smaller than 1.0.
    pub fn with_linesearch_gtol(mut self, gtol: f64) -> Self {
        assert!(
            gtol >= 0.0 && gtol < 1.0 && gtol > self.param.linesearch.ftol,
            "Invalid parameter gtol specified."
        );

        self.param.linesearch.gtol = gtol;

        self
    }

    /// xtol is a nonnegative input variable. termination occurs when the
    /// relative width of the interval of uncertainty is at most xtol.
    ///
    /// The machine precision for floating-point values.
    ///
    ///  This parameter must be a positive value set by a client program to
    ///  estimate the machine precision. The line search routine will terminate
    ///  with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
    ///  of the interval of uncertainty is less than this parameter.
    pub fn with_linesearch_xtol(mut self, xtol: f64) -> Self {
        assert!(xtol >= 0.0, "Invalid parameter xtol specified.");

        self.param.linesearch.xtol = xtol;
        self
    }

    /// The minimum step of the line search routine.
    ///
    /// The default value is 1e-20. This value need not be modified unless the
    /// exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    pub fn with_linesearch_min_step(mut self, min_step: f64) -> Self {
        assert!(min_step >= 0.0, "Invalid parameter min_step specified.");

        self.param.linesearch.min_step = min_step;
        self
    }

    /// Set the maximum number of iterations.
    ///
    /// The lbfgs optimization terminates when the iteration count exceedes this
    /// parameter. Setting this parameter to zero continues an optimization
    /// process until a convergence or error.
    ///
    /// The default value is 0.
    pub fn with_max_iterations(mut self, niter: usize) -> Self {
        self.param.max_iterations = niter;
        self
    }

    /// The maximum allowed number of evaluations of function value and
    /// gradients. This number could be larger than max_iterations since line
    /// search procedure may involve one or more evaluations.
    ///
    /// Setting this parameter to zero continues an optimization process until a
    /// convergence or error. The default value is 0.
    pub fn with_max_evaluations(mut self, neval: usize) -> Self {
        self.param.max_evaluations = neval;
        self
    }

    /// This parameter determines the minimum rate of decrease of the objective
    /// function. The library stops iterations when the following condition is
    /// met: |f' - f| / f < delta, where f' is the objective value of past
    /// iterations ago, and f is the objective value of the current iteration.
    ///
    /// If `past` is zero, the library does not perform the delta-based
    /// convergence test.
    ///
    /// The default value of delta is 1e-5.
    ///
    pub fn with_fx_delta(mut self, delta: f64, past: usize) -> Self {
        assert!(delta >= 0.0, "Invalid parameter delta specified.");

        self.param.past = past;
        self.param.delta = delta;
        self
    }

    /// Select line search algorithm
    ///
    /// The default is "MoreThuente" line search algorithm.
    pub fn with_linesearch_algorithm(mut self, algo: &str) -> Self {
        match algo {
            "MoreThuente" => self.param.linesearch.algorithm = LineSearchAlgorithm::MoreThuente,
            "BacktrackingArmijo" => {
                self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingArmijo
            }
            "BacktrackingStrongWolfe" => {
                self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingStrongWolfe
            }
            "BacktrackingWolfe" | "Backtracking" => {
                self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingWolfe
            }
            _ => unimplemented!(),
        }

        self
    }
}
// builder:1 ends here

// src

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*src][src:1]]
impl<F> LBFGS<F>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// Start the L-BFGS optimization; this will invoke the callback functions evaluate
    /// and progress.
    /// 
    /// # Parameters
    /// 
    /// * x       : The array of input variables.
    /// * evaluate: A closure for evaluating function value and gradient
    /// * progress: A closure for monitor progress or defining stopping condition
    /// 
    /// # Return
    /// 
    /// * on success, return final evaluated `Problem`.
    pub fn minimize<'a, G>(
        self,
        x: &'a mut [f64],
        eval_fn: F,
        mut prgr_fn: G,
    ) -> Result<Problem<'a, F>>
    where
        G: FnMut(&Progress) -> bool,
    {
        // Initialize the limited memory.
        let param = &self.param;
        let m = param.m;
        let mut lm_arr: Vec<_> = (0..m).map(|_| IterationData::new(x.len())).collect();

        // Allocate working space for LBFGS optimization
        let owlqn = if param.orthantwise {
            Some(param.owlqn.clone())
        } else {
            None
        };
        let mut problem = Problem::new(x, eval_fn, owlqn);

        // Evaluate the function value and its gradient.
        problem.evaluate()?;

        // Compute the L1 norm of the variable and add it to the object value.
        problem.update_owlqn_gradient();

        // Compute the search direction with current gradient.
        problem.update_search_direction();

        // Compute the initial step:
        let mut step = problem.initial_step();

        let mut end = 0;
        let linesearch = param.linesearch;

        // Allocate an array for storing previous values of the objective function.
        let mut pf = vec![0.0; param.past as usize];

        info!("start lbfgs loop...");
        for k in 1.. {
            // Store the current position and gradient vectors.
            problem.save_state();

            // Search for an optimal step.
            let ls = linesearch.find(&mut problem, &mut step)?;
            problem.update_owlqn_gradient();

            // Monitor the progress.
            let prgr = Progress::new(&problem, k, ls, step);

            // User defined callback function
            let cancel = prgr_fn(&prgr);
            if cancel {
                info!("The minimization process has been canceled.");
                break;
            }
            if satisfying_stop_conditions(param, prgr, &mut pf) {
                break;
            }

            // Update LBFGS iteration data.
            let it = &mut lm_arr[end];
            let gamma = it.update(&problem.x, &problem.xp, &problem.gx, &problem.gp);

            // Compute the steepest direction
            problem.update_search_direction();
            let d = problem.search_direction_mut();

            // Apply LBFGS recursion procedure.
            end = lbfgs_two_loop_recursion(&mut lm_arr, d, gamma, m, k, end);

            // Constrain the search direction for orthant-wise updates.
            problem.constrain_search_direction();

            // Now the search direction d is ready. We try step = 1 first.
            step = 1.0
        }

        // Return the final value of the objective function.
        Ok(problem)
    }
}
// src:1 ends here

// recursion

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*recursion][recursion:1]]
/// Algorithm 7.4, in Nocedal, J.; Wright, S. Numerical Optimization; Springer Science & Business Media, 2006.
fn lbfgs_two_loop_recursion(
    lm_arr: &mut [IterationData],
    d: &mut [f64], // search direction
    gamma: f64,    // H_k^{0} = \gamma I
    m: usize,
    k: usize,
    end: usize,
) -> usize {
    let end = (end + 1) % m;
    let mut j = end;
    let bound = m.min(k);

    // L-BFGS two-loop recursion, part1
    for _ in 0..bound {
        j = (j + m - 1) % m;
        let it = &mut lm_arr[j as usize];

        // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}.
        it.alpha = it.s.vecdot(&d) / it.ys;
        // q_{i} = q_{i+1} - \alpha_{i} y_{i}.
        d.vecadd(&it.y, -it.alpha);
    }
    d.vecscale(gamma);

    // L-BFGS two-loop recursion, part2
    for _ in 0..bound {
        let it = &mut lm_arr[j as usize];
        // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}.
        let beta = it.y.vecdot(d) / it.ys;
        // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}.
        d.vecadd(&it.s, it.alpha - beta);
        j = (j + 1) % m;
    }

    end
}
// recursion:1 ends here

// iteration data

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*iteration%20data][iteration data:1]]
/// Internal iternation data for L-BFGS
#[derive(Clone)]
struct IterationData {
    pub alpha: f64,

    pub s: Vec<f64>,

    pub y: Vec<f64>,

    /// vecdot(y, s)
    pub ys: f64,
}

impl IterationData {
    fn new(n: usize) -> Self {
        IterationData {
            alpha: 0.0,
            ys: 0.0,
            s: vec![0.0; n],
            y: vec![0.0; n],
        }
    }

    /// Update LBFGS iteration state, returns Cholesky factor \gamma for scaling
    /// the initial inverse Hessian matrix $H_k^0$
    fn update(&mut self, x: &[f64], xp: &[f64], gx: &[f64], gp: &[f64]) -> f64 {
        // Update vectors s and y:
        // s_{k} = x_{k+1} - x_{k} = \alpha * d_{k}.
        // y_{k} = g_{k+1} - g_{k}.
        self.s.vecdiff(x, xp);
        self.y.vecdiff(gx, gp);

        // Compute scalars ys and yy:
        // ys = y^t \cdot s = 1 / \rho.
        // yy = y^t \cdot y.
        // Notice that yy is used for scaling the intial inverse hessian matrix H_0 (Cholesky factor).
        let ys = self.y.vecdot(&self.s);
        let yy = self.y.vecdot(&self.y);
        self.ys = ys;

        ys / yy
    }
}
// iteration data:1 ends here

// stopping conditions

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*stopping%20conditions][stopping conditions:1]]
/// test if progress satisfying stop condition
#[inline]
fn satisfying_stop_conditions(param: &LbfgsParam, prgr: Progress, pf: &mut [f64]) -> bool {
    // Buildin tests for stopping conditions
    if satisfying_max_iterations(&prgr, param.max_iterations)
        || satisfying_max_evaluations(&prgr, param.max_evaluations)
        || satisfying_scaled_gnorm(&prgr, param.epsilon)
        || satisfying_delta(&prgr, pf, param.delta)
    // || satisfying_max_gnorm(&prgr, self.param.max_gnorm)
    {
        return true;
    }

    false
}

/// The criterion is given by the following formula:
///     |g(x)| / \max(1, |x|) < \epsilon
#[inline]
fn satisfying_scaled_gnorm(prgr: &Progress, epsilon: f64) -> bool {
    if prgr.gnorm / prgr.xnorm.max(1.0) <= epsilon {
        // Convergence.
        info!("L-BFGS reaches convergence.");
        true
    } else {
        false
    }
}

/// Maximum number of lbfgs iterations.
#[inline]
fn satisfying_max_iterations(prgr: &Progress, max_iterations: usize) -> bool {
    if max_iterations == 0 {
        false
    } else if prgr.niter >= max_iterations {
        warn!("max_iterations reached!");
        true
    } else {
        false
    }
}

/// Maximum number of function evaluations
#[inline]
fn satisfying_max_evaluations(prgr: &Progress, max_evaluations: usize) -> bool {
    if max_evaluations == 0 {
        false
    } else if prgr.neval >= max_evaluations {
        warn!("Max allowed evaluations reached!");
        true
    } else {
        false
    }
}

#[inline]
fn satisfying_max_gnorm(prgr: &Progress, max_gnorm: f64) -> bool {
    prgr.gx.vec2norm() <= max_gnorm
}

/// Functiona value (fx) delta based stopping criterion
///
/// Test for stopping criterion.
/// The criterion is given by the following formula:
///    |f(past_x) - f(x)| / f(x) < delta
///
/// # Parameters
///
/// * pf: an array for storing previous values of the objective function.
/// * delta: max fx delta allowed
///
#[inline]
fn satisfying_delta<'a>(prgr: &Progress, pf: &'a mut [f64], delta: f64) -> bool {
    let k = prgr.niter;
    let fx = prgr.fx;
    let past = pf.len();
    if past < 1 {
        return false;
    }

    // We don't test the stopping criterion while k < past.
    if past <= k {
        // Compute the relative improvement from the past.
        let rate = (pf[(k % past) as usize] - fx).abs() / fx;
        // The stopping criterion.
        if rate < delta {
            info!("The stopping criterion.");
            return true;
        }
    }
    // Store the current value of the objective function.
    pf[(k % past) as usize] = fx;

    false
}
// stopping conditions:1 ends here
