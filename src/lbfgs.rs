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

// return value

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*return%20value][return value:1]]
//
//  Return values of lbfgs().
//
//   Roughly speaking, a negative value indicates an error.

// The algorithm routine reaches the maximum number of iterations.
pub const LBFGSERR_MAXIMUMITERATION: i32 = -997;
// The line-search routine reaches the maximum number of evaluations.
pub const LBFGSERR_MAXIMUMLINESEARCH: i32 = -998;
// The line-search step became larger than lbfgs_parameter_t::max_step.
pub const LBFGSERR_MAXIMUMSTEP: i32 = -999;
// The line-search step became smaller than lbfgs_parameter_t::min_step.
pub const LBFGSERR_MINIMUMSTEP: i32 = -1000;
// return value:1 ends here

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
    ///     ||g|| < epsilon * max(1, ||x||),
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
    /// met: (f' - f) / f < delta, where f' is the objective value of \ref past
    /// iterations ago, and f is the objective value of the current iteration.
    /// The default value is 1e-5.
    ///
    pub delta: f64,

    /// The maximum number of iterations.
    ///
    ///  The lbfgs() function terminates an optimization process with
    ///  ::LBFGSERR_MAXIMUMITERATION status code when the iteration count
    ///  exceedes this parameter. Setting this parameter to zero continues an
    ///  optimization process until a convergence or error.
    ///
    /// The default value is 0.
    pub max_iterations: usize,

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
            orthantwise: false,
            owlqn: Orthantwise::default(),
            linesearch: LineSearch::default(),
        }
    }
}

impl LbfgsParam {
    // Check the input parameters for errors.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.epsilon >= 0.0, "Invalid parameter epsilon specified.");

        ensure!(self.delta >= 0.0, "Invalid parameter delta specified.");

        // check line search parameters
        let ls = self.linesearch;

        ensure!(ls.min_step >= 0.0, "Invalid parameter min_step specified.");
        ensure!(
            ls.max_step >= ls.min_step,
            "Invalid parameter max_step specified."
        );

        ensure!(ls.ftol >= 0.0, "Invalid parameter ftol specified.");
        ensure!(ls.gtol >= 0.0, "Invalid parameter gtol specified.");
        ensure!(ls.xtol >= 0.0, "Invalid parameter xtol specified.");

        use self::LineSearchAlgorithm::*;
        match ls.algorithm {
            BacktrackingWolfe | BacktrackingStrongWolfe => ensure!(
                ls.wolfe > ls.ftol && ls.wolfe < 1.0,
                "Invalid parameter lbfgs_parameter_t::wolfe specified."
            ),
            _ => {
                if self.orthantwise {
                    warn!("Only the backtracking line search is available for OWL-QN algorithm.");
                }
            }
        }

        // FIXME: take care below
        ensure!(
            self.owlqn.c >= 0.0,
            "Invalid parameter lbfgs_parameter_t::orthantwise_c specified."
        );
        // ensure!(
        //     self.owlqn.start >= 0 && self.owlqn.start < n,
        //     "Invalid parameter orthantwise_start specified."
        // );

        // ensure!(
        //     self.owlqn.end > 0 && self.owlqn.end <= n,
        //     "Invalid parameter orthantwise_end specified."
        // );

        Ok(())
    }
}
// parameters:1 ends here

// problem

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*problem][problem:1]]
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
    pub xp: Vec<f64>,

    /// Cached gradient vector of previous step.
    pub gp: Vec<f64>,

    /// Pseudo gradient for OrthantWise Limited-memory Quasi-Newton (owlqn) algorithm.
    pub pg: Vec<f64>,

    /// Store callback function for evaluating objective function.
    eval_fn: E,

    /// Orthantwise operations
    owlqn: Option<Orthantwise>,
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
            x,
            eval_fn,
            owlqn,
        }
    }

    /// Compute the initial gradient in the search direction.
    pub fn dginit(&self, d: &[f64]) -> Result<f64> {
        if self.owlqn.is_none() {
            let dginit = self.gx.vecdot(d);
            ensure!(
                dginit <= 0.0,
                "The current search direction increases the objective function value."
            );

            Ok(dginit)
        } else {
            Ok(self.pg.vecdot(d))
        }
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

        Ok(())
    }

    /// Copies all elements from src into self.
    pub fn clone_from(&mut self, src: &Problem<E>) {
        self.x.clone_from_slice(&src.x);
        self.gx.clone_from_slice(&src.gx);
        self.fx = src.fx;
    }

    /// Store the current position and gradient vectors.
    pub fn update_state(&mut self) {
        self.xp.veccpy(&self.x);
        self.gp.veccpy(&self.gx);
    }

    /// Compute the direction;
    /// we assume the initial hessian matrix H_0 as the identity matrix.
    pub fn update_search_direction(&self, d: &mut [f64]) {
        if self.owlqn.is_some() {
            d.vecncpy(&self.pg);
        } else {
            d.vecncpy(&self.gx);
        }
    }

    /// For line search
    ///
    /// Compute the current value of x: x <- x + (*stp) * s.
    pub fn take_line_step(&mut self, s: &[f64], stp: f64) {
        self.x.veccpy(&self.xp);
        self.x.vecadd(s, stp);

        // Choose the orthant for the new point.
        // The current point is projected onto the orthant.
        if let Some(owlqn) = self.owlqn {
            owlqn.project(&mut self.x, &self.xp, &self.gp);
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

    /// Constrain the search direction for orthant-wise updates.
    pub fn constrain_search_direction(&self, d: &mut [f64]) {
        if let Some(owlqn) = self.owlqn {
            owlqn.constrain(d, &self.pg);
        }
    }

    // FIXME
    pub fn update_owlqn_gradient(&mut self) {
        if let Some(owlqn) = self.owlqn {
            owlqn.pseudo_gradient(&mut self.pg, &self.x, &self.gx);
        }
    }

    pub fn orthantwise(&self) -> bool {
        self.owlqn.is_some()
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
    /// The number of evaluations called for this iteration.
    pub ncall: usize,
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
    ///     |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
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
    pub fn x1norm(&self, x: &[f64]) -> f64 {
        let (start, end) = self.start_end(x);

        let mut s = 0.0;
        for i in start..end {
            s += self.c * x[i].abs();
        }

        s
    }

    /// Compute the psuedo-gradients.
    pub fn pseudo_gradient(&self, pg: &mut [f64], x: &[f64], g: &[f64]) {
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
    pub fn project(&self, x: &mut [f64], xp: &[f64], gp: &[f64]) {
        let (start, end) = self.start_end(xp);

        for i in start..end {
            let sign = if xp[i] == 0.0 { -gp[i] } else { xp[i] };
            if x[i] * sign <= 0.0 {
                x[i] = 0.0
            }
        }
    }

    pub fn constrain(&self, d: &mut [f64], pg: &[f64]) {
        let (start, end) = self.start_end(pg);

        for i in start..end {
            if d[i] * pg[i] >= 0.0 {
                d[i] = 0.0;
            }
        }
    }
}
// orthantwise:1 ends here

// common

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*common][common:1]]
/// Internal iternation data for L-BFGS
#[derive(Clone)]
struct IterationData {
    pub alpha: f64,

    /// [n]
    pub s: Vec<f64>,

    /// [n]
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
}
// common:1 ends here

// lbfgs

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*lbfgs][lbfgs:1]]
pub fn lbfgs<F, G>(
    x: &mut [f64],
    ptr_fx: &mut f64,
    mut proc_evaluate: F,
    mut proc_progress: Option<G>,
    param: &LbfgsParam,
) -> Result<i32>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    G: FnMut(&Progress) -> bool,
{
    param.validate()?;

    // Initialize the limited memory.
    let n = x.len();
    let m = param.m;
    let mut lm_arr: Vec<_> = (0..m).map(|_| IterationData::new(n)).collect();

    // Allocate an array for storing previous values of the objective function.
    let mut pf = vec![0.0; param.past as usize];

    // Allocate working space for OWL-QN algorithm.
    let owlqn = if param.orthantwise {
        Some(param.owlqn.clone())
    } else {
        None
    };

    // Evaluate the function value and its gradient.
    // Compute the L1 norm of the variable and add it to the object value.
    let mut problem = Problem::new(x, &mut proc_evaluate, owlqn);
    problem.evaluate()?;
    problem.update_owlqn_gradient();

    // Compute the direction;
    // we assume the initial hessian matrix H_0 as the identity matrix.
    let mut d = vec![0.0; n];
    problem.update_search_direction(&mut d);

    // Make sure that the initial variables are not a minimizer.
    let xnorm = problem.xnorm();
    let gnorm = problem.gnorm();
    if gnorm / xnorm.max(1.0) <= param.epsilon {
        info!("The initial variables already minimize the objective function.");
        return Ok(0);
    }

    // Compute the initial step:
    let mut step = d.vec2norminv();
    let mut end = 0;

    // FIXME: return code
    let mut ret = 0;
    let linesearch = &param.linesearch;
    info!("start lbfgs loop...");
    for k in 1.. {
        // Store the current position and gradient vectors.
        problem.update_state();

        // Search for an optimal step.
        let ls = linesearch.find(&mut problem, &d, &mut step)?;
        problem.update_owlqn_gradient();

        // Compute x and g norms.
        let xnorm = problem.xnorm();
        let gnorm = problem.gnorm();

        // Report the progress.
        if let Some(ref mut prgr_fn) = proc_progress {
            let prgr = Progress {
                x: &problem.x,
                gx: &problem.gx,
                fx: problem.fx,
                niter: k,
                ncall: ls as usize,
                xnorm,
                gnorm,
                step,
            };

            let cancel = prgr_fn(&prgr);
            if cancel {
                info!("The minimization process has been canceled.");
                break;
            }
        }

        // Convergence test.
        // The criterion is given by the following formula:
        //     |g(x)| / \max(1, |x|) < \epsilon
        if gnorm / xnorm.max(1.0) <= param.epsilon {
            // Convergence.
            info!("L-BFGS reaches convergence.");
            ret = 0;
            break;
        }

        // Test for stopping criterion.
        // The criterion is given by the following formula:
        //    (f(past_x) - f(x)) / f(x) < \delta
        let fx = problem.fx;
        if pf.len() > 0 {
            // We don't test the stopping criterion while k < past.
            if param.past <= k {
                // Compute the relative improvement from the past.
                let rate = (pf[(k % param.past) as usize] - fx) / fx;
                // The stopping criterion.
                if rate < param.delta {
                    info!("The stopping criterion.");
                    ret = 1i32;
                    break;
                }
            }
            // Store the current value of the objective function.
            pf[(k % param.past) as usize] = fx;
        }

        if param.max_iterations != 0 && param.max_iterations < k + 1 {
            // Maximum number of iterations.
            warn!("max_iterations reached!");
            ret = LBFGSERR_MAXIMUMITERATION as i32;
            break;
        }

        // Update vectors s and y:
        // s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        // y_{k+1} = g_{k+1} - g_{k}.
        let it = &mut lm_arr[end];
        it.s.vecdiff(&problem.x, &problem.xp);
        it.y.vecdiff(&problem.gx, &problem.gp);

        // Compute scalars ys and yy:
        // ys = y^t \cdot s = 1 / \rho.
        // yy = y^t \cdot y.
        // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
        let ys = it.y.vecdot(&it.s);
        let yy = it.y.vecdot(&it.y);

        it.ys = ys;

        // Recursive formula to compute dir = -(H \cdot g).
        // This is described in page 779 of:
        // Jorge Nocedal.
        // Updating Quasi-Newton Matrices with Limited Storage.
        // Mathematics of Computation, Vol. 35, No. 151,
        // pp. 773--782, 1980.
        let bound = if m <= k { m } else { k };

        end = (end + 1) % m;
        // Compute the steepest direction.
        problem.update_search_direction(&mut d);

        let mut j = end;
        for _ in 0..bound {
            j = (j + m - 1) % m;
            let it = &mut lm_arr[j as usize];

            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}.
            it.alpha = it.s.vecdot(&d) / it.ys;
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}.
            d.vecadd(&it.y, -it.alpha);
        }
        d.vecscale(ys / yy);

        for _ in 0..bound {
            let it = &mut lm_arr[j as usize];
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}.
            let beta = it.y.vecdot(&d) / it.ys;
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}.
            d.vecadd(&it.s, it.alpha - beta);
            j = (j + 1) % m;
        }

        // Constrain the search direction for orthant-wise updates.
        problem.constrain_search_direction(&mut d);

        // Now the search direction d is ready. We try step = 1 first.
        step = 1.0
    }

    // Return the final value of the objective function.
    *ptr_fx = problem.fx;

    Ok(ret)
}
// lbfgs:1 ends here
