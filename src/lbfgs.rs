// header

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*header][header:1]]
//       Limited memory BFGS (L-BFGS).
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

// TODO return value

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*return%20value][return value:1]]
//
//  Return values of lbfgs().
//
//   Roughly speaking, a negative value indicates an error.

pub type unnamed = i32;
/* * The current search direction increases the objective function value. */
pub const LBFGSERR_INCREASEGRADIENT: unnamed = -994;
/* * A logic error (negative line-search step) occurred. */
pub const LBFGSERR_INVALIDPARAMETERS: unnamed = -995;
/* * Relative width of the interval of uncertainty is at most
lbfgs_parameter_t::xtol. */
pub const LBFGSERR_WIDTHTOOSMALL: unnamed = -996;
/* * The algorithm routine reaches the maximum number of iterations. */
pub const LBFGSERR_MAXIMUMITERATION: unnamed = -997;
/* * The line-search routine reaches the maximum number of evaluations. */
pub const LBFGSERR_MAXIMUMLINESEARCH: unnamed = -998;
/* * The line-search step became larger than lbfgs_parameter_t::max_step. */
pub const LBFGSERR_MAXIMUMSTEP: unnamed = -999;
/* * The line-search step became smaller than lbfgs_parameter_t::min_step. */
pub const LBFGSERR_MINIMUMSTEP: unnamed = -1000;
/* * A rounding error occurred; alternatively, no line-search step
satisfies the sufficient decrease and curvature conditions. */
pub const LBFGSERR_ROUNDING_ERROR: unnamed = -1001;
/* * A logic error occurred; alternatively, the interval of uncertainty
became too small. */
pub const LBFGSERR_INCORRECT_TMINMAX: unnamed = -1002;
/* * The line-search step went out of the interval of uncertainty. */
pub const LBFGSERR_OUTOFINTERVAL: unnamed = -1003;
/* * Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
pub const LBFGSERR_INVALID_ORTHANTWISE_END: unnamed = -1004;
/* * Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
pub const LBFGSERR_INVALID_ORTHANTWISE_START: unnamed = -1005;
/* * Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
pub const LBFGSERR_INVALID_ORTHANTWISE: unnamed = -1006;
/* * Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
pub const LBFGSERR_INVALID_MAXLINESEARCH: unnamed = -1007;
/* * Invalid parameter lbfgs_parameter_t::xtol specified. */
pub const LBFGSERR_INVALID_XTOL: unnamed = -1008;
/* * Invalid parameter lbfgs_parameter_t::gtol specified. */
pub const LBFGSERR_INVALID_GTOL: unnamed = -1009;
/* * Invalid parameter lbfgs_parameter_t::wolfe specified. */
pub const LBFGSERR_INVALID_WOLFE: unnamed = -1010;
/* * Invalid parameter lbfgs_parameter_t::ftol specified. */
pub const LBFGSERR_INVALID_FTOL: unnamed = -1011;
/* * Invalid parameter lbfgs_parameter_t::max_step specified. */
pub const LBFGSERR_INVALID_MAXSTEP: unnamed = -1012;
/* * Invalid parameter lbfgs_parameter_t::max_step specified. */
pub const LBFGSERR_INVALID_MINSTEP: unnamed = -1013;
/* * Invalid parameter lbfgs_parameter_t::linesearch specified. */
pub const LBFGSERR_INVALID_LINESEARCH: unnamed = -1014;
/* * Invalid parameter lbfgs_parameter_t::delta specified. */
pub const LBFGSERR_INVALID_DELTA: unnamed = -1015;
/* * Invalid parameter lbfgs_parameter_t::past specified. */
pub const LBFGSERR_INVALID_TESTPERIOD: unnamed = -1016;
/* * Invalid parameter lbfgs_parameter_t::epsilon specified. */
pub const LBFGSERR_INVALID_EPSILON: unnamed = -1017;
/* * The array x must be aligned to 16 (for SSE). */
pub const LBFGSERR_INVALID_X_SSE: unnamed = -1018;
/* * Invalid number of variables (for SSE) specified. */
pub const LBFGSERR_INVALID_N_SSE: unnamed = -1019;
/* * Invalid number of variables specified. */
pub const LBFGSERR_INVALID_N: unnamed = -1020;
/* * The minimization process has been canceled. */
pub const LBFGSERR_CANCELED: unnamed = -1021;
/* * Insufficient memory. */
pub const LBFGSERR_OUTOFMEMORY: unnamed = -1022;
/* * Logic error. */
pub const LBFGSERR_LOGICERROR: unnamed = -1023;
/* * Unknown error. */
pub const LBFGSERR_UNKNOWNERROR: unnamed = -1024;
/* * The initial variables already minimize the objective function. */
pub const LBFGS_ALREADY_MINIMIZED: unnamed = 2;
pub const LBFGS_STOP: unnamed = 1;
pub const LBFGS_CONVERGENCE: unnamed = 0;
/* * L-BFGS reaches convergence. */
pub const LBFGS_SUCCESS: unnamed = 0;
// return value:1 ends here

// lbfgs parameters

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*lbfgs%20parameters][lbfgs parameters:1]]
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
    pub linesearch: LineSearchParam,

    /// Coeefficient for the L1 norm of variables.
    ///
    ///  This parameter should be set to zero for standard minimization
    ///  problems. Setting this parameter to a positive value activates
    ///  Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
    ///  minimizes the objective function F(x) combined with the L1 norm |x| of
    ///  the variables, {F(x) + C |x|}. This parameter is the coeefficient for
    ///  the |x|, i.e., C. As the L1 norm |x| is not differentiable at zero, the
    ///  library modifies function and gradient evaluations from a client
    ///  program suitably; a client program thus have only to return the
    ///  function value F(x) and gradients G(x) as usual. The default value is
    ///  zero.
    pub orthantwise_c: f64,

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
    pub orthantwise_start: i32,

    /// End index for computing L1 norm of the variables.
    ///
    /// This parameter is valid only for OWL-QN method (i.e., \ref orthantwise_c
    /// != 0). This parameter e (0 < e <= N) specifies the index number at which
    /// the library stops computing the L1 norm of the variables x,
    pub orthantwise_end: i32,
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
            orthantwise_c: 0.0,
            orthantwise_start: 0,
            orthantwise_end: -1,
            linesearch: LineSearchParam::default(),
        }
    }
}

impl LbfgsParam {
    // Check the input parameters for errors.
    pub fn validate(&self) -> Result<()> {
        if self.epsilon < 0.0 {
            bail!("LBFGSERR_INVALID_EPSILON");
        }
        if self.past < 0 {
            bail!("LBFGSERR_INVALID_TESTPERIOD");
        }
        if self.delta < 0.0 {
            bail!("LBFGSERR_INVALID_DELTA");
        }
        if self.linesearch.min_step < 0.0f64 {
            bail!("LBFGSERR_INVALID_MINSTEP");
        }
        if self.linesearch.max_step < self.linesearch.min_step {
            bail!("LBFGSERR_INVALID_MINSTEP");
        }
        if self.linesearch.ftol < 0.0 {
            bail!("LBFGSERR_INVALID_FTOL");
        }

        // FIXME: review needed
        use self::LineSearchAlgorithm::*;
        if self.linesearch.algorithm == BacktrackingWolfe
            || self.linesearch.algorithm == BacktrackingStrongWolfe
        {
            if self.linesearch.wolfe <= self.linesearch.ftol || 1.0 <= self.linesearch.wolfe {
                bail!("LBFGSERR_INVALID_WOLFE");
            }
        }

        if self.linesearch.gtol < 0.0 {
            bail!("LBFGSERR_INVALID_GTOL");
        }
        if self.linesearch.xtol < 0.0 {
            bail!("LBFGSERR_INVALID_XTOL");
        }
        if self.linesearch.max_linesearch <= 0 {
            bail!("LBFGSERR_INVALID_MAXLINESEARCH");
        }

        // FIXME: take care below
        if self.orthantwise_c < 0.0 {
            bail!("LBFGSERR_INVALID_ORTHANTWISE");
        }
        // if self.orthantwise_start < 0 || n < self.orthantwise_start {
        //     bail!("LBFGSERR_INVALID_ORTHANTWISE_START");
        // }

        // if self.orthantwise_end < 0 {
        //     self.orthantwise_end = n;
        // }

        // if n < self.orthantwise_end {
        //     bail!("LBFGSERR_INVALID_ORTHANTWISE_END");
        // }

        Ok(())
    }
}
// lbfgs parameters:1 ends here

// problem

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*problem][problem:1]]
#[derive(Debug, Clone)]
pub struct Problem {
    /// x is an array of length n. on input it must contain the base point for
    /// the line search.
    pub x: Vec<f64>,

    /// `fx` is a variable. It must contain the value of problem `f` at
    /// x.
    pub fx: f64,

    /// `gx` is an array of length n. It must contain the gradient of `f` at
    /// x.
    pub gx: Vec<f64>,
}

impl Problem {
    /// Initialize problem with array length n
    pub fn new(x: &[f64]) -> Self {
        let n = x.len();
        Problem {
            x: x.into(),
            fx: 0.0,
            gx: vec![0.0; n],
        }
    }

    /// Copies all elements from src into self.
    pub fn clone_from(&mut self, src: &Problem) {
        self.x.clone_from_slice(&src.x);
        self.gx.clone_from_slice(&src.gx);
        self.fx = src.fx;
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

// owlqn pseduo gradient

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*owlqn%20pseduo%20gradient][owlqn pseduo gradient:1]]
/// Compute the psuedo-gradients.
fn owlqn_pseudo_gradient(
    pg: &mut [f64],
    x: &[f64],
    g: &[f64],
    c: f64,
    start: usize,
    end: usize,
) {
    // Compute the negative of gradients.
    for i in 0..start {
        pg[i] = g[i];
    }

    // Compute the psuedo-gradients.
    for i in start..end {
        if x[i] < 0.0 {
            // Differentiable.
            pg[i] = g[i] - c;
        } else if (0.0 < x[i]) {
            pg[i] = g[i] + c;
        } else {
            if (g[i] < -c) {
                // Take the right partial derivative.
                pg[i] = g[i] + c;
            } else if (c < g[i]) {
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
// owlqn pseduo gradient:1 ends here

// common

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*common][common:1]]
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
// common:1 ends here

// old lbfgs

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old%20lbfgs][old lbfgs:1]]
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
    // FIXME: make param immutable
    let mut param = param.clone();
    let m = param.m;
    // FIXME: remove n
    let n = x.len();
    param.validate()?;

    // FIXME: make param immutable
    if param.orthantwise_start < 0 || (n as i32) < param.orthantwise_start {
        bail!("LBFGSERR_INVALID_ORTHANTWISE_START");
    }
    if param.orthantwise_end < 0 {
        //bail!("LBFGSERR_INVALID_ORTHANTWISE_END");
        param.orthantwise_end = n as i32
    }
    if (n as i32) < param.orthantwise_end {
        bail!("LBFGSERR_INVALID_ORTHANTWISE_END");
    }

    // FIXME: review below
    if param.orthantwise_c != 0.0 {
        warn!("Only the backtracking method is available.");
    }

    // Allocate working space.
    let mut xp = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut gp = g.clone();
    let mut d = vec![0.0; n];

    // FIXME: check param.orthantwise_c or not?
    // Allocate working space for OW-LQN.
    let mut pg = vec![0.0; n];

    // Allocate limited memory storage.
    let mut lm_arr: Vec<IterationData> = Vec::with_capacity(m);

    // Initialize the limited memory.
    for i in 0..m {
        lm_arr.push(IterationData {
            alpha: 0.0,
            ys: 0.0,
            s: vec![0.0; n],
            y: vec![0.0; n],
        });
    }

    // Allocate an array for storing previous values of the objective function.
    let mut pf = vec![0.0; param.past as usize];

    // Store the initial value of the objective function.
    let mut fx = 0.0;
    if pf.len() > 0 {
        pf[0] = fx;
    }

    // Evaluate the function value and its gradient.

    fx = proc_evaluate(&x, &mut g)?;

    if 0.0 != param.orthantwise_c {
        // Compute the L1 norm of the variable and add it to the object value.
        // xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
        let xnorm = x.owlqn_x1norm(
            param.orthantwise_start as usize,
            param.orthantwise_end as usize,
        );

        fx += xnorm * param.orthantwise_c;
        owlqn_pseudo_gradient(
            &mut pg,
            &x,
            &g,
            param.orthantwise_c,
            param.orthantwise_start as usize,
            param.orthantwise_end as usize,
        );
    }

    // Compute the direction;
    // we assume the initial hessian matrix H_0 as the identity matrix.
    if param.orthantwise_c == 0.0 {
        d.vecncpy(&g);
    } else {
        d.vecncpy(&pg);
    }

    // Make sure that the initial variables are not a minimizer.
    let xnorm = x.vec2norm().max(1.0);
    let mut gnorm = if param.orthantwise_c == 0.0 {
        g.vec2norm()
    } else {
        pg.vec2norm()
    };

    if gnorm / xnorm <= param.epsilon {
        bail!("LBFGS_ALREADY_MINIMIZED");
    }

    // Compute the initial step:
    // step = 1.0 / sqrt(vecdot(d, d, n))
    let mut step = d.vec2norminv();
    let mut k: usize = 1;
    let mut end = 0;

    // FIXME: return code
    let mut ret = 0;
    let mut ls = 0i32;

    use crate::line::LineSearchAlgorithm::*;
    info!("start lbfgs loop...");
    loop {
        // Store the current position and gradient vectors.
        xp.veccpy(&x);
        gp.veccpy(&g);

        // Search for an optimal step.
        if param.orthantwise_c == 0.0 {
            if param.linesearch.algorithm == MoreThuente {
                ls = line_search_morethuente(
                    x,
                    &mut fx,
                    &mut g,
                    &d,
                    &mut step,
                    &xp,
                    &gp,
                    &mut proc_evaluate,
                    &param,
                )?;
            } else {
                ls = line_search_backtracking(
                    x,
                    &mut fx,
                    &mut g,
                    &d,
                    &mut step,
                    &xp,
                    &gp,
                    &mut proc_evaluate,
                    &param,
                )?;
            }
        } else {
            ls = line_search_backtracking(
                x,
                &mut fx,
                &mut g,
                &d,
                &mut step,
                &xp,
                &pg,
                &mut proc_evaluate,
                &param,
            )?;
            owlqn_pseudo_gradient(
                &mut pg,
                &x,
                &g,
                param.orthantwise_c,
                param.orthantwise_start as usize,
                param.orthantwise_end as usize,
            );
        }

        // FIXME: to be better
        // Recover from failed line search?
        if ls < 0 {
            warn!("line search failed, revert to the previous point!");
            // Revert to the previous point.
            x.veccpy(&xp);
            g.veccpy(&gp);

            ret = ls;
            break;
        }

        // Compute x and g norms.
        let xnorm = x.vec2norm();
        let gnorm = if param.orthantwise_c == 0.0 {
            g.vec2norm()
        } else {
            pg.vec2norm()
        };

        // Report the progress.
        if let Some(ref mut prgr_fn) = proc_progress {
            let prgr = Progress {
                x: &x,
                gx: &g,
                fx,
                xnorm,
                gnorm,
                step,
                niter: k,
                ncall: ls as usize,
            };

            let cancel = prgr_fn(&prgr);
            if cancel {
                info!("canceled by callback function.");
                break;
            }
        }

        // Convergence test.
        // The criterion is given by the following formula:
        //     |g(x)| / \max(1, |x|) < \epsilon

        let xnorm = xnorm.max(1.0);
        if gnorm / xnorm <= param.epsilon {
            info!("lbfgs converged");
            // Convergence.
            ret = LBFGS_SUCCESS;
            break;
        }

        // Test for stopping criterion.
        // The criterion is given by the following formula:
        //    (f(past_x) - f(x)) / f(x) < \delta
        if pf.len() > 0 {
            // We don't test the stopping criterion while k < past.
            if param.past <= k {
                // Compute the relative improvement from the past.
                // rate = (*pf.offset((k % param.past) as isize) - fx) / fx;
                let rate = (pf[(k % param.past) as usize] - fx) / fx;
                // The stopping criterion.
                if rate < param.delta {
                    ret = LBFGS_STOP as i32;
                    break;
                }
            }
            // Store the current value of the objective function.
            // *pf.offset((k % param.past) as isize) = fx
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
        // it = &mut *lm.offset(end as isize) as *mut iteration_data_t;
        let mut it = &mut lm_arr[end];
        it.s.vecdiff(&x, &xp);
        it.y.vecdiff(&g, &gp);

        // Compute scalars ys and yy:
        // ys = y^t \cdot s = 1 / \rho.
        // yy = y^t \cdot y.
        // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
        let ys = it.y.vecdot(&it.s);
        let yy = it.y.vecdot(&it.y);

        (*it).ys = ys;

        // Recursive formula to compute dir = -(H \cdot g).
        // This is described in page 779 of:
        // Jorge Nocedal.
        // Updating Quasi-Newton Matrices with Limited Storage.
        // Mathematics of Computation, Vol. 35, No. 151,
        // pp. 773--782, 1980.
        let bound = if m <= k { m } else { k };

        k += 1;
        end = (end + 1) % m;
        // Compute the steepest direction.
        if param.orthantwise_c == 0.0 {
            // Compute the negative of gradients.
            d.vecncpy(&g);
        } else {
            d.vecncpy(&pg);
        }

        let mut i = 0;
        let mut j = end;
        while i < bound {
            // if (--j == -1) j = m-1;
            j = (j + m - 1) % m;
            // it = &mut *lm.offset(j as isize) as *mut iteration_data_t;
            let mut it = &mut lm_arr[j as usize];

            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}.
            it.alpha = it.s.vecdot(&d);
            it.alpha /= it.ys;
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}.
            d.vecadd(&it.y, -it.alpha);
            i += 1
        }
        d.vecscale(ys / yy);

        let mut i = 0;
        while i < bound {
            // it = &mut *lm.offset(j as isize) as *mut iteration_data_t;
            let it = &mut lm_arr[j as usize];
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}.
            let mut beta = 0.;
            beta = it.y.vecdot(&d);

            beta /= (*it).ys;
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}.
            d.vecadd(&it.s, it.alpha - beta);

            // if (++j == m) j = 0;
            j = (j + 1) % m;
            i += 1
        }

        // Constrain the search direction for orthant-wise updates.
        if param.orthantwise_c != 0.0 {
            let j = param.orthantwise_start as usize;
            let k = param.orthantwise_end as usize;
            for i in j..k {
                if d[i] * pg[i] >= 0.0 {
                    d[i] = 0.0;
                }
            }
        }

        // Now the search direction d is ready. We try step = 1 first.
        step = 1.0
    }

    // Return the final value of the objective function.
    *ptr_fx = fx;

    Ok(ret)
}
// old lbfgs:1 ends here
