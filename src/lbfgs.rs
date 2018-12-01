// header

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*header][header:1]]
//       Limited memory BFGS (L-BFGS).
//
//  Copyright (c) 1990, Jorge Nocedal
//  Copyright (c) 2007-2010 Naoaki Okazaki
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

// This library is a C port of the FORTRAN implementation of Limited-memory
// Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
// The original FORTRAN source code is available at:
// http://www.ece.northwestern.edu/~nocedal/lbfgs.html

// The L-BFGS algorithm is described in:
//     - Jorge Nocedal.
//       Updating Quasi-Newton Matrices with Limited Storage.
//       <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
//     - Dong C. Liu and Jorge Nocedal.
//       On the limited memory BFGS method for large scale optimization.
//       <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.

// The line search algorithms used in this implementation are described in:
//     - John E. Dennis and Robert B. Schnabel.
//       <i>Numerical Methods for Unconstrained Optimization and Nonlinear
//       Equations</i>, Englewood Cliffs, 1983.
//     - Jorge J. More and David J. Thuente.
//       Line search algorithm with guaranteed sufficient decrease.
//       <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
//       pp. 286-307, 1994.

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

use libc;

extern "C" {
    #[no_mangle]
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    #[no_mangle]
    fn free(__ptr: *mut libc::c_void);
    #[no_mangle]
    fn memset(_: *mut libc::c_void, _: libc::c_int, _: libc::c_ulong) -> *mut libc::c_void;
    #[no_mangle]
    fn memcpy(_: *mut libc::c_void, _: *const libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
}

pub type size_t = libc::c_ulong;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct tag_callback_data {
    pub n: libc::c_int,
    pub instance: *mut libc::c_void,
    pub proc_evaluate: lbfgs_evaluate_t,
    pub proc_progress: lbfgs_progress_t,
}

pub type lbfgsfloatval_t = libc::c_double;

/* *
 * \addtogroup liblbfgs_api libLBFGS API
 * @{
 *
 *  The libLBFGS API.
 */
/* *
 * Return values of lbfgs().
 *
 *  Roughly speaking, a negative value indicates an error.
 */
pub type unnamed = libc::c_int;
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

// Line search algorithms.
pub type unnamed_0 = libc::c_uint;
/* *
 * Backtracking method with strong Wolfe condition.
 *  The backtracking method finds the step length such that it satisfies
 *  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
 *  and the following condition,
 *    - |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
 *
 *  where x is the current point, d is the current search direction, and
 *  a is the step length.
 */

pub const LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE: unnamed_0 = 3;
//
// Backtracking method with regular Wolfe condition.
//  The backtracking method finds the step length such that it satisfies
//  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
//  and the curvature condition,
//    - g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d,
//
//  where x is the current point, d is the current search direction, and
//  a is the step length.

pub const LBFGS_LINESEARCH_BACKTRACKING_WOLFE: unnamed_0 = 2;
/* * The backtracking method with the defualt (regular Wolfe) condition. */
pub const LBFGS_LINESEARCH_BACKTRACKING: unnamed_0 = 2;
/* *
 * Backtracking method with the Armijo condition.
 *  The backtracking method finds the step length such that it satisfies
 *  the sufficient decrease (Armijo) condition,
 *    - f(x + a * d) <= f(x) + lbfgs_parameter_t::ftol * a * g(x)^T d,
 *
 *  where x is the current point, d is the current search direction, and
 *  a is the step length.
 */
pub const LBFGS_LINESEARCH_BACKTRACKING_ARMIJO: unnamed_0 = 1;
/* * MoreThuente method proposd by More and Thuente. */
pub const LBFGS_LINESEARCH_MORETHUENTE: unnamed_0 = 0;
/* * The default algorithm (MoreThuente method). */
pub const LBFGS_LINESEARCH_DEFAULT: unnamed_0 = 0;

/* *
 * Callback interface to provide objective function and gradient evaluations.
 *
 *  The lbfgs() function call this function to obtain the values of objective
 *  function and its gradients when needed. A client program must implement
 *  this function to evaluate the values of the objective function and its
 *  gradients, given current values of variables.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The gradient vector. The callback function must compute
 *                      the gradient values for the current variables.
 *  @param  n           The number of variables.
 *  @param  step        The current step of the line search routine.
 *  @retval lbfgsfloatval_t The value of the objective function for the current
 *                          variables.
 */
pub type lbfgs_evaluate_t = Option<
    unsafe extern "C" fn(
        _: *mut libc::c_void,
        _: *const lbfgsfloatval_t,
        _: *mut lbfgsfloatval_t,
        _: libc::c_int,
        _: lbfgsfloatval_t,
    ) -> lbfgsfloatval_t,
>;
/* *
 * Callback interface to receive the progress of the optimization process.
 *
 *  The lbfgs() function call this function for each iteration. Implementing
 *  this function, a client program can store or display the current progress
 *  of the optimization process.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The current gradient values of variables.
 *  @param  fx          The current value of the objective function.
 *  @param  xnorm       The Euclidean norm of the variables.
 *  @param  gnorm       The Euclidean norm of the gradients.
 *  @param  step        The line-search step used for this iteration.
 *  @param  n           The number of variables.
 *  @param  k           The iteration count.
 *  @param  ls          The number of evaluations called for this iteration.
 *  @retval int         Zero to continue the optimization process. Returning a
 *                      non-zero value will cancel the optimization process.
 */
pub type lbfgs_progress_t = Option<
    unsafe extern "C" fn(
        _: *mut libc::c_void,
        _: *const lbfgsfloatval_t,
        _: *const lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: lbfgsfloatval_t,
        _: libc::c_int,
        _: libc::c_int,
        _: libc::c_int,
    ) -> libc::c_int,
>;
pub type iteration_data_t = tag_iteration_data;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct tag_iteration_data {
    pub alpha: lbfgsfloatval_t,
    pub s: *mut lbfgsfloatval_t,
    pub y: *mut lbfgsfloatval_t,
    pub ys: lbfgsfloatval_t,
}
pub type callback_data_t = tag_callback_data;
// base:1 ends here

// parameter

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*parameter][parameter:1]]
/// L-BFGS optimization parameters.
///
/// Call lbfgs_parameter_t::default() function to initialize parameters to the
/// default values.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct lbfgs_parameter_t {
    /// The number of corrections to approximate the inverse hessian matrix.
    ///
    /// The L-BFGS routine stores the computation results of previous \ref m
    /// iterations to approximate the inverse hessian matrix of the current
    /// iteration. This parameter controls the size of the limited memories
    /// (corrections). The default value is \c 6. Values less than \c 3 are not
    /// recommended. Large values will result in excessive computing time.
    pub m: libc::c_int,

    /// Epsilon for convergence test.
    ///
    /// This parameter determines the accuracy with which the solution is to be
    /// found. A minimization terminates when
    ///
    ///     ||g|| < epsilon * max(1, ||x||),
    ///
    /// where ||.|| denotes the Euclidean (L2) norm. The default value is \c
    /// 1e-5.
    pub epsilon: lbfgsfloatval_t,

    /// Distance for delta-based convergence test.
    ///
    /// This parameter determines the distance, in iterations, to compute the
    /// rate of decrease of the objective function. If the value of this
    /// parameter is zero, the library does not perform the delta-based
    /// convergence test.
    ///
    /// The default value is 0.
    pub past: libc::c_int,

    /// Delta for convergence test.
    ///
    /// This parameter determines the minimum rate of decrease of the objective
    /// function. The library stops iterations when the following condition is
    /// met: (f' - f) / f < delta, where f' is the objective value of \ref past
    /// iterations ago, and f is the objective value of the current iteration.
    /// The default value is 0.
    ///
    pub delta: lbfgsfloatval_t,

    /// The maximum number of iterations.
    ///
    ///  The lbfgs() function terminates an optimization process with
    ///  ::LBFGSERR_MAXIMUMITERATION status code when the iteration count
    ///  exceedes this parameter. Setting this parameter to zero continues an
    ///  optimization process until a convergence or error.
    ///
    /// The default value is 0.
    pub max_iterations: libc::c_int,

    /// The line search algorithm.
    ///
    ///  This parameter specifies a line search algorithm to be used by the
    ///  L-BFGS routine.
    ///
    pub linesearch: libc::c_int,

    ///
    /// The maximum number of iterations.
    ///
    /// The lbfgs() function terminates an optimization process with
    /// ::LBFGSERR_MAXIMUMITERATION status code when the iteration count
    /// exceedes this parameter. Setting this parameter to zero continues an
    /// optimization process until a convergence or error.
    ///
    /// The default value is 0.
    pub max_linesearch: libc::c_int,

    /// The minimum step of the line search routine.
    ///
    /// The default value is \c 1e-20. This value need not be modified unless
    /// the exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    pub min_step: lbfgsfloatval_t,

    /// The maximum step of the line search.
    ///
    ///  The default value is \c 1e+20. This value need not be modified unless
    ///  the exponents are too large for the machine being used, or unless the
    ///  problem is extremely badly scaled (in which case the exponents should
    ///  be increased).
    pub max_step: lbfgsfloatval_t,

    /// A parameter to control the accuracy of the line search routine.
    ///
    ///  The default value is \c 1e-4. This parameter should be greater
    ///  than zero and smaller than \c 0.5.
    pub ftol: lbfgsfloatval_t,

    /// A parameter to control the accuracy of the line search routine.
    ///
    ///  The default value is \c 0.9. If the function and gradient
    ///  evaluations are inexpensive with respect to the cost of the
    ///  iteration (which is sometimes the case when solving very large
    ///  problems) it may be advantageous to set this parameter to a small
    ///  value. A typical small value is \c 0.1. This parameter shuold be
    ///  greater than the \ref ftol parameter (\c 1e-4) and smaller than
    ///  \c 1.0.
    pub gtol: lbfgsfloatval_t,

    /// The machine precision for floating-point values.
    ///
    ///  This parameter must be a positive value set by a client program to
    ///  estimate the machine precision. The line search routine will terminate
    ///  with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
    ///  of the interval of uncertainty is less than this parameter.
    pub xtol: lbfgsfloatval_t,

    /// A coefficient for the Wolfe condition.
    ///
    ///  This parameter is valid only when the backtracking line-search
    ///  algorithm is used with the Wolfe condition,
    ///  ::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
    ///  ::LBFGS_LINESEARCH_BACKTRACKING_WOLFE. The default value is \c 0.9.
    ///  This parameter should be greater the \ref ftol parameter and smaller
    ///  than \c 1.0.
    pub wolfe: lbfgsfloatval_t,

    /// Coeefficient for the L1 norm of variables.
    ///
    ///  This parameter should be set to zero for standard minimization
    ///  problems. Setting this parameter to a positive value activates
    ///  Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
    ///  minimizes the objective function F(x) combined with the L1 norm |x|
    ///  of the variables, {F(x) + C |x|}. This parameter is the coeefficient
    ///  for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
    ///  zero, the library modifies function and gradient evaluations from
    ///  a client program suitably; a client program thus have only to return
    ///  the function value F(x) and gradients G(x) as usual. The default value
    ///  is zero.
    pub orthantwise_c: lbfgsfloatval_t,

    /// Start index for computing L1 norm of the variables.
    ///
    /// This parameter is valid only for OWL-QN method
    /// (i.e., \ref orthantwise_c != 0). This parameter b (0 <= b < N)
    /// specifies the index number from which the library computes the
    /// L1 norm of the variables x,
    ///
    ///     |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
    ///
    /// In other words, variables x_1, ..., x_{b-1} are not used for
    /// computing the L1 norm. Setting b (0 < b < N), one can protect
    /// variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
    /// regression) from being regularized. The default value is zero.
    pub orthantwise_start: libc::c_int,

    /// End index for computing L1 norm of the variables.
    ///
    /// This parameter is valid only for OWL-QN method
    /// (i.e., \ref orthantwise_c != 0). This parameter e (0 < e <= N)
    /// specifies the index number at which the library stops computing the
    /// L1 norm of the variables x,
    pub orthantwise_end: libc::c_int,
}

impl Default for lbfgs_parameter_t {
    /// Initialize L-BFGS parameters to the default values.
    ///
    /// Call this function to fill a parameter structure with the default values
    /// and overwrite parameter values if necessary.
    fn default() -> Self {
        lbfgs_parameter_t {
            m: 6,
            epsilon: 1e-5,
            past: 0,
            delta: 1e-5,
            max_iterations: 0,
            linesearch: LBFGS_LINESEARCH_DEFAULT as libc::c_int,
            max_linesearch: 40,
            min_step: 1e-20,
            max_step: 1e20,
            ftol: 1e-4,
            wolfe: 0.9,
            gtol: 0.9,
            xtol: 1.0e-16,
            orthantwise_c: 0.0,
            orthantwise_start: 0,
            orthantwise_end: -1,
        }
    }
}
// parameter:1 ends here

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
pub struct LinesearchOption {
    /// ftol and gtol are nonnegative input variables. (in this reverse
    /// communication implementation gtol is defined in a common statement.)
    ///
    /// Termination occurs when the sufficient decrease condition and the
    /// directional derivative condition are satisfied.
    ftol: f64,
    gtol: f64,

    /// xtol is a nonnegative input variable. termination occurs when the
    /// relative width of the interval of uncertainty is at most xtol.
    xtol: f64,

    max_step: f64,

    min_step: f64,

    max_linesearch: usize,
}

// TODO: better defaults
impl Default for LinesearchOption {
    fn default() -> Self {
        LinesearchOption {
            ftol: 1e-4,
            gtol: 1e-4,
            xtol: 1e-4,
            max_step: 1.0,
            min_step: 1e-3,
            max_linesearch: 20,
        }
    }
}

trait LineSearching {
    fn find();
}
// new:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
pub type line_search_proc = Option<
    unsafe extern "C" fn(
        // The number of variables.
        n: libc::c_int,
        // The array of variables.
        x: *mut lbfgsfloatval_t,
        // Evaluated function value
        f: *mut lbfgsfloatval_t,
        // Evaluated gradient array
        g: *mut lbfgsfloatval_t,
        // Search direction array
        s: *mut lbfgsfloatval_t,
        // Step size
        stp: *mut lbfgsfloatval_t,
        // Variable vector of previous step
        xp: *const lbfgsfloatval_t,
        // Gradient vector of previous step
        gp: *const lbfgsfloatval_t,
        // work array?
        wp: *mut lbfgsfloatval_t,
        // callback struct
        cd: *mut callback_data_t,
        // LBFGS parameter
        param: &lbfgs_parameter_t,
    ) -> libc::c_int,
>;
// old:1 ends here

// Original documentation by J. Nocera (lbfgs.f)
//                 subroutine mcsrch

// A slight modification of the subroutine CSRCH of More' and Thuente.
// The changes are to allow reverse communication, and do not affect
// the performance of the routine.

// The purpose of mcsrch is to find a step which satisfies
// a sufficient decrease condition and a curvature condition.

// At each stage the subroutine updates an interval of
// uncertainty with endpoints stx and sty. the interval of
// uncertainty is initially chosen so that it contains a
// minimizer of the modified function

//      f(x+stp*s) - f(x) - ftol*stp*(gradf(x)'s).

// If a step is obtained for which the modified function
// has a nonpositive function value and nonnegative derivative,
// then the interval of uncertainty is chosen so that it
// contains a minimizer of f(x+stp*s).

// The algorithm is designed to find a step which satisfies
// the sufficient decrease condition

//       f(x+stp*s) <= f(x) + ftol*stp*(gradf(x)'s),

// and the curvature condition

//       abs(gradf(x+stp*s)'s)) <= gtol*abs(gradf(x)'s).

// If ftol is less than gtol and if, for example, the function
// is bounded below, then there is always a step which satisfies
// both conditions. if no step can be found which satisfies both
// conditions, then the algorithm usually stops when rounding
// errors prevent further progress. in this case stp only
// satisfies the sufficient decrease condition.

// The subroutine statement is

//    subroutine mcsrch(n,x,f,g,s,stp,ftol,xtol, maxfev,info,nfev,wa)
// where

//   n is a positive integer input variable set to the number
//     of variables.

//   x is an array of length n. on input it must contain the
//     base point for the line search. on output it contains
//     x + stp*s.

//   f is a variable. on input it must contain the value of f
//     at x. on output it contains the value of f at x + stp*s.

//   g is an array of length n. on input it must contain the
//     gradient of f at x. on output it contains the gradient
//     of f at x + stp*s.

//   s is an input array of length n which specifies the
//     search direction.

//   stp is a nonnegative variable. on input stp contains an
//     initial estimate of a satisfactory step. on output
//     stp contains the final estimate.

//   ftol and gtol are nonnegative input variables. (in this reverse
//     communication implementation gtol is defined in a common
//     statement.) termination occurs when the sufficient decrease
//     condition and the directional derivative condition are
//     satisfied.

//   xtol is a nonnegative input variable. termination occurs
//     when the relative width of the interval of uncertainty
//     is at most xtol.

//   stpmin and stpmax are nonnegative input variables which
//     specify lower and upper bounds for the step. (In this reverse
//     communication implementatin they are defined in a common
//     statement).

//   maxfev is a positive integer input variable. termination
//     occurs when the number of calls to fcn is at least
//     maxfev by the end of an iteration.

//   info is an integer output variable set as follows:

//     info = 0  improper input parameters.

//     info =-1  a return is made to compute the function and gradient.

//     info = 1  the sufficient decrease condition and the
//               directional derivative condition hold.

//     info = 2  relative width of the interval of uncertainty
//               is at most xtol.

//     info = 3  number of calls to fcn has reached maxfev.

//     info = 4  the step is at the lower bound stpmin.

//     info = 5  the step is at the upper bound stpmax.

//     info = 6  rounding errors prevent further progress.
//               there may not be a step which satisfies the
//               sufficient decrease and curvature conditions.
//               tolerances may be too small.

//   nfev is an integer output variable set to the number of
//     calls to fcn.

//   wa is a work array of length n.

// subprograms called

//   mcstep

//   fortran-supplied...abs,max,min

// ARgonne National Laboratory. Minpack Project. June 1983
// Jorge J. More', David J. Thuente


// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*Original%20documentation%20by%20J.%20Nocera%20(lbfgs.f)][Original documentation by J. Nocera (lbfgs.f):1]]

// Original documentation by J. Nocera (lbfgs.f):1 ends here

// src

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*src][src:1]]
/// The purpose of mcsrch is to find a step which satisfies a sufficient
/// decrease condition and a curvature condition.
mod mcsrch {
    // dependencies
    use super::mcstep;
    use super::LbfgsMath;
    use super::LinesearchOption;
    use quicli::prelude::{bail, Result};

    /// A struct represents MCSRCH subroutine in original lbfgs.f by J. Nocera
    struct Mcsrch<'a> {
        /// x is an array of length n. on input it must contain the base point for
        /// the line search. on output it contains x + stp*s.
        x: &'a mut [f64],

        /// f is a variable. on input it must contain the value of f at x. on
        /// output it contains the value of f at x + stp*s.
        f: f64,

        /// g is an array of length n. on input it must contain the gradient of f at
        /// x. on output it contains the gradient of f at x + stp*s.
        g: &'a mut [f64],

        /// s is an input array of length n which specifies the search direction.
        s: &'a [f64],

        /// stp is a nonnegative variable. on input stp contains an initial estimate
        /// of a satisfactory step. on output stp contains the final estimate.
        stp: f64,

        param: LinesearchOption,
    }

    impl<'a> Mcsrch<'a> {
        /// # Arguments
        ///
        /// - direction: search direction array
        fn new(direction: &[f64]) -> Self {
            unimplemented!()
        }

        /// Find a step which satisfies a sufficient decrease condition and a
        /// curvature condition (strong wolfe conditions).
        ///
        /// # Arguments
        ///
        /// * cd    : a closure to evaluate the function and gradient values.
        /// * xp    : an array of input variables in previous step
        /// * param : line search options
        ///
        /// # Example
        ///
        /// - TODO
        ///
        pub fn find<F>(&mut self, mut cd: F, xp: &[f64]) -> Result<usize>
        where
            F: FnMut(&[f64], &mut [f64]) -> Result<f64>,
        {
            // Check the input parameters for errors.
            if !self.stp.is_sign_positive() {
                bail!("A logic error (negative line-search step) occurred.");
            }

            // Compute the initial gradient in the search direction.
            let s = &self.s;
            let g = &mut self.g;
            // vecdot(&mut dginit, g, s, n);
            let mut dginit = (*g).vecdot(*s);

            // Make sure that s points to a descent direction.
            if 0.0 < dginit {
                bail!("The current search direction increases the objective function value!");
            }

            // Initialize local variables.
            let param = &self.param;
            let mut brackt = false;
            let mut stage1 = 1;
            let f = &mut self.f;
            let finit = *f;
            let dgtest = param.ftol * dginit;
            let mut width = param.max_step - param.min_step;
            let mut prev_width = 2.0 * width;

            // The variables stx, fx, dgx contain the values of the step,
            // function, and directional derivative at the best step.
            // The variables sty, fy, dgy contain the value of the step,
            // function, and derivative at the other endpoint of
            // the interval of uncertainty.
            // The variables stp, f, dg contain the values of the step,
            // function, and derivative at the current step.
            let mut stx = 0f64;
            let mut sty = 0f64;
            let mut fx = finit;
            let mut fy = finit;
            let mut dgx = dginit;
            let mut dgy = dginit;
            let mut dg = 0f64;

            let mut stmin = 0.;
            let mut stmax = 0.;
            let mut uinfo = 0;
            let mut count = 0;

            let mut fxm = 0.;
            let mut dgxm = 0.;
            let mut fym = 0.;
            let mut dgym = 0.;
            let mut ftest1 = 0.;
            let mut fm = 0.;
            let mut dgm = 0.;

            let x = &mut self.x; // input variables
            let f = &mut self.f; // function value
            let stp = &mut self.stp; // step length
            loop {
                // Set the minimum and maximum steps to correspond to the
                // present interval of uncertainty.
                if brackt {
                    stmin = stx.min(sty);
                    stmax = stx.max(sty);
                } else {
                    stmin = stx;
                    stmax = *stp + 4.0 * (*stp - stx)
                }

                // Clip the step in the range of [stpmin, stpmax].
                if *stp < param.min_step {
                    *stp = param.min_step
                }
                if param.max_step < *stp {
                    *stp = param.max_step
                }

                // If an unusual termination is to occur then let
                // stp be the lowest point obtained so far.
                if brackt
                    && (*stp <= stmin
                        || stmax <= *stp
                        || param.max_linesearch <= count + 1
                        || uinfo != 0)
                    || brackt && stmax - stmin <= param.xtol * stmax
                {
                    *stp = stx
                }

                // Compute the current value of x: x <- x + (*stp) * s.
                // veccpy(x, xp, n);
                (*x).veccpy(&xp);
                // vecadd(x, s, *stp, n);
                (*x).vecadd(&s, *stp);

                // Evaluate the function and gradient values.
                // *f = cd(x, g, *stp)?;
                *f = cd(x, g)?;

                // vecdot(&mut dg, g, s, n);
                dg = g.vecdot(s);

                ftest1 = finit + *stp * dgtest;
                count += 1;

                // Test for errors and convergence.
                if brackt && (*stp <= stmin || stmax <= *stp || uinfo != 0) {
                    /* Rounding errors prevent further progress. */
                    // return LBFGSERR_ROUNDING_ERROR as libc::c_int;
                    bail!("LBFGSERR_ROUNDING_ERROR");
                }
                if *stp == param.max_step && *f <= ftest1 && dg <= dgtest {
                    /* The step is the maximum value. */
                    // return LBFGSERR_MAXIMUMSTEP as libc::c_int;
                    bail!("LBFGSERR_MAXIMUMSTEP");
                }
                if *stp == param.min_step && (ftest1 < *f || dgtest <= dg) {
                    /* The step is the minimum value. */
                    // return LBFGSERR_MINIMUMSTEP as libc::c_int;
                    bail!("LBFGSERR_MINIMUMSTEP");
                }
                if brackt && stmax - stmin <= param.xtol * stmax {
                    /* Relative width of the interval of uncertainty is at most xtol. */
                    // return LBFGSERR_WIDTHTOOSMALL as libc::c_int;
                    bail!("LBFGSERR_WIDTHTOOSMALL");
                }
                if param.max_linesearch <= count {
                    // Maximum number of iteration.
                    // return LBFGSERR_MAXIMUMLINESEARCH as libc::c_int;
                    bail!("LBFGSERR_MAXIMUMLINESEARCH");
                }
                if *f <= ftest1 && dg.abs() <= param.gtol * -dginit {
                    // The sufficient decrease condition and the directional derivative condition hold.
                    return Ok(count);
                }

                // In the first stage we seek a step for which the modified
                // function has a nonpositive value and nonnegative derivative.
                if 0 != stage1 && *f <= ftest1 && param.ftol.min(param.gtol) * dginit <= dg {
                    stage1 = 0
                }

                // A modified function is used to predict the step only if
                // we have not obtained a step for which the modified
                // function has a nonpositive function value and nonnegative
                // derivative, and if a lower function value has been
                // obtained but the decrease is not sufficient.
                if 0 != stage1 && ftest1 < *f && *f <= fx {
                    // Define the modified function and derivative values.
                    fm = *f - *stp * dgtest;
                    fxm = fx - stx * dgtest;
                    fym = fy - sty * dgtest;
                    dgm = dg - dgtest;
                    dgxm = dgx - dgtest;
                    dgym = dgy - dgtest;

                    // Call update_trial_interval() to update the interval of
                    // uncertainty and to compute the new step.
                    uinfo = mcstep::update_trial_interval(
                        &mut stx,
                        &mut fxm,
                        &mut dgxm,
                        &mut sty,
                        &mut fym,
                        &mut dgym,
                        stp,
                        fm,
                        dgm,
                        stmin,
                        stmax,
                        &mut brackt,
                    )?;

                    // Reset the function and gradient values for f.
                    fx = fxm + stx * dgtest;
                    fy = fym + sty * dgtest;
                    dgx = dgxm + dgtest;
                    dgy = dgym + dgtest
                } else {
                    uinfo = mcstep::update_trial_interval(
                        &mut stx,
                        &mut fx,
                        &mut dgx,
                        &mut sty,
                        &mut fy,
                        &mut dgy,
                        stp,
                        *f,
                        dg,
                        stmin,
                        stmax,
                        &mut brackt,
                    )?;
                }

                // Force a sufficient decrease in the interval of uncertainty.
                if brackt {
                    if 0.66 * prev_width <= (sty - stx).abs() {
                        *stp = stx + 0.5 * (sty - stx)
                    }

                    prev_width = width;
                    width = (sty - stx).abs()
                }
            }

            bail!("Logic error!");
        }
    }
}
// src:1 ends here

// lbfgs.c

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*lbfgs.c][lbfgs.c:1]]
unsafe extern "C" fn line_search_morethuente(
    n: libc::c_int,
    mut x: *mut lbfgsfloatval_t,
    mut f: *mut lbfgsfloatval_t,
    mut g: *mut lbfgsfloatval_t,
    mut s: *mut lbfgsfloatval_t,
    mut stp: *mut lbfgsfloatval_t,
    mut xp: *const lbfgsfloatval_t,
    mut gp: *const lbfgsfloatval_t,
    mut wa: *mut lbfgsfloatval_t,
    mut cd: *mut callback_data_t,
    param: &lbfgs_parameter_t,
) -> libc::c_int {
    // Check the input parameters for errors.
    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Compute the initial gradient in the search direction.
    let mut dginit = 0.0;
    vecdot(&mut dginit, g, s, n);

    // Make sure that s points to a descent direction.
    if 0.0 < dginit {
        return LBFGSERR_INCREASEGRADIENT as libc::c_int;
    }

    // Initialize local variables.
    let mut brackt = false;
    let mut stage1 = 1;
    let mut uinfo  = 0;

    let mut finit = *f;
    let dgtest = param.ftol * dginit;
    let mut width = param.max_step - param.min_step;
    let mut prev_width = 2.0f64 * width;

    // The variables stx, fx, dgx contain the values of the step,
    // function, and directional derivative at the best step.
    // The variables sty, fy, dgy contain the value of the step,
    // function, and derivative at the other endpoint of
    // the interval of uncertainty.
    // The variables stp, f, dg contain the values of the step,
    // function, and derivative at the current step.
    let mut stx = 0.0;
    let mut sty = 0.0;
    let mut fx = finit;
    let mut fy = finit;
    let mut dgy = dginit;
    let mut dgx = dgy;

    let mut count = 0;
    let mut stmin = 0.;
    let mut stmax = 0.;
    loop {
        // Set the minimum and maximum steps to correspond to the
        // present interval of uncertainty.
        if brackt {
            stmin = if stx <= sty { stx } else { sty };
            stmax = if stx >= sty { stx } else { sty }
        } else {
            stmin = stx;
            stmax = *stp + 4.0f64 * (*stp - stx)
        }

        // Clip the step in the range of [stpmin, stpmax].
        if *stp < param.min_step {
            *stp = param.min_step
        }
        if param.max_step < *stp {
            *stp = param.max_step
        }

        // If an unusual termination is to occur then let
        // stp be the lowest point obtained so far.
        if brackt
            && (*stp <= stmin
                || stmax <= *stp
                || param.max_linesearch <= count + 1i32
                || uinfo != 0i32) || brackt && stmax - stmin <= param.xtol * stmax
        {
            *stp = stx
        }

        // Compute the current value of x: x <- x + (*stp) * s.
        veccpy(x, xp, n);
        vecadd(x, s, *stp, n);

        // Evaluate the function and gradient values.
        *f = (*cd).proc_evaluate.expect("non-null function pointer")(
            (*cd).instance,
            x,
            g,
            (*cd).n,
            *stp,
        );

        let mut dg = 0.;
        vecdot(&mut dg, g, s, n);
        let ftest1 = finit + *stp * dgtest;
        count += 1;

        // Test for errors and convergence.
        if brackt && (*stp <= stmin || stmax <= *stp || uinfo != 0i32) {
            /* Rounding errors prevent further progress. */
            return LBFGSERR_ROUNDING_ERROR as libc::c_int;
        } else if *stp == param.max_step && *f <= ftest1 && dg <= dgtest {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP as libc::c_int;
        } else if *stp == param.min_step && (ftest1 < *f || dgtest <= dg) {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP as libc::c_int;
        } else if brackt && stmax - stmin <= param.xtol * stmax {
            /* Relative width of the interval of uncertainty is at most xtol. */
            return LBFGSERR_WIDTHTOOSMALL as libc::c_int;
        } else if param.max_linesearch <= count {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH as libc::c_int;
        } else if *f <= ftest1 && dg.abs() <= param.gtol * -dginit {
            /* The sufficient decrease condition and the directional derivative condition hold. */
            return count;
        } else {
            // In the first stage we seek a step for which the modified
            // function has a nonpositive value and nonnegative derivative.
            if 0 != stage1 && *f <= ftest1 && param.ftol.min(param.gtol) * dginit <= dg
            {
                stage1 = 0;
            }

            // A modified function is used to predict the step only if
            // we have not obtained a step for which the modified
            // function has a nonpositive function value and nonnegative
            // derivative, and if a lower function value has been
            // obtained but the decrease is not sufficient.
            if 0 != stage1 && ftest1 < *f && *f <= fx {
                // Define the modified function and derivative values.
                let fm = *f - *stp * dgtest;
                let mut fxm = fx - stx * dgtest;
                let mut fym = fy - sty * dgtest;
                let dgm = dg - dgtest;
                let mut dgxm = dgx - dgtest;
                let mut dgym = dgy - dgtest;

                // Call update_trial_interval() to update the interval of
                // uncertainty and to compute the new step.
                uinfo = mcstep::update_trial_interval(
                    &mut stx,
                    &mut fxm,
                    &mut dgxm,
                    &mut sty,
                    &mut fym,
                    &mut dgym,
                    &mut *stp,
                    fm,
                    dgm,
                    stmin,
                    stmax,
                    &mut brackt,
                ).expect("FIXME");

                // Reset the function and gradient values for f.
                fx = fxm + stx * dgtest;
                fy = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest
            } else {
                uinfo = mcstep::update_trial_interval(
                    &mut stx,
                    &mut fx,
                    &mut dgx,
                    &mut sty,
                    &mut fy,
                    &mut dgy,
                    &mut *stp,
                    *f,
                    dg,
                    stmin,
                    stmax,
                    &mut brackt,
                ).expect("FIXME")
            }

            // Force a sufficient decrease in the interval of uncertainty.
            if !(brackt) {
                continue;
            }

            if 0.66f64 * prev_width <= (sty - stx).abs() {
                *stp = stx + 0.5f64 * (sty - stx)
            }

            prev_width = width;
            width = (sty - stx).abs()
        }
    }
}
// lbfgs.c:1 ends here

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
/// Represents the original MCSTEP subroutine by J. Nocera, which is a variant
/// of More' and Thuente's routine.
///
/// The purpose of mcstep is to compute a safeguarded step for a linesearch and
/// to update an interval of uncertainty for a minimizer of the function.
///
/// Documentation is adopted from the original Fortran codes.
mod mcstep {
    // dependencies
    use super::{
        cubic_minimizer,
        cubic_minimizer2,
        quard_minimizer,
        quard_minimizer2,
    };

    use quicli::prelude::{
        bail,
        Result
    };

    ///
    /// Update a safeguarded trial value and interval for line search.
    ///
    /// This function assumes that the derivative at the point of x in the
    /// direction of the step. If the bracket is set to true, the minimizer has
    /// been bracketed in an interval of uncertainty with endpoints between x
    /// and y.
    ///
    /// # Arguments
    ///
    /// * x, fx, and dx: variables which specify the step, the function, and the
    /// derivative at the best step obtained so far. The derivative must be
    /// negative in the direction of the step, that is, dx and t-x must have
    /// opposite signs. On output these parameters are updated appropriately.
    ///
    /// * y, fy, and dy: variables which specify the step, the function, and
    /// the derivative at the other endpoint of the interval of uncertainty. On
    /// output these parameters are updated appropriately.
    ///
    /// * t, ft, and dt: variables which specify the step, the function, and the
    /// derivative at the current step. If bracket is set true then on input t
    /// must be between x and y. On output t is set to the new step.
    ///
    /// * tmin, tmax: lower and upper bounds for the step.
    ///
    /// * `brackt`: Specifies if a minimizer has been bracketed. If the
    /// minimizer has not been bracketed then on input brackt must be set false.
    /// If the minimizer is bracketed then on output `brackt` is set true.
    ///
    /// # Return
    /// - Status value. Zero indicates a normal termination.
    ///
    pub(crate) fn update_trial_interval(
        x: &mut f64,
        fx: &mut f64,
        dx: &mut f64,
        y: &mut f64,
        fy: &mut f64,
        dy: &mut f64,
        t: &mut f64,
        ft: f64,
        dt: f64,
        tmin: f64,
        tmax: f64,
        brackt: &mut bool,
    ) -> Result<i32> {
        let mut bound = 0;
        // fsigndiff
        let mut dsign = (dt * (*dx / (*dx).abs()) < 0.0) as libc::c_int;
        // minimizer of an interpolated cubic.
        let mut mc = 0.;
        // minimizer of an interpolated quadratic.
        let mut mq = 0.;
        // new trial value.
        let mut newt = 0.;

        // Check the input parameters for errors.
        if *brackt {
            if *t <= x.min(*y) || x.max(*y) <= *t {
                // The trival value t is out of the interval.
                // return LBFGSERR_OUTOFINTERVAL as libc::c_int;
                bail!("LBFGSERR_OUTOFINTERVAL");
            } else if 0.0 <= *dx * (*t - *x) {
                /* The function must decrease from x. */
                // return LBFGSERR_INCREASEGRADIENT as libc::c_int;
                bail!("LBFGSERR_INCREASEGRADIENT");
            } else if tmax < tmin {
                /* Incorrect tmin and tmax specified. */
                // return LBFGSERR_INCORRECT_TMINMAX as libc::c_int;
                bail!("LBFGSERR_INCORRECT_TMINMAX");
            }
        }

        // Trial value selection.
        let mut p = 0.;
        let mut q = 0.;
        let mut r = 0.;
        let mut s = 0.;
        if *fx < ft {
            // Case 1: a higher function value.
            // The minimum is brackt. If the cubic minimizer is closer
            // to x than the quadratic one, the cubic one is taken, else
            // the average of the minimizers is taken.
            *brackt = true;
            bound = 1;
            cubic_minimizer(&mut mc, *x, *fx, *dx, *t, ft, dt);
            quard_minimizer(&mut mq, *x, *fx, *dx, *t, ft);
            if (mc - *x).abs() < (mq - *x).abs() {
                newt = mc
            } else {
                newt = mc + 0.5 * (mq - mc)
            }
        } else if 0 != dsign {
            // Case 2: a lower function value and derivatives of
            // opposite sign. The minimum is brackt. If the cubic
            // minimizer is closer to x than the quadratic (secant) one,
            // the cubic one is taken, else the quadratic one is taken.
            *brackt = true;
            bound = 0;
            cubic_minimizer(&mut mc, *x, *fx, *dx, *t, ft, dt);
            quard_minimizer2(&mut mq, *x, *dx, *t, dt);
            if (mc - *t).abs() > (mq - *t).abs() {
                newt = mc
            } else {
                newt = mq
            }
        } else if dt.abs() < (*dx).abs() {
            // Case 3: a lower function value, derivatives of the
            // same sign, and the magnitude of the derivative decreases.
            // The cubic minimizer is only used if the cubic tends to
            // infinity in the direction of the minimizer or if the minimum
            // of the cubic is beyond t. Otherwise the cubic minimizer is
            // defined to be either tmin or tmax. The quadratic (secant)
            // minimizer is also computed and if the minimum is brackt
            // then the the minimizer closest to x is taken, else the one
            // farthest away is taken.
            bound = 1;
            cubic_minimizer2(&mut mc, *x, *fx, *dx, *t, ft, dt, tmin, tmax);
            quard_minimizer2(&mut mq, *x, *dx, *t, dt);
            if *brackt {
                if (*t - mc).abs() < (*t - mq).abs() {
                    newt = mc
                } else {
                    newt = mq
                }
            } else if (*t - mc).abs() > (*t - mq).abs() {
                newt = mc
            } else {
                newt = mq
            }
        } else {
            // Case 4: a lower function value, derivatives of the
            // same sign, and the magnitude of the derivative does
            // not decrease. If the minimum is not brackt, the step
            // is either tmin or tmax, else the cubic minimizer is taken.

            bound = 0;
            if *brackt {
                cubic_minimizer(&mut newt, *t, ft, dt, *y, *fy, *dy);
            } else if *x < *t {
                newt = tmax
            } else {
                newt = tmin
            }
        }

        // Update the interval of uncertainty. This update does not
        // depend on the new step or the case analysis above.
        // - Case a: if f(x) < f(t),
        //    x <- x, y <- t.
        // - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
        //   x <- t, y <- y.
        // - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
        //   x <- t, y <- x.
        if *fx < ft {
            /* Case a */
            *y = *t;
            *fy = ft;
            *dy = dt
        } else {
            /* Case c */
            if 0 != dsign {
                *y = *x;
                *fy = *fx;
                *dy = *dx
            }
            /* Cases b and c */
            *x = *t;
            *fx = ft;
            *dx = dt
        }

        // Clip the new trial value in [tmin, tmax].
        if tmax < newt {
            newt = tmax
        }
        if newt < tmin {
            newt = tmin
        }

        // Redefine the new trial value if it is close to the upper bound of the
        // interval.
        if *brackt && 0 != bound {
            mq = *x + 0.66 * (*y - *x);
            if *x < *y {
                if mq < newt {
                    newt = mq
                }
            } else if newt < mq {
                newt = mq
            }
        }

        // Return the new trial value.
        *t = newt;

        Ok(0)
    }
}
// new:1 ends here

// interpolation

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*interpolation][interpolation:1]]
/// Find a minimizer of an interpolated cubic function.
///
/// # Arguments
///  * `cm`: The minimizer of the interpolated cubic.
///  * `u` : The value of one point, u.
///  * `fu`: The value of f(u).
///  * `du`: The value of f'(u).
///  * `v` : The value of another point, v.
///  * `fv`: The value of f(v).
///  * `dv`:  The value of f'(v).
#[inline]
fn cubic_minimizer(cm: &mut f64, u: f64, fu: f64, du: f64, v: f64, fv: f64, dv: f64) {
    let d = v - u;
    let theta = (fu - fv) * 3.0 / d + du + dv;

    let mut p = theta.abs();
    let mut q = du.abs();
    let mut r = dv.abs();
    let s = (p.max(q)).max(r); // max3(p, q, r)
    let a = theta / s;
    let mut gamma = s * (a * a - du / s * (dv / s)).sqrt();
    if v < u {
        gamma = -gamma
    }
    p = gamma - du + theta;
    q = gamma - du + gamma + dv;
    r = p / q;
    *cm = u + r * d;
}

/// Find a minimizer of an interpolated cubic function.
///
/// # Arguments
///  * cm  :   The minimizer of the interpolated cubic.
///  * u   :   The value of one point, u.
///  * fu  :   The value of f(u).
///  * du  :   The value of f'(u).
///  * v   :   The value of another point, v.
///  * fv  :   The value of f(v).
///  * dv  :   The value of f'(v).
///  * xmin:   The minimum value.
///  * xmax:   The maximum value.
#[inline]
fn cubic_minimizer2(
    cm   : &mut f64,
    u    : f64,
    fu   : f64,
    du   : f64,
    v    : f64,
    fv   : f64,
    dv   : f64,
    xmin : f64,
    xmax : f64,
) {
    let d = v - u;
    let theta = (fu - fv) * 3.0 / d + du + dv;
    let mut p = theta.abs();
    let mut q = du.abs();
    let mut r = dv.abs();
    // s = max3(p, q, r);
    let s = (p.max(q)).max(r); // max3(p, q, r)
    let a = theta / s;

    let mut gamma = s * (0f64.max(a * a - du / s * (dv / s)).sqrt());
    if u < v {
        gamma = -gamma
    }
    p = gamma - dv + theta;
    q = gamma - dv + gamma + du;
    r = p / q;
    if r < 0.0 && gamma != 0.0 {
        *cm = v - r * d;
    } else if a < 0 as f64 {
        *cm = xmax;
    } else {
        *cm = xmin;
    }
}

/// Find a minimizer of an interpolated quadratic function.
///
/// # Arguments
/// * qm : The minimizer of the interpolated quadratic.
/// * u  : The value of one point, u.
/// * fu : The value of f(u).
/// * du : The value of f'(u).
/// * v  : The value of another point, v.
/// * fv : The value of f(v).
#[inline]
fn quard_minimizer(qm: &mut f64, u: f64, fu: f64, du: f64, v: f64, fv: f64) {
    let a = v - u;
    *qm = u + du / ((fu - fv) / a + du) / 2.0 * a;
}

/// Find a minimizer of an interpolated quadratic function.
///
/// # Arguments
/// * `qm` :    The minimizer of the interpolated quadratic.
/// * `u`  :    The value of one point, u.
/// * `du` :    The value of f'(u).
/// * `v`  :    The value of another point, v.
/// * `dv` :    The value of f'(v).
#[inline]
fn quard_minimizer2(qm: &mut f64, u: f64, du: f64, v: f64, dv: f64) {
    let a = u - v;
    *qm = v + dv / (dv - du) * a;
}
// interpolation:1 ends here

// unsafe

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*unsafe][unsafe:1]]
unsafe extern "C" fn update_trial_interval(
    x: *mut lbfgsfloatval_t,
    fx: *mut lbfgsfloatval_t,
    dx: *mut lbfgsfloatval_t,
    y: *mut lbfgsfloatval_t,
    fy: *mut lbfgsfloatval_t,
    dy: *mut lbfgsfloatval_t,
    t: *mut lbfgsfloatval_t,
    ft: *const lbfgsfloatval_t,
    dt: *const lbfgsfloatval_t,
    tmin: lbfgsfloatval_t,
    tmax: lbfgsfloatval_t,
    brackt: *mut libc::c_int,
) -> libc::c_int {
    let mut bound: libc::c_int = 0;
    let mut dsign: libc::c_int = (*dt * (*dx / (*dx).abs()) < 0.0f64) as libc::c_int;
    // minimizer of an interpolated cubic.
    let mut mc = 0.;
    // minimizer of an interpolated quadratic.
    let mut mq = 0.;
    // new trial value.
    let mut newt = 0.;

    // Check the input parameters for errors.
    if 0 != *brackt {
        if *t <= if *x <= *y { *x } else { *y } || if *x >= *y { *x } else { *y } <= *t {
            /* The trival value t is out of the interval. */
            return LBFGSERR_OUTOFINTERVAL as libc::c_int;
        } else if 0.0f64 <= *dx * (*t - *x) {
            /* The function must decrease from x. */
            return LBFGSERR_INCREASEGRADIENT as libc::c_int;
        } else if tmax < tmin {
            /* Incorrect tmin and tmax specified. */
            return LBFGSERR_INCORRECT_TMINMAX as libc::c_int;
        }
    }

    // Trial value selection.
    let mut p = 0.;
    let mut q = 0.;
    let mut r = 0.;
    let mut s = 0.;

    if *fx < *ft {
        // Case 1: a higher function value.
        // The minimum is brackt. If the cubic minimizer is closer
        // to x than the quadratic one, the cubic one is taken, else
        // the average of the minimizers is taken.
        *brackt = 1;
        bound = 1;
        cubic_minimizer(&mut mc, *x, *fx, *dx, *t, *ft, *dt);
        quard_minimizer(&mut mq, *x, *fx, *dx, *t, *ft);
        if (mc - *x).abs() < (mq - *x).abs() {
            newt = mc
        } else {
            newt = mc + 0.5 * (mq - mc)
        }
    } else if 0 != dsign {
        // Case 2: a lower function value and derivatives of
        // opposite sign. The minimum is brackt. If the cubic
        // minimizer is closer to x than the quadratic (secant) one,
        // the cubic one is taken, else the quadratic one is taken.
        *brackt = 1;
        bound = 0;
        cubic_minimizer(&mut mc, *x, *fx, *dx, *t, *ft, *dt);
        quard_minimizer2(&mut mq, *x, *dx, *t, *dt);
        if (mc - *t).abs() > (mq - *t).abs() {
            newt = mc
        } else {
            newt = mq
        }
    } else if (*dt).abs() < (*dx).abs() {
        // Case 3: a lower function value, derivatives of the
        // same sign, and the magnitude of the derivative decreases.
        // The cubic minimizer is only used if the cubic tends to
        // infinity in the direction of the minimizer or if the minimum
        // of the cubic is beyond t. Otherwise the cubic minimizer is
        // defined to be either tmin or tmax. The quadratic (secant)
        // minimizer is also computed and if the minimum is brackt
        // then the the minimizer closest to x is taken, else the one
        // farthest away is taken.
        bound = 1;
        cubic_minimizer2(&mut mc, *x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
        quard_minimizer2(&mut mq, *x, *dx, *t, *dt);
        // a = *x - *t;
        // mq = *t + *dt / (*dt - *dx) * a;
        if 0 != *brackt {
            if (*t - mc).abs() < (*t - mq).abs() {
                newt = mc
            } else {
                newt = mq
            }
        } else if (*t - mc).abs() > (*t - mq).abs() {
            newt = mc
        } else {
            newt = mq
        }
    } else {
        // Case 4: a lower function value, derivatives of the
        // same sign, and the magnitude of the derivative does
        // not decrease. If the minimum is not brackt, the step
        // is either tmin or tmax, else the cubic minimizer is taken.
        bound = 0i32;
        if 0 != *brackt {
            cubic_minimizer(&mut newt, *t, *ft, *dt, *y, *fy, *dy);
        } else if *x < *t {
            newt = tmax
        } else {
            newt = tmin
        }
    }

    // Update the interval of uncertainty. This update does not
    // depend on the new step or the case analysis above.

    // - Case a: if f(x) < f(t),
    // x <- x, y <- t.
    // - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
    // x <- t, y <- y.
    // - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
    // x <- t, y <- x.
    if *fx < *ft {
        /* Case a */
        *y = *t;
        *fy = *ft;
        *dy = *dt
    } else {
        /* Case c */
        if 0 != dsign {
            *y = *x;
            *fy = *fx;
            *dy = *dx
        }
        /* Cases b and c */
        *x = *t;
        *fx = *ft;
        *dx = *dt
    }

    // Clip the new trial value in [tmin, tmax].
    if tmax < newt {
        newt = tmax
    }
    if newt < tmin {
        newt = tmin
    }

    // Redefine the new trial value if it is close to the upper bound of the
    // interval.
    if 0 != *brackt && 0 != bound {
        mq = *x + 0.66f64 * (*y - *x);
        if *x < *y {
            if mq < newt {
                newt = mq
            }
        } else if newt < mq {
            newt = mq
        }
    }
    // Return the new trial value.
    *t = newt;
    return 0;
}
// unsafe:1 ends here

// BackTracking

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*BackTracking][BackTracking:1]]
unsafe extern "C" fn line_search_backtracking(
    n: libc::c_int,
    mut x: *mut f64,
    mut f: *mut f64,
    mut g: *mut f64,
    mut s: *mut f64,
    mut stp: *mut f64,
    mut xp: *const f64,
    mut gp: *const f64,
    mut wp: *mut f64,
    mut cd: *mut callback_data_t,
    param: &lbfgs_parameter_t,
) -> libc::c_int {
    let mut width: f64 = 0.;
    let mut dg: f64 = 0.;
    let mut finit: f64 = 0.;
    let mut dginit: f64 = 0.0f64;
    let mut dgtest: f64 = 0.;
    let dec: f64 = 0.5f64;
    let inc: f64 = 2.1f64;

    // Check the input parameters for errors.
    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Compute the initial gradient in the search direction.
    vecdot(&mut dginit, g, s, n);

    // Make sure that s points to a descent direction.
    if 0.0 < dginit {
        return LBFGSERR_INCREASEGRADIENT as libc::c_int;
    }

    // The initial value of the objective function.
    finit = *f;
    dgtest = param.ftol * dginit;

    let mut count: libc::c_int = 0i32;
    loop {
        veccpy(x, xp, n);
        vecadd(x, s, *stp, n);
        // Evaluate the function and gradient values.
        *f = (*cd).proc_evaluate.expect("non-null function pointer")(
            (*cd).instance,
            x,
            g,
            (*cd).n,
            *stp,
        );
        count += 1;
        if *f > finit + *stp * dgtest {
            width = dec
        } else if param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO as libc::c_int {
            // Exit with the Armijo condition.
            return count;
        } else {
            // Check the Wolfe condition.
            vecdot(&mut dg, g, s, n);
            if dg < param.wolfe * dginit {
                width = inc
            } else if param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE as libc::c_int {
                // Exit with the regular Wolfe condition.
                return count;
            } else if dg > - param.wolfe * dginit {
                width = dec
            } else {
                return count;
            }
        }
        if *stp < param.min_step {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP as libc::c_int;
        } else if *stp > param.max_step {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP as libc::c_int;
        } else if param.max_linesearch <= count {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH as libc::c_int;
        } else {
            *stp *= width
        }
    }
}

unsafe extern "C" fn line_search_backtracking_owlqn(
    n: libc::c_int,
    mut x: *mut f64,
    mut f: *mut f64,
    mut g: *mut f64,
    mut s: *mut f64,
    mut stp: *mut f64,
    mut xp: *const f64,
    mut gp: *const f64,
    mut wp: *mut f64,
    mut cd: *mut callback_data_t,
    param: &lbfgs_parameter_t,
) -> libc::c_int {
    let mut i: libc::c_int = 0;
    let mut count: libc::c_int = 0i32;
    let mut width: f64 = 0.5f64;
    let mut norm: f64 = 0.0f64;
    let mut finit: f64 = *f;
    let mut dgtest: f64 = 0.;

    // Check the input parameters for errors.
    if *stp <= 0.0f64 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Choose the orthant for the new point.
    i = 0i32;
    while i < n {
        *wp.offset(i as isize) = if *xp.offset(i as isize) == 0.0f64 {
            -*gp.offset(i as isize)
        } else {
            *xp.offset(i as isize)
        };
        i += 1
    }

    loop {
        // Update the current point.
        veccpy(x, xp, n);
        vecadd(x, s, *stp, n);
        /* The current point is projected onto the orthant. */
        owlqn_project(x, wp, param.orthantwise_start, param.orthantwise_end);
        /* Evaluate the function and gradient values. */
        *f = (*cd).proc_evaluate.expect("non-null function pointer")(
            (*cd).instance,
            x,
            g,
            (*cd).n,
            *stp,
        );
        /* Compute the L1 norm of the variables and add it to the object value. */
        norm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
        *f += norm * param.orthantwise_c;
        count += 1;
        dgtest = 0.0f64;
        i = 0i32;
        while i < n {
            dgtest += (*x.offset(i as isize) - *xp.offset(i as isize)) * *gp.offset(i as isize);
            i += 1
        }
        if *f <= finit + param.ftol * dgtest {
            /* The sufficient decrease condition. */
            return count;
        } else if *stp < param.min_step {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP as libc::c_int;
        } else if *stp > param.max_step {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP as libc::c_int;
        } else if param.max_linesearch <= count {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH as libc::c_int;
        } else {
            *stp *= width
        }
    }
}
// BackTracking:1 ends here

// vector operations

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*vector%20operations][vector operations:1]]
/// Abstracting lbfgs required math operations
pub trait LbfgsMath<T> {
    /// y += c*x
    fn vecadd(&mut self, x: &[T], c: T);

    /// vector dot product
    /// s = x.dot(y)
    fn vecdot(&self, other: &[T]) -> f64;

    /// y = z
    fn veccpy(&mut self, x: &[T]);

    /// y = -x
    fn vecncpy(&mut self, x: &[T]);

    /// z = x - y
    fn vecdiff(&mut self, x: &[T], y: &[T]);

    /// y *= c
    fn vecscale(&mut self, c: T);

    /// ||x||
    fn vec2norm(&self) -> T;

    /// 1 / ||x||
    fn vec2norminv(&self) -> T;

    /// norm = sum(..., |x|, ...)
    fn owlqn_x1norm(&self, start: usize) -> T;
}

impl LbfgsMath<f64> for [f64] {
    /// y += c*x
    fn vecadd(&mut self, x: &[f64], c: f64) {
        for (y, x) in self.iter_mut().zip(x) {
            *y += c * x;
        }
    }

    /// s = y.dot(x)
    fn vecdot(&self, other: &[f64]) -> f64 {
        self.iter().zip(other).map(|(x, y)| x * y).sum()
    }

    /// y *= c
    fn vecscale(&mut self, c: f64) {
        for y in self.iter_mut() {
            *y *= c;
        }
    }

    /// y = x
    fn veccpy(&mut self, x: &[f64]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = *x;
        }
    }

    /// y = -x
    fn vecncpy(&mut self, x: &[f64]) {
        for (v, x) in self.iter_mut().zip(x) {
            *v = -x;
        }
    }

    /// z = x - y
    fn vecdiff(&mut self, x: &[f64], y: &[f64]) {
        for ((z, x), y) in self.iter_mut().zip(x).zip(y) {
            *z = x - y;
        }
    }

    /// ||x||
    fn vec2norm(&self) -> f64 {
        let n2 = self.vecdot(&self);
        n2.sqrt()
    }

    /// 1/||x||
    fn vec2norminv(&self) -> f64 {
        1.0 / self.vec2norm()
    }

    fn owlqn_x1norm(&self, start: usize) -> f64 {
        self.iter().skip(start).map(|v| v.abs()).sum()
    }
}

#[test]
fn test_lbfgs_math() {
    // vector scaled add
    let x = [1.0, 1.0, 1.0];
    let c = 2.;

    let mut y = [1.0, 2.0, 3.0];
    y.vecadd(&x, c);

    assert_eq!(3.0, y[0]);
    assert_eq!(4.0, y[1]);
    assert_eq!(5.0, y[2]);

    // vector dot
    let v = y.vecdot(&x);
    assert_eq!(12.0, v);

    // vector scale
    y.vecscale(2.0);
    assert_eq!(6.0, y[0]);
    assert_eq!(8.0, y[1]);
    assert_eq!(10.0, y[2]);

    // vector diff
    let mut z = y.clone();
    z.vecdiff(&x, &y);
    assert_eq!(-5.0, z[0]);
    assert_eq!(-7.0, z[1]);
    assert_eq!(-9.0, z[2]);

    // vector copy
    y.veccpy(&x);

    // y = -x
    y.vecncpy(&x);
    assert_eq!(-1.0, y[0]);
    assert_eq!(-1.0, y[1]);
    assert_eq!(-1.0, y[2]);

    // let x = z.as_ptr();
    // unsafe {
    //     let v = owlqn_x1norm(x, 0, 3);
    //     println!("{:#?}", v);
    // }
    let v = z.owlqn_x1norm(1);
    assert_eq!(v, 16.0);
}

// x(k+1) = x(k) + alpha_k * d_k
// y += c*x
unsafe extern "C" fn vecadd(
    mut y: *mut lbfgsfloatval_t,
    x: *const lbfgsfloatval_t,
    c: lbfgsfloatval_t,
    n: libc::c_int,
) {
    // convert pointer to native data type
    let n = n as usize;

    let arr_x = unsafe { ::std::slice::from_raw_parts(x, n) };
    let mut arr_y = unsafe { ::std::slice::from_raw_parts_mut(y, n) };

    arr_y.vecadd(&arr_x, c);
}

unsafe extern "C" fn vecdot(
    mut s: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    mut y: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    // convert pointer to native data type
    let n = n as usize;
    let arr_x = unsafe { ::std::slice::from_raw_parts(x, n) };
    let arr_y = unsafe { ::std::slice::from_raw_parts(y, n) };

    *s = arr_x.vecdot(arr_y);
}

/// y *= c
unsafe extern "C" fn vecscale(mut y: *mut lbfgsfloatval_t, c: lbfgsfloatval_t, n: libc::c_int) {
    // convert pointer to native data type
    let n = n as usize;
    let mut arr_y = unsafe { ::std::slice::from_raw_parts_mut(y, n) };
    arr_y.vecscale(c);
}

/// y = -x
unsafe extern "C" fn vecncpy(
    mut y: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    let n = n as usize;
    let arr_x = unsafe { ::std::slice::from_raw_parts(x, n) };
    let mut arr_y = unsafe { ::std::slice::from_raw_parts_mut(y, n) };

    arr_y.vecncpy(&arr_x);
}

/// z = x - y
unsafe extern "C" fn vecdiff(
    mut z: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    mut y: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    let n = n as usize;

    let arr_x = unsafe { ::std::slice::from_raw_parts(x, n) };
    let arr_y = unsafe { ::std::slice::from_raw_parts(y, n) };
    let mut arr_z = unsafe { ::std::slice::from_raw_parts_mut(z, n) };

    arr_z.vecdiff(&arr_x, &arr_y);
}

/// s = ||x||
unsafe extern "C" fn vec2norm(
    mut s: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    vecdot(s, x, x, n);
    *s = (*s).sqrt();
}

/// y = x
unsafe extern "C" fn veccpy(
    mut y: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    let n = n as usize;

    let arr_x = unsafe { ::std::slice::from_raw_parts(x, n) };
    let mut arr_y = unsafe { ::std::slice::from_raw_parts_mut(y, n) };

    arr_y.veccpy(&arr_x);
}

unsafe extern "C" fn vec2norminv(
    mut s: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    vec2norm(s, x, n);
    *s = 1.0 / *s;
}

/// norm = sum(|x|, ...)
unsafe extern "C" fn owlqn_x1norm(
    mut x: *const lbfgsfloatval_t,
    start: libc::c_int,
    n: libc::c_int,
) -> lbfgsfloatval_t {
    // let mut i: libc::c_int = 0;
    // let mut norm: lbfgsfloatval_t = 0.0f64;
    // i = start;
    // while i < n {
    //     norm += (*x.offset(i as isize)).abs();
    //     i += 1
    // }
    // return norm;

    let arr_x = unsafe { ::std::slice::from_raw_parts(x, n as usize) };
    arr_x.owlqn_x1norm(start as usize)
}

// TODO: clean up
unsafe extern "C" fn vecalloc(mut size: size_t) -> *mut libc::c_void {
    let mut memblock: *mut libc::c_void = malloc(size);
    if !memblock.is_null() {
        memset(memblock, 0i32, size);
    }
    return memblock;
}

unsafe extern "C" fn vecfree(mut memblock: *mut libc::c_void) {
    free(memblock);
}
// vector operations:1 ends here

// src

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*src][src:1]]
#[no_mangle]
pub unsafe extern "C" fn lbfgs(
    arr_x: &mut [f64],
    ptr_fx: &mut lbfgsfloatval_t,
    mut proc_evaluate: lbfgs_evaluate_t,
    mut proc_progress: lbfgs_progress_t,
    mut instance: *mut libc::c_void,
    param: &lbfgs_parameter_t,
) -> libc::c_int {
    let n = arr_x.len() as i32;
    let mut x = arr_x.as_mut_ptr();

    // FIXME: make param immutable
    let mut param = param.clone();

    let mut current_block: u64;
    let mut ret: libc::c_int = 0;
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    let mut k: libc::c_int = 0;
    let mut ls: libc::c_int = 0;
    let mut end: libc::c_int = 0;
    let mut bound: libc::c_int = 0;
    let mut step: lbfgsfloatval_t = 0.;

    let m: libc::c_int = param.m;
    let mut xp: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut g: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut gp: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut pg: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut d: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut w: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut pf: *mut lbfgsfloatval_t = 0 as *mut lbfgsfloatval_t;
    let mut lm: *mut iteration_data_t = 0 as *mut iteration_data_t;
    let mut it: *mut iteration_data_t = 0 as *mut iteration_data_t;
    let mut ys = 0.;
    let mut yy = 0.;
    let mut xnorm = 0.;
    let mut gnorm = 0.;
    let mut beta = 0.;
    let mut fx = 0.0;
    let mut rate = 0.0;
    let mut linesearch: line_search_proc = Some(line_search_morethuente);

    // Construct a callback data.
    let mut cd: callback_data_t = tag_callback_data {
        n: 0,
        instance: 0 as *mut libc::c_void,
        proc_evaluate: None,
        proc_progress: None,
    };

    cd.n = n;
    cd.instance = instance;
    cd.proc_evaluate = proc_evaluate;
    cd.proc_progress = proc_progress;

    // Check the input parameters for errors.
    if n <= 0i32 {
        return LBFGSERR_INVALID_N as libc::c_int;
    } else if param.epsilon < 0.0f64 {
        return LBFGSERR_INVALID_EPSILON as libc::c_int;
    } else if param.past < 0i32 {
        return LBFGSERR_INVALID_TESTPERIOD as libc::c_int;
    } else if param.delta < 0.0f64 {
        return LBFGSERR_INVALID_DELTA as libc::c_int;
    } else if param.min_step < 0.0f64 {
        return LBFGSERR_INVALID_MINSTEP as libc::c_int;
    } else if param.max_step < param.min_step {
        return LBFGSERR_INVALID_MAXSTEP as libc::c_int;
    } else if param.ftol < 0.0f64 {
        return LBFGSERR_INVALID_FTOL as libc::c_int;
    }

    if param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE as libc::c_int
        || param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE as libc::c_int
    {
        if param.wolfe <= param.ftol || 1.0f64 <= param.wolfe {
            return LBFGSERR_INVALID_WOLFE as libc::c_int;
        }
    }

    if param.gtol < 0.0 {
        return LBFGSERR_INVALID_GTOL as libc::c_int;
    } else if param.xtol < 0.0 {
        return LBFGSERR_INVALID_XTOL as libc::c_int;
    } else if param.max_linesearch <= 0 {
        return LBFGSERR_INVALID_MAXLINESEARCH as libc::c_int;
    } else if param.orthantwise_c < 0.0 {
        return LBFGSERR_INVALID_ORTHANTWISE as libc::c_int;
    } else if param.orthantwise_start < 0 || n < param.orthantwise_start {
        return LBFGSERR_INVALID_ORTHANTWISE_START as libc::c_int;
    }

    if param.orthantwise_end < 0 {
        param.orthantwise_end = n
    }
    if n < param.orthantwise_end {
        return LBFGSERR_INVALID_ORTHANTWISE_END as libc::c_int;
    }
    if param.orthantwise_c != 0.0f64 {
        match param.linesearch {
            2 => linesearch = Some(line_search_backtracking_owlqn),
            _ => {
                // Only the backtracking method is available.
                return LBFGSERR_INVALID_LINESEARCH as libc::c_int;
            }
        }
    } else {
        match param.linesearch {
            0 => linesearch = Some(line_search_morethuente),
            1 | 2 | 3 => linesearch = Some(line_search_backtracking),
            _ => return LBFGSERR_INVALID_LINESEARCH as libc::c_int,
        }
    }

    // Allocate working space.
    xp = vecalloc(
        (n as libc::c_ulong)
            .wrapping_mul(::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong),
    ) as *mut lbfgsfloatval_t;
    g = vecalloc(
        (n as libc::c_ulong)
            .wrapping_mul(::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong),
    ) as *mut lbfgsfloatval_t;
    gp = vecalloc(
        (n as libc::c_ulong)
            .wrapping_mul(::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong),
    ) as *mut lbfgsfloatval_t;
    d = vecalloc(
        (n as libc::c_ulong)
            .wrapping_mul(::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong),
    ) as *mut lbfgsfloatval_t;

    w = vecalloc(
        (n as libc::c_ulong)
            .wrapping_mul(::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong),
    ) as *mut lbfgsfloatval_t;

    if xp.is_null() || g.is_null() || gp.is_null() || d.is_null() || w.is_null() {
        ret = LBFGSERR_OUTOFMEMORY as libc::c_int
    } else {
        if param.orthantwise_c != 0.0 {
            // Allocate working space for OW-LQN.
            pg = vecalloc(
                (n as libc::c_ulong)
                    .wrapping_mul(::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong),
            ) as *mut lbfgsfloatval_t;
            if pg.is_null() {
                ret = LBFGSERR_OUTOFMEMORY as libc::c_int;
                current_block = 13422061289108735151;
            } else {
                current_block = 14763689060501151050;
            }
        } else {
            current_block = 14763689060501151050;
        }

        info!("start lbfgs loop...");
        match current_block {
            13422061289108735151 => {}
            _ => {
                // Allocate limited memory storage.
                lm = vecalloc(
                    (m as libc::c_ulong)
                        .wrapping_mul(::std::mem::size_of::<iteration_data_t>() as libc::c_ulong),
                ) as *mut iteration_data_t;

                if lm.is_null() {
                    ret = LBFGSERR_OUTOFMEMORY as libc::c_int
                } else {
                    // Initialize the limited memory.
                    i = 0i32;
                    loop {
                        if !(i < m) {
                            current_block = 2891135413264362348;
                            break;
                        }
                        it = &mut *lm.offset(i as isize) as *mut iteration_data_t;
                        (*it).alpha = 0i32 as lbfgsfloatval_t;
                        (*it).ys = 0i32 as lbfgsfloatval_t;
                        (*it).s = vecalloc(
                            (n as libc::c_ulong).wrapping_mul(
                                ::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong,
                            ),
                        ) as *mut lbfgsfloatval_t;
                        (*it).y = vecalloc(
                            (n as libc::c_ulong).wrapping_mul(
                                ::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong,
                            ),
                        ) as *mut lbfgsfloatval_t;
                        if (*it).s.is_null() || (*it).y.is_null() {
                            ret = LBFGSERR_OUTOFMEMORY as libc::c_int;
                            current_block = 13422061289108735151;
                            break;
                        } else {
                            i += 1
                        }
                    }
                    match current_block {
                        13422061289108735151 => {}
                        _ => {
                            // Allocate an array for storing previous values of the objective function.
                            if 0 < param.past {
                                pf = vecalloc((param.past as libc::c_ulong).wrapping_mul(
                                    ::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong,
                                )) as *mut lbfgsfloatval_t
                            }
                            // Evaluate the function value and its gradient.
                            fx = cd.proc_evaluate.expect("non-null function pointer")(
                                cd.instance,
                                x,
                                g,
                                cd.n,
                                0i32 as lbfgsfloatval_t,
                            );
                            if 0.0 != param.orthantwise_c {
                                // Compute the L1 norm of the variable and add it to the object value.
                                xnorm =
                                    owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
                                fx += xnorm * param.orthantwise_c;
                                owlqn_pseudo_gradient(
                                    pg,
                                    x,
                                    g,
                                    n,
                                    param.orthantwise_c,
                                    param.orthantwise_start,
                                    param.orthantwise_end,
                                );
                            }
                            // Store the initial value of the objective function.
                            if !pf.is_null() {
                                *pf.offset(0isize) = fx
                            }

                            // Compute the direction;
                            // we assume the initial hessian matrix H_0 as the identity matrix.
                            if param.orthantwise_c == 0.0f64 {
                                vecncpy(d, g, n);
                            } else {
                                vecncpy(d, pg, n);
                            }

                            // Make sure that the initial variables are not a minimizer.
                            vec2norm(&mut xnorm, x, n);
                            if param.orthantwise_c == 0.0f64 {
                                vec2norm(&mut gnorm, g, n);
                            } else {
                                vec2norm(&mut gnorm, pg, n);
                            }
                            if xnorm < 1.0f64 {
                                xnorm = 1.0f64
                            }
                            if gnorm / xnorm <= param.epsilon {
                                ret = LBFGS_ALREADY_MINIMIZED as libc::c_int
                            }

                            // Compute the initial step:
                            // step = 1.0 / sqrt(vecdot(d, d, n))
                            vec2norminv(&mut step, d, n);
                            k = 1i32;
                            end = 0i32;

                            loop {
                                // Store the current position and gradient vectors.
                                veccpy(xp, x, n);
                                veccpy(gp, g, n);
                                // Search for an optimal step.
                                if param.orthantwise_c == 0.0f64 {
                                    ls = linesearch.expect("non-null function pointer")(
                                        n, x, &mut fx, g, d, &mut step, xp, gp, w, &mut cd, &param,
                                    )
                                } else {
                                    ls = linesearch.expect("non-null function pointer")(
                                        n, x, &mut fx, g, d, &mut step, xp, pg, w, &mut cd, &param,
                                    );
                                    owlqn_pseudo_gradient(
                                        pg,
                                        x,
                                        g,
                                        n,
                                        param.orthantwise_c,
                                        param.orthantwise_start,
                                        param.orthantwise_end,
                                    );
                                }
                                if ls < 0 {
                                    error!("line search failed, revert to the previous point!");
                                    /* Revert to the previous point. */
                                    veccpy(x, xp, n);
                                    veccpy(g, gp, n);
                                    ret = ls;
                                    break;
                                }
                                /* Compute x and g norms. */
                                vec2norm(&mut xnorm, x, n);
                                if param.orthantwise_c == 0.0f64 {
                                    vec2norm(&mut gnorm, g, n);
                                } else {
                                    vec2norm(&mut gnorm, pg, n);
                                }

                                // Report the progress.
                                if cd.proc_progress.is_some() {
                                    ret = cd.proc_progress.expect("non-null function pointer")(
                                        cd.instance,
                                        x,
                                        g,
                                        fx,
                                        xnorm,
                                        gnorm,
                                        step,
                                        cd.n,
                                        k,
                                        ls,
                                    );
                                    if 0 != ret {
                                        break;
                                    }
                                }

                                // Convergence test.
                                // The criterion is given by the following formula:
                                //     |g(x)| / \max(1, |x|) < \epsilon

                                xnorm = xnorm.max(1.0);
                                if gnorm / xnorm <= param.epsilon {
                                    info!("lbfgs converged");
                                    // Convergence.
                                    ret = LBFGS_SUCCESS as libc::c_int;
                                    break;
                                }

                                // Test for stopping criterion.
                                // The criterion is given by the following formula:
                                //    (f(past_x) - f(x)) / f(x) < \delta
                                if !pf.is_null() {
                                    // We don't test the stopping criterion while k < past.
                                    if param.past <= k {
                                        // Compute the relative improvement from the past.
                                        rate = (*pf.offset((k % param.past) as isize) - fx) / fx;
                                        // The stopping criterion.
                                        if rate < param.delta {
                                            ret = LBFGS_STOP as libc::c_int;
                                            break;
                                        }
                                    }
                                    // Store the current value of the objective function.
                                    *pf.offset((k % param.past) as isize) = fx
                                }

                                if param.max_iterations != 0 && param.max_iterations < k + 1 {
                                    // Maximum number of iterations.
                                    warn!("max_iterations reached!");
                                    ret = LBFGSERR_MAXIMUMITERATION as libc::c_int;
                                    break;
                                }

                                // Update vectors s and y:
                                // s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
                                // y_{k+1} = g_{k+1} - g_{k}.
                                it = &mut *lm.offset(end as isize) as *mut iteration_data_t;
                                vecdiff((*it).s, x, xp, n);
                                vecdiff((*it).y, g, gp, n);

                                // Compute scalars ys and yy:
                                // ys = y^t \cdot s = 1 / \rho.
                                // yy = y^t \cdot y.
                                // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
                                vecdot(&mut ys, (*it).y, (*it).s, n);
                                vecdot(&mut yy, (*it).y, (*it).y, n);
                                (*it).ys = ys;

                                // Recursive formula to compute dir = -(H \cdot g).
                                // This is described in page 779 of:
                                // Jorge Nocedal.
                                // Updating Quasi-Newton Matrices with Limited Storage.
                                // Mathematics of Computation, Vol. 35, No. 151,
                                // pp. 773--782, 1980.
                                bound = if m <= k { m } else { k };

                                k += 1;
                                end = (end + 1i32) % m;
                                // Compute the steepest direction.
                                if param.orthantwise_c == 0.0f64 {
                                    // Compute the negative of gradients.
                                    vecncpy(d, g, n);
                                } else {
                                    vecncpy(d, pg, n);
                                }

                                j = end;
                                i = 0i32;
                                while i < bound {
                                    // if (--j == -1) j = m-1;
                                    j = (j + m - 1i32) % m;
                                    it = &mut *lm.offset(j as isize) as *mut iteration_data_t;
                                    // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}.
                                    vecdot(&mut (*it).alpha, (*it).s, d, n);
                                    (*it).alpha /= (*it).ys;
                                    // q_{i} = q_{i+1} - \alpha_{i} y_{i}.
                                    vecadd(d, (*it).y, -(*it).alpha, n);
                                    i += 1
                                }
                                vecscale(d, ys / yy, n);
                                i = 0i32;
                                while i < bound {
                                    it = &mut *lm.offset(j as isize) as *mut iteration_data_t;
                                    // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}.
                                    vecdot(&mut beta, (*it).y, d, n);
                                    beta /= (*it).ys;
                                    // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}.
                                    vecadd(d, (*it).s, (*it).alpha - beta, n);
                                    // if (++j == m) j = 0;
                                    j = (j + 1i32) % m;
                                    i += 1
                                }

                                // Constrain the search direction for orthant-wise updates.
                                if param.orthantwise_c != 0.0 {
                                    i = param.orthantwise_start;
                                    while i < param.orthantwise_end {
                                        if *d.offset(i as isize) * *pg.offset(i as isize)
                                            >= 0i32 as libc::c_double
                                        {
                                            *d.offset(i as isize) = 0i32 as lbfgsfloatval_t
                                        }
                                        i += 1
                                    }
                                }

                                // Now the search direction d is ready. We try step = 1 first.
                                step = 1.0
                            }
                        }
                    }
                }
            }
        }
    }

    // Return the final value of the objective function.
    *ptr_fx = fx;

    vecfree(pf as *mut libc::c_void);
    /* Free memory blocks used by this function. */
    if !lm.is_null() {
        i = 0i32;
        while i < m {
            vecfree((*lm.offset(i as isize)).s as *mut libc::c_void);
            vecfree((*lm.offset(i as isize)).y as *mut libc::c_void);
            i += 1
        }
        vecfree(lm as *mut libc::c_void);
    }
    vecfree(pg as *mut libc::c_void);
    vecfree(w as *mut libc::c_void);
    vecfree(d as *mut libc::c_void);
    vecfree(gp as *mut libc::c_void);
    vecfree(g as *mut libc::c_void);
    vecfree(xp as *mut libc::c_void);
    return ret;
}

unsafe extern "C" fn owlqn_pseudo_gradient(
    mut pg: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    mut g: *const lbfgsfloatval_t,
    n: libc::c_int,
    c: lbfgsfloatval_t,
    start: libc::c_int,
    end: libc::c_int,
) {
    let mut i: libc::c_int = 0;
    /* Compute the negative of gradients. */
    i = 0i32;
    while i < start {
        *pg.offset(i as isize) = *g.offset(i as isize);
        i += 1
    }

    // Compute the psuedo-gradients.
    i = start;
    while i < end {
        if *x.offset(i as isize) < 0.0f64 {
            /* Differentiable. */
            *pg.offset(i as isize) = *g.offset(i as isize) - c
        } else if 0.0f64 < *x.offset(i as isize) {
            /* Differentiable. */
            *pg.offset(i as isize) = *g.offset(i as isize) + c
        } else if *g.offset(i as isize) < -c {
            /* Take the right partial derivative. */
            *pg.offset(i as isize) = *g.offset(i as isize) + c
        } else if c < *g.offset(i as isize) {
            /* Take the left partial derivative. */
            *pg.offset(i as isize) = *g.offset(i as isize) - c
        } else {
            *pg.offset(i as isize) = 0.0f64
        }
        i += 1
    }
    i = end;
    while i < n {
        *pg.offset(i as isize) = *g.offset(i as isize);
        i += 1
    }
}

/* *
 * Allocate an array for variables.
 *
 *  This function allocates an array of variables for the convenience of
 *  ::lbfgs function; the function has a requreiemt for a variable array
 *  when libLBFGS is built with SSE/SSE2 optimization routines. A user does
 *  not have to use this function for libLBFGS built without SSE/SSE2
 *  optimization.
 *
 *  @param  n           The number of variables.
 */
#[no_mangle]
pub unsafe extern "C" fn lbfgs_malloc(mut n: libc::c_int) -> *mut lbfgsfloatval_t {
    /*defined(USE_SSE)*/
    return vecalloc(
        (::std::mem::size_of::<lbfgsfloatval_t>() as libc::c_ulong)
            .wrapping_mul(n as libc::c_ulong),
    ) as *mut lbfgsfloatval_t;
}

//
//  Free an array of variables.
//
//  @param  x           The array of variables allocated by ::lbfgs_malloc
//                      function.
#[no_mangle]
pub unsafe extern "C" fn lbfgs_free(mut x: *mut lbfgsfloatval_t) {
    vecfree(x as *mut libc::c_void);
}

unsafe extern "C" fn owlqn_project(
    mut d: *mut lbfgsfloatval_t,
    mut sign: *const lbfgsfloatval_t,
    start: libc::c_int,
    end: libc::c_int,
) {
    let mut i: libc::c_int = 0;
    i = start;
    while i < end {
        if *d.offset(i as isize) * *sign.offset(i as isize) <= 0i32 as libc::c_double {
            *d.offset(i as isize) = 0i32 as lbfgsfloatval_t
        }
        i += 1
    }
}
// src:1 ends here
