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

    /// A coefficient for the Wolfe condition.
    ///
    ///  This parameter is valid only when the backtracking line-search
    ///  algorithm is used with the Wolfe condition,
    ///  ::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
    ///  ::LBFGS_LINESEARCH_BACKTRACKING_WOLFE .
    ///  The default value is \c 0.9. This parameter should be greater
    ///  the \ref ftol parameter and smaller than \c 1.0.
    pub wolfe: lbfgsfloatval_t,

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

// BackTracking

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*BackTracking][BackTracking:1]]
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
        // ??
        wp: *mut lbfgsfloatval_t,
        // callback struct
        cd: *mut callback_data_t,
        // LBFGS parameter
        param: &lbfgs_parameter_t,
    ) -> libc::c_int,
>;

unsafe extern "C" fn line_search_backtracking(
    n: libc::c_int,
    mut x: *mut lbfgsfloatval_t,
    mut f: *mut lbfgsfloatval_t,
    mut g: *mut lbfgsfloatval_t,
    mut s: *mut lbfgsfloatval_t,
    mut stp: *mut lbfgsfloatval_t,
    mut xp: *const lbfgsfloatval_t,
    mut gp: *const lbfgsfloatval_t,
    mut wp: *mut lbfgsfloatval_t,
    mut cd: *mut callback_data_t,
    param: &lbfgs_parameter_t,
) -> libc::c_int {
    let mut width: lbfgsfloatval_t = 0.;
    let mut dg: lbfgsfloatval_t = 0.;
    let mut finit: lbfgsfloatval_t = 0.;
    let mut dginit: lbfgsfloatval_t = 0.0f64;
    let mut dgtest: lbfgsfloatval_t = 0.;
    let dec: lbfgsfloatval_t = 0.5f64;
    let inc: lbfgsfloatval_t = 2.1f64;

    // Check the input parameters for errors.
    if *stp <= 0.0f64 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Compute the initial gradient in the search direction.
    vecdot(&mut dginit, g, s, n);

    // Make sure that s points to a descent direction.
    if (0i32 as libc::c_double) < dginit {
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
            /* Check the Wolfe condition. */
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
    mut x: *mut lbfgsfloatval_t,
    mut f: *mut lbfgsfloatval_t,
    mut g: *mut lbfgsfloatval_t,
    mut s: *mut lbfgsfloatval_t,
    mut stp: *mut lbfgsfloatval_t,
    mut xp: *const lbfgsfloatval_t,
    mut gp: *const lbfgsfloatval_t,
    mut wp: *mut lbfgsfloatval_t,
    mut cd: *mut callback_data_t,
    param: &lbfgs_parameter_t,
) -> libc::c_int {
    let mut i: libc::c_int = 0;
    let mut count: libc::c_int = 0i32;
    let mut width: lbfgsfloatval_t = 0.5f64;
    let mut norm: lbfgsfloatval_t = 0.0f64;
    let mut finit: lbfgsfloatval_t = *f;
    let mut dgtest: lbfgsfloatval_t = 0.;

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

// src

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*src][src:1]]
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
    let mut count: libc::c_int = 0i32;
    let mut brackt: libc::c_int = 0;
    let mut stage1: libc::c_int = 0;
    let mut uinfo: libc::c_int = 0i32;
    let mut dg: lbfgsfloatval_t = 0.;
    let mut stx: lbfgsfloatval_t = 0.;
    let mut fx: lbfgsfloatval_t = 0.;
    let mut dgx: lbfgsfloatval_t = 0.;
    let mut sty: lbfgsfloatval_t = 0.;
    let mut fy: lbfgsfloatval_t = 0.;
    let mut dgy: lbfgsfloatval_t = 0.;
    let mut fxm: lbfgsfloatval_t = 0.;
    let mut dgxm: lbfgsfloatval_t = 0.;
    let mut fym: lbfgsfloatval_t = 0.;
    let mut dgym: lbfgsfloatval_t = 0.;
    let mut fm: lbfgsfloatval_t = 0.;
    let mut dgm: lbfgsfloatval_t = 0.;
    let mut finit: lbfgsfloatval_t = 0.;
    let mut ftest1: lbfgsfloatval_t = 0.;
    let mut dginit: lbfgsfloatval_t = 0.;
    let mut dgtest: lbfgsfloatval_t = 0.;
    let mut width: lbfgsfloatval_t = 0.;
    let mut prev_width: lbfgsfloatval_t = 0.;

    // Check the input parameters for errors.
    if *stp <= 0.0f64 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Compute the initial gradient in the search direction.
    vecdot(&mut dginit, g, s, n);

    // Make sure that s points to a descent direction.
    if (0i32 as libc::c_double) < dginit {
        return LBFGSERR_INCREASEGRADIENT as libc::c_int;
    }

    // Initialize local variables.
    brackt = 0i32;
    stage1 = 1i32;
    finit = *f;
    dgtest = param.ftol * dginit;
    width = param.max_step - param.min_step;
    prev_width = 2.0f64 * width;

    // The variables stx, fx, dgx contain the values of the step,
    // function, and directional derivative at the best step.
    // The variables sty, fy, dgy contain the value of the step,
    // function, and derivative at the other endpoint of
    // the interval of uncertainty.
    // The variables stp, f, dg contain the values of the step,
    // function, and derivative at the current step.
    sty = 0.0f64;
    stx = sty;
    fy = finit;
    fx = fy;
    dgy = dginit;
    dgx = dgy;

    let mut stmin = 0.;
    let mut stmax = 0.;
    loop {
        // Set the minimum and maximum steps to correspond to the
        // present interval of uncertainty.
        if 0 != brackt {
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
        if 0 != brackt
            && (*stp <= stmin
                || stmax <= *stp
                || param.max_linesearch <= count + 1i32
                || uinfo != 0i32)
            || 0 != brackt && stmax - stmin <= param.xtol * stmax
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

        vecdot(&mut dg, g, s, n);
        ftest1 = finit + *stp * dgtest;
        count += 1;

        // Test for errors and convergence.
        if 0 != brackt && (*stp <= stmin || stmax <= *stp || uinfo != 0i32) {
            /* Rounding errors prevent further progress. */
            return LBFGSERR_ROUNDING_ERROR as libc::c_int;
        } else if *stp == param.max_step && *f <= ftest1 && dg <= dgtest {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP as libc::c_int;
        } else if *stp == param.min_step && (ftest1 < *f || dgtest <= dg) {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP as libc::c_int;
        } else if 0 != brackt && stmax - stmin <= param.xtol * stmax {
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
            if 0 != stage1 && *f <= ftest1 && if param.ftol <= param.gtol {
                param.ftol
            } else {
                param.gtol
            } * dginit <= dg
            {
                stage1 = 0i32
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
                uinfo = update_trial_interval(
                    &mut stx,
                    &mut fxm,
                    &mut dgxm,
                    &mut sty,
                    &mut fym,
                    &mut dgym,
                    stp,
                    &mut fm,
                    &mut dgm,
                    stmin,
                    stmax,
                    &mut brackt,
                );

                // Reset the function and gradient values for f.
                fx = fxm + stx * dgtest;
                fy = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest
            } else {
                uinfo = update_trial_interval(
                    &mut stx,
                    &mut fx,
                    &mut dgx,
                    &mut sty,
                    &mut fy,
                    &mut dgy,
                    stp,
                    f,
                    &mut dg,
                    stmin,
                    stmax,
                    &mut brackt,
                )
            }

            // Force a sufficient decrease in the interval of uncertainty.
            if !(0 != brackt) {
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
// src:1 ends here

// src

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*src][src:1]]
unsafe extern "C" fn update_trial_interval(
    mut x: *mut lbfgsfloatval_t,
    mut fx: *mut lbfgsfloatval_t,
    mut dx: *mut lbfgsfloatval_t,
    mut y: *mut lbfgsfloatval_t,
    mut fy: *mut lbfgsfloatval_t,
    mut dy: *mut lbfgsfloatval_t,
    mut t: *mut lbfgsfloatval_t,
    mut ft: *mut lbfgsfloatval_t,
    mut dt: *mut lbfgsfloatval_t,
    tmin: lbfgsfloatval_t,
    tmax: lbfgsfloatval_t,
    mut brackt: *mut libc::c_int,
) -> libc::c_int {
    let mut bound: libc::c_int = 0;
    let mut dsign: libc::c_int = (*dt * (*dx / (*dx).abs()) < 0.0f64) as libc::c_int;
    // minimizer of an interpolated cubic.
    let mut mc = 0.;
    // minimizer of an interpolated quadratic.
    let mut mq = 0.;
    // new trial value.
    let mut newt = 0.;

    // for CUBIC_MINIMIZER and QUARD_MINIMIZER.
    let mut a = 0.;
    let mut d = 0.;
    let mut gamma = 0.;
    let mut theta = 0.;

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
        *brackt = 1i32;
        bound = 1i32;
        d = *t - *x;
        theta = (*fx - *ft) * 3i32 as libc::c_double / d + *dx + *dt;
        p = theta.abs();
        q = (*dx).abs();
        r = (*dt).abs();
        s = if if p >= q { p } else { q } >= r {
            if p >= q {
                p
            } else {
                q
            }
        } else {
            r
        };
        a = theta / s;
        gamma = s * (a * a - *dx / s * (*dt / s)).sqrt();
        if *t < *x {
            gamma = -gamma
        }
        p = gamma - *dx + theta;
        q = gamma - *dx + gamma + *dt;
        r = p / q;
        mc = *x + r * d;
        a = *t - *x;
        mq = *x + *dx / ((*fx - *ft) / a + *dx) / 2i32 as libc::c_double * a;
        if (mc - *x).abs() < (mq - *x).abs() {
            newt = mc
        } else {
            newt = mc + 0.5f64 * (mq - mc)
        }
    } else if 0 != dsign {
        // Case 2: a lower function value and derivatives of
        // opposite sign. The minimum is brackt. If the cubic
        // minimizer is closer to x than the quadratic (secant) one,
        // the cubic one is taken, else the quadratic one is taken.
        *brackt = 1i32;
        bound = 0i32;
        d = *t - *x;
        theta = (*fx - *ft) * 3i32 as libc::c_double / d + *dx + *dt;
        p = theta.abs();
        q = (*dx).abs();
        r = (*dt).abs();
        s = if if p >= q { p } else { q } >= r {
            if p >= q {
                p
            } else {
                q
            }
        } else {
            r
        };
        a = theta / s;
        gamma = s * (a * a - *dx / s * (*dt / s)).sqrt();
        if *t < *x {
            gamma = -gamma
        }
        p = gamma - *dx + theta;
        q = gamma - *dx + gamma + *dt;
        r = p / q;
        mc = *x + r * d;
        a = *x - *t;
        mq = *t + *dt / (*dt - *dx) * a;
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
        bound = 1i32;
        d = *t - *x;
        theta = (*fx - *ft) * 3i32 as libc::c_double / d + *dx + *dt;
        p = theta.abs();
        q = (*dx).abs();
        r = (*dt).abs();
        s = if if p >= q { p } else { q } >= r {
            if p >= q {
                p
            } else {
                q
            }
        } else {
            r
        };

        a = theta / s;
        gamma = s * (if 0i32 as libc::c_double >= a * a - *dx / s * (*dt / s) {
            0i32 as libc::c_double
        } else {
            a * a - *dx / s * (*dt / s)
        }).sqrt();

        if *x < *t {
            gamma = -gamma
        }
        p = gamma - *dt + theta;
        q = gamma - *dt + gamma + *dx;
        r = p / q;
        if r < 0.0f64 && gamma != 0.0f64 {
            mc = *t - r * d
        } else if a < 0i32 as libc::c_double {
            mc = tmax
        } else {
            mc = tmin
        }
        a = *x - *t;
        mq = *t + *dt / (*dt - *dx) * a;
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
            d = *y - *t;

            theta = (*ft - *fy) * 3i32 as libc::c_double / d + *dt + *dy;
            p = theta.abs();
            q = (*dt).abs();
            r = (*dy).abs();

            s = if if p >= q { p } else { q } >= r {
                if p >= q {
                    p
                } else {
                    q
                }
            } else {
                r
            };

            a = theta / s;
            gamma = s * (a * a - *dt / s * (*dy / s)).sqrt();
            if *y < *t {
                gamma = -gamma
            }
            p = gamma - *dt + theta;
            q = gamma - *dt + gamma + *dy;
            r = p / q;
            newt = *t + r * d
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
    return 0i32;
}
// src:1 ends here

// vector operations

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*vector%20operations][vector operations:1]]
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

    vecadd_(&mut arr_y, &arr_x, c);
}

/// x(k+1) = x(k) + alpha_k * d_k
/// y += c*x
/// # Parameters
/// - arr_y: target array
/// - arr_x: initial array
/// - c    : scale factor for arr_x
fn vecadd_(arr_y: &mut [f64], arr_x: &[f64], c: f64) {
    let n = arr_y.len();
    assert_eq!(n, arr_x.len());

    for i in 0..n {
        arr_y[i] += c * arr_x[i];
    }
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

    *s = vecdot_(&arr_x, &arr_y);
}

/// s = x.dot(y)
#[inline]
fn vecdot_(arr_x: &[f64], arr_y: &[f64]) -> f64 {
    arr_x.iter().zip(arr_y).map(|(x, y)| x * y).sum()
}

/// y *= c
unsafe extern "C" fn vecscale(mut y: *mut lbfgsfloatval_t, c: lbfgsfloatval_t, n: libc::c_int) {
    // convert pointer to native data type
    let n = n as usize;
    let mut arr_y = unsafe { ::std::slice::from_raw_parts_mut(y, n) };
    vecscale_(&mut arr_y, c);
}

fn vecscale_(y: &mut [f64], c: f64) {
    for v in y.iter_mut() {
        *v *= c;
    }
}

unsafe extern "C" fn vecncpy(
    mut y: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    let mut i: libc::c_int = 0;
    i = 0i32;
    while i < n {
        *y.offset(i as isize) = -*x.offset(i as isize);
        i += 1
    }
}

/// z = x - y
unsafe extern "C" fn vecdiff(
    mut z: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    mut y: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    let mut i: libc::c_int = 0;
    i = 0i32;
    while i < n {
        *z.offset(i as isize) = *x.offset(i as isize) - *y.offset(i as isize);
        i += 1
    }
}

unsafe extern "C" fn vec2norm(
    mut s: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    vecdot(s, x, x, n);
    *s = (*s).sqrt();
}

unsafe extern "C" fn veccpy(
    mut y: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    let mut i: libc::c_int = 0;
    i = 0i32;
    while i < n {
        *y.offset(i as isize) = *x.offset(i as isize);
        i += 1
    }
}

unsafe extern "C" fn vec2norminv(
    mut s: *mut lbfgsfloatval_t,
    mut x: *const lbfgsfloatval_t,
    n: libc::c_int,
) {
    vec2norm(s, x, n);
    *s = 1.0f64 / *s;
}

unsafe extern "C" fn owlqn_x1norm(
    mut x: *const lbfgsfloatval_t,
    start: libc::c_int,
    n: libc::c_int,
) -> lbfgsfloatval_t {
    let mut i: libc::c_int = 0;
    let mut norm: lbfgsfloatval_t = 0.0f64;
    i = start;
    while i < n {
        norm += (*x.offset(i as isize)).abs();
        i += 1
    }
    return norm;
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
    let mut ys: lbfgsfloatval_t = 0.;
    let mut yy: lbfgsfloatval_t = 0.;
    let mut xnorm: lbfgsfloatval_t = 0.;
    let mut gnorm: lbfgsfloatval_t = 0.;
    let mut beta: lbfgsfloatval_t = 0.;
    let mut fx: lbfgsfloatval_t = 0.0f64;
    let mut rate: lbfgsfloatval_t = 0.0f64;
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
                                xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
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
                                if ls < 0i32 {
                                    error!("line search failed, revert to the previous point!");
                                    /* Revert to the previous point. */
                                    veccpy(x, xp, n);
                                    veccpy(g, gp, n);
                                    ret = ls;
                                    break;
                                } else {
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
                                            rate =
                                                (*pf.offset((k % param.past) as isize) - fx) / fx;
                                            // The stopping criterion.
                                            if rate < param.delta {
                                                ret = LBFGS_STOP as libc::c_int;
                                                break;
                                            }
                                        }
                                        // Store the current value of the objective function.
                                        *pf.offset((k % param.past) as isize) = fx
                                    }

                                    if param.max_iterations != 0 && param.max_iterations < k + 1
                                    {
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
