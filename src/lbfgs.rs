// header

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*header][header:1]]
//       Limited memory BFGS (L-BFGS).
//
//  Copyright (c) 1990, Jorge Nocedal
//  Copyright (c) 2007-2010 Naoaki Okazaki
//  Copyright (c) 2018, Wenping Guo
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

pub type size_t = libc::c_ulong;

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
// base:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct tag_callback_data {
    pub n: libc::c_int,
    pub instance: *mut libc::c_void,
    pub proc_evaluate: lbfgs_evaluate_t,
    pub proc_progress: lbfgs_progress_t,
}

pub type callback_data_t = tag_callback_data;

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
// old:1 ends here

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
    fn owlqn_x1norm(&self, start: usize, end: usize) -> T;
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

    fn owlqn_x1norm(&self, start: usize, end: usize) -> f64 {
        let mut s = 0.0;
        for i in start..end {
            s += self[i].abs();
        }
        s
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
    let v = z.owlqn_x1norm(1, 3);
    assert_eq!(v, 16.0);
}
// vector operations:1 ends here

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
#[derive(Debug, Copy, Clone)]
pub struct LineSearchParam {
    algorithm: LineSearchAlgorithm,

    /// ftol and gtol are nonnegative input variables. (in this reverse
    /// communication implementation gtol is defined in a common statement.)
    ///
    /// Termination occurs when the sufficient decrease condition and the
    /// directional derivative condition are satisfied.
    ///
    /// A parameter to control the accuracy of the line search routine.
    ///
    ///  The default value is \c 1e-4. This parameter should be greater
    ///  than zero and smaller than \c 0.5.
    ftol: f64,

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 0.9. If the function and gradient evaluations are
    /// inexpensive with respect to the cost of the iteration (which is
    /// sometimes the case when solving very large problems) it may be
    /// advantageous to set this parameter to a small value. A typical small
    /// value is 0.1. This parameter shuold be greater than the \ref ftol
    /// parameter (1e-4) and smaller than 1.0.
    gtol: f64,

    /// xtol is a nonnegative input variable. termination occurs when the
    /// relative width of the interval of uncertainty is at most xtol.
    ///
    /// The machine precision for floating-point values.
    ///
    ///  This parameter must be a positive value set by a client program to
    ///  estimate the machine precision. The line search routine will terminate
    ///  with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
    ///  of the interval of uncertainty is less than this parameter.
    xtol: f64,

    /// The minimum step of the line search routine.
    ///
    /// The default value is \c 1e-20. This value need not be modified unless
    /// the exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    min_step: f64,

    /// The maximum step of the line search.
    ///
    ///  The default value is \c 1e+20. This value need not be modified unless
    ///  the exponents are too large for the machine being used, or unless the
    ///  problem is extremely badly scaled (in which case the exponents should
    ///  be increased).
    max_step: f64,

    /// The maximum number of trials for the line search.
    ///
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is 40.
    ///
    max_linesearch: usize,

    condition: LineSearchCondition,

    /// A coefficient for the Wolfe condition.
    /// *  This parameter is valid only when the backtracking line-search
    /// *  algorithm is used with the Wolfe condition,
    /// *  The default value is 0.9. This parameter should be greater the ftol parameter and smaller than 1.0.
    wolfe: f64,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum LineSearchCondition {
    Armijo,
    Wolfe,
    StrongWolf,
}

// TODO: better defaults
impl Default for LineSearchParam {
    fn default() -> Self {
        LineSearchParam {
            ftol: 1e-4,
            gtol: 0.9,
            xtol: 1e-16,
            min_step: 1e-20,
            max_step: 1e20,
            max_linesearch: 40,

            // FIXME: only useful for backtracking
            wolfe: 0.9,
            condition: LineSearchCondition::StrongWolf,
            algorithm: LineSearchAlgorithm::default(),
        }
    }
}

trait LineSearching<E> {
    /// Apply line search algorithm to find satisfactory step size
    ///
    /// # Arguments
    ///
    /// * step: initial step size
    /// * direction: proposed searching direction
    ///
    /// # Return
    ///
    /// * the number of line searching iterations
    fn find(&mut self, step: &mut f64, direction: &[f64], eval_fn: E) -> Result<usize>;
}
// new:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
pub type line_search_proc = Option<
    unsafe extern "C" fn(
        // The number of variables.
        n: libc::c_int,
        // The array of variables.
        x: &mut [f64],
        // Evaluated function value
        f: *mut lbfgsfloatval_t,
        // Evaluated gradient array
        g: &mut [f64],
        // Search direction array
        s: &[f64],
        // Step size
        stp: *mut lbfgsfloatval_t,
        // Variable vector of previous step
        xp: &[f64],
        // Gradient vector of previous step
        gp: &[f64],
        // work array?
        wp: &mut [f64],
        // callback struct
        cd: *mut callback_data_t,
        // LBFGS parameter
        param: &LbfgsParam,
    ) -> libc::c_int,
>;
// old:1 ends here

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
/// Line search algorithms.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LineSearchAlgorithm {
    /// MoreThuente method proposd by More and Thuente.
    MoreThuente,

    ///
    /// Backtracking method with the Armijo condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// the sufficient decrease (Armijo) condition,
    ///   - f(x + a * d) <= f(x) + ftol * a * g(x)^T d,
    ///
    /// where x is the current point, d is the current search direction, and
    /// a is the step length.
    ///
    BacktrackingArmijo,

    /// The backtracking method with the defualt (regular Wolfe) condition.
    Backtracking,

    /// Backtracking method with strong Wolfe condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (BacktrackingArmijo)
    /// and the following condition,
    /// FIXME: gtol vs wolfe?
    ///   - |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
    ///
    /// where x is the current point, d is the current search direction, and
    /// a is the step length.
    ///
    BacktrackingStrongWolfe,

    ///
    /// Backtracking method with regular Wolfe condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (BacktrackingArmijo)
    /// and the curvature condition,
    /// FIXME: gtol vs wolfe?
    ///   - g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d,
    ///
    /// where x is the current point, d is the current search direction, and a
    /// is the step length.
    ///
    BacktrackingWolfe,
}

impl Default for LineSearchAlgorithm {
    /// The default algorithm (MoreThuente method).
    fn default() -> Self {
        LineSearchAlgorithm::MoreThuente
    }
}
// new:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
pub type unnamed_0 = libc::c_uint;
/// Backtracking method with strong Wolfe condition.
///  The backtracking method finds the step length such that it satisfies
///  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
///  and the following condition,
///    - |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
///
///  where x is the current point, d is the current search direction, and
///  a is the step length.

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

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
/// The purpose of mcsrch is to find a step which satisfies a sufficient
/// decrease condition and a curvature condition.
mod mcsrch {
    // dependencies
    use super::mcstep;
    use super::LbfgsMath;
    use super::LineSearchParam;
    use super::Evaluate;
    use super::LineSearching;
    use super::Problem;
    use quicli::prelude::{bail, Result};

    /// A struct represents MCSRCH subroutine in original lbfgs.f by J. Nocera
    struct Mcsrch<'a> {
        /// `prob` holds input variables `x`, gradient `gx` arrays of length
        /// n, and function value `fx`. on input it must contain the base point
        /// for the line search. on output it contains data on x + stp*s.
        prob: &'a mut Problem,

        param: LineSearchParam,
    }

    impl<'a, E> LineSearching<E> for Mcsrch<'a>
    where
        E: FnMut(&mut Problem) -> Result<()>,
    {
        /// Find a step which satisfies a sufficient decrease condition and a
        /// curvature condition (strong wolfe conditions).
        ///
        /// # Arguments
        ///
        /// * stp: a nonnegative variable. on input stp contains an initial
        /// estimate of a satisfactory step. on output stp contains the final
        /// estimate.
        /// * direction: is an input array of length n which specifies the search direction.
        ///
        /// # Return
        ///
        /// * the final estimate of a satisfactory step.
        ///
        /// # Example
        ///
        /// - TODO
        ///
        fn find(&mut self, stp: &mut f64, s: &[f64], mut eval_fn: E) -> Result<usize> {
            // Check the input parameters for errors.
            if !stp.is_sign_positive() {
                bail!("A logic error (negative line-search step) occurred.");
            }

            // Compute the initial gradient in the search direction.
            // vecdot(&mut dginit, g, s, n);
            let mut dginit = self.prob.gx.vecdot(s);

            // Make sure that s points to a descent direction.
            if dginit.is_sign_positive() {
                bail!("The current search direction increases the objective function value!");
            }

            // Initialize local variables.
            let param = &self.param;
            let mut brackt = false;
            let mut stage1 = 1;
            let finit = self.prob.fx;
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
                // vecadd(x, s, *stp, n);
                self.prob.x.vecadd(&s, *stp);

                // Evaluate the function and gradient values.
                // *f = cd(x, g, *stp)?;
                eval_fn(&mut self.prob)?;

                let f = self.prob.fx;

                // vecdot(&mut dg, g, s, n);
                dg = self.prob.gx.vecdot(s);

                // FIXME: wolfe constant?
                ftest1 = finit + *stp * dgtest;
                count += 1;

                // Test for errors and convergence.
                if brackt && (*stp <= stmin || stmax <= *stp || uinfo != 0) {
                    /* Rounding errors prevent further progress. */
                    bail!("LBFGSERR_ROUNDING_ERROR");
                }
                if *stp == param.max_step && f <= ftest1 && dg <= dgtest {
                    /* The step is the maximum value. */
                    bail!("LBFGSERR_MAXIMUMSTEP");
                }
                if *stp == param.min_step && (ftest1 < f || dgtest <= dg) {
                    /* The step is the minimum value. */
                    bail!("LBFGSERR_MINIMUMSTEP");
                }
                if brackt && stmax - stmin <= param.xtol * stmax {
                    /* Relative width of the interval of uncertainty is at most xtol. */
                    bail!("LBFGSERR_WIDTHTOOSMALL");
                }
                if param.max_linesearch <= count {
                    // Maximum number of iteration.
                    bail!("LBFGSERR_MAXIMUMLINESEARCH");
                }
                if f <= ftest1 && dg.abs() <= param.gtol * -dginit {
                    // The sufficient decrease condition and the directional derivative condition hold.
                    return Ok(count);
                }

                // In the first stage we seek a step for which the modified
                // function has a nonpositive value and nonnegative derivative.
                if 0 != stage1 && f <= ftest1 && param.ftol.min(param.gtol) * dginit <= dg {
                    stage1 = 0
                }

                // A modified function is used to predict the step only if
                // we have not obtained a step for which the modified
                // function has a nonpositive function value and nonnegative
                // derivative, and if a lower function value has been
                // obtained but the decrease is not sufficient.
                if 0 != stage1 && ftest1 < f && f <= fx {
                    // Define the modified function and derivative values.
                    fm = f - *stp * dgtest;
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
                        f,
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
// new:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
unsafe extern "C" fn line_search_morethuente(
    // The number of variables.
    n: libc::c_int,
    // The array of variables.
    x: &mut [f64],
    // Evaluated function value
    f: *mut lbfgsfloatval_t,
    // Evaluated gradient array
    g: &mut [f64],
    // Search direction array
    s: &[f64],
    // Step size
    stp: *mut lbfgsfloatval_t,
    // Variable vector of previous step
    xp: &[f64],
    // Gradient vector of previous step
    gp: &[f64],
    // work array?
    wa: &mut [f64],
    // callback struct
    cd: *mut callback_data_t,
    // LBFGS parameter
    param: &LbfgsParam,
) -> libc::c_int {
    // quick wrapper

    let param = &param.linesearch;
    // FIXME: remove
    let n = n as usize;
    // let s = unsafe { ::std::slice::from_raw_parts(s, n) };
    // let x = unsafe { ::std::slice::from_raw_parts_mut(x, n) };
    // let g = unsafe { ::std::slice::from_raw_parts_mut(g, n) };
    // let xp = unsafe { ::std::slice::from_raw_parts(xp, n) };
    // let gp = unsafe { ::std::slice::from_raw_parts(gp, n) };

    // Check the input parameters for errors.
    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Compute the initial gradient in the search direction.
    // vecdot(&mut dginit, g, s, n);
    let mut dginit = g.vecdot(s);

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
                || uinfo != 0) || brackt && stmax - stmin <= param.xtol * stmax
        {
            *stp = stx
        }

        // Compute the current value of x: x <- x + (*stp) * s.
        // veccpy(x, xp, n);
        x.veccpy(xp);
        // vecadd(x, s, *stp, n);
        x.vecadd(s, *stp);

        // Evaluate the function and gradient values.
        // FIXME: improve
        *f = (*cd).proc_evaluate.expect("non-null function pointer")(
            (*cd).instance,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            (*cd).n,
            *stp,
        );

        // vecdot(&mut dg, g, s, n);
        let mut dg = g.vecdot(s);
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
            return count as i32;
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

            if 0.66 * prev_width <= (sty - stx).abs() {
                *stp = stx + 0.5 * (sty - stx)
            }

            prev_width = width;
            width = (sty - stx).abs()
        }
    }
}
// old:1 ends here

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

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
pub mod backtracking {
    // dependencies
    use super::Evaluate;
    use super::LbfgsMath ;
    use super::LineSearchCondition;
    use super::LineSearchParam;
    use super::LineSearching;
    use super::Problem;
    use quicli::prelude::{bail, Result};

    pub struct BackTracking<'a> {
        /// `prob` holds input variables `x`, gradient `gx` arrays of length
        /// n, and function value `fx`. on input it must contain the base point
        /// for the line search. on output it contains data on x + stp*s.
        prob: &'a mut Problem,

        param: LineSearchParam,
    }

    impl<'a, E> LineSearching<E> for BackTracking<'a>
    where
        E: FnMut(&mut Problem) -> Result<()>,
    {
        fn find(&mut self, stp: &mut f64, s: &[f64], mut eval_fn: E) -> Result<usize> {
            let mut width: f64 = 0.;
            let mut dg: f64 = 0.;
            let mut finit: f64 = 0.;
            let mut dgtest: f64 = 0.;
            let dec: f64 = 0.5f64;
            let inc: f64 = 2.1f64;

            // Check the input parameters for errors.
            if *stp <= 0.0 {
                bail!("LBFGSERR_INVALIDPARAMETERS");
            }

            // Compute the initial gradient in the search direction.
            // vecdot(&mut dginit, g, s, n);
            let mut dginit = self.prob.gx.vecdot(s);

            // Make sure that s points to a descent direction.
            if dginit.is_sign_positive() {
                bail!("LBFGSERR_INCREASEGRADIENT");
            }

            // The initial value of the objective function.
            finit = self.prob.fx;

            let param = &self.param;
            dgtest = param.ftol * dginit;

            let mut count = 0;
            loop {
                // FIXME: handle xp in Problem
                // veccpy(x, xp, n);
                // vecadd(x, s, *stp, n);
                self.prob.x.vecadd(s, *stp);

                // Evaluate the function and gradient values.
                eval_fn(&mut self.prob)?;

                count += 1;
                if self.prob.fx > finit + *stp * dgtest {
                    width = dec
                } else {
                    // The sufficient decrease condition (Armijo condition).
                    if param.condition == LineSearchCondition::Armijo {
                        // Exit with the Armijo condition.
                        return Ok(count);
                    }

                    // Check the Wolfe condition.
                    // vecdot(&mut dg, g, s, n);
                    dg = self.prob.gx.vecdot(s);
                    // FIXME: param.gtol vs param.wolfe?
                    if dg < param.wolfe * dginit {
                        width = inc
                    } else if param.condition == LineSearchCondition::Wolfe {
                        // Exit with the regular Wolfe condition.
                        return Ok(count);
                    } else if dg > -param.wolfe * dginit {
                        width = dec
                    } else {
                        return Ok(count);
                    }
                }

                // The step is the minimum value.
                if *stp < param.min_step {
                    bail!("LBFGSERR_MINIMUMSTEP");
                }
                // The step is the maximum value.
                if *stp > param.max_step {
                    bail!("LBFGSERR_MAXIMUMSTEP");
                }
                // Maximum number of iteration.
                if param.max_linesearch <= count {
                    bail!("LBFGSERR_MAXIMUMLINESEARCH");
                }

                *stp *= width
            }

            bail!("LOGICAL ERROR!");
        }
    }

    // backtracking Owlqn variant
    pub struct BacktrackingOwlqn<'a> {
        /// `prob` holds input variables `x`, gradient `gx` arrays of length
        /// n, and function value `fx`. on input it must contain the base point
        /// for the line search. on output it contains data on x + stp*s.
        prob: &'a mut Problem,

        param: LineSearchParam,
    }

    impl<'a, E> LineSearching<E> for BacktrackingOwlqn<'a> {
        fn find(&mut self, stp: &mut f64, s: &[f64], mut eval_fn: E) -> Result<usize> {
            // quick wrapper
            let param = &self.param;
            let x = &self.prob.x;

            unimplemented!()
        }
    }
}
// new:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
unsafe extern "C" fn line_search_backtracking(
    // The number of variables.
    n: libc::c_int,
    // The array of variables.
    x: &mut [f64],
    // Evaluated function value
    f: *mut lbfgsfloatval_t,
    // Evaluated gradient array
    g: &mut [f64],
    // Search direction array
    s: &[f64],
    // Step size
    stp: *mut lbfgsfloatval_t,
    // Variable vector of previous step
    xp: &[f64],
    // Gradient vector of previous step
    gp: &[f64],
    // work array?
    wp: &mut [f64],
    // callback struct
    cd: *mut callback_data_t,
    // LBFGS parameter
    param: &LbfgsParam,
) -> libc::c_int {
    // quick wrapper
    let param = &param.linesearch;

    // FIXME: remove
    let n = n as usize;
    // let s = unsafe { ::std::slice::from_raw_parts(s, n) };
    // let x = unsafe { ::std::slice::from_raw_parts_mut(x, n) };
    // let g = unsafe { ::std::slice::from_raw_parts_mut(g, n) };
    // let xp = unsafe { ::std::slice::from_raw_parts(xp, n) };
    // let gp = unsafe { ::std::slice::from_raw_parts(gp, n) };
    // let wp = unsafe { ::std::slice::from_raw_parts_mut(wp, n) };

    let mut width: f64 = 0.;
    let dec: f64 = 0.5;
    let inc: f64 = 2.1;

    // Check the input parameters for errors.
    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // Compute the initial gradient in the search direction.
    // vecdot(&mut dginit, g, s, n);
    let mut dginit = g.vecdot(s);

    // Make sure that s points to a descent direction.
    if 0.0 < dginit {
        return LBFGSERR_INCREASEGRADIENT as libc::c_int;
    }

    // The initial value of the objective function.
    let mut finit = *f;
    let mut dgtest = param.ftol * dginit;

    use crate::LineSearchAlgorithm::*;
    let mut count = 0;
    loop {
        // veccpy(x, xp, n);
        x.veccpy(xp);
        // vecadd(x, s, *stp, n);
        x.vecadd(s, *stp);

        // FIXME: improve below
        // Evaluate the function and gradient values.
        *f = (*cd).proc_evaluate.expect("non-null function pointer")(
            (*cd).instance,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            (*cd).n,
            *stp,
        );

        count += 1;
        if *f > finit + *stp * dgtest {
            width = dec
        } else if param.algorithm == BacktrackingArmijo {
            // Exit with the Armijo condition.
            return count as i32;
        } else {
            // Check the Wolfe condition.
            // vecdot(&mut dg, g, s, n);
            let dg = g.vecdot(s);
            if dg < param.wolfe * dginit {
                width = inc
            } else if param.algorithm == BacktrackingWolfe {
                // Exit with the regular Wolfe condition.
                return count as i32;
            } else if dg > -param.wolfe * dginit {
                width = dec
            } else {
                return count as i32;
            }
        }
        if *stp < param.min_step {
            /* The step is the minimum value. */
            return LBFGSERR_MINIMUMSTEP as libc::c_int;
        }
        if *stp > param.max_step {
            /* The step is the maximum value. */
            return LBFGSERR_MAXIMUMSTEP as libc::c_int;
        }
        if param.max_linesearch <= count {
            /* Maximum number of iteration. */
            return LBFGSERR_MAXIMUMLINESEARCH as libc::c_int;
        }
        *stp *= width
    }
}
// old:1 ends here

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
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
// new:1 ends here

// new

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*new][new:1]]
fn owlqn_project(d: &mut [f64], sign: &[f64], start: usize, end: usize) {
    let mut i = start;
    while i < end {
        if d[i] * sign[i] <= 0.0 {
            d[i] = 0.0
        }
        i += 1
    }
}
// new:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
unsafe extern "C" fn line_search_backtracking_owlqn(
    // The number of variables.
    n: libc::c_int,
    // The array of variables.
    x: &mut [f64],
    // Evaluated function value
    f: *mut lbfgsfloatval_t,
    // Evaluated gradient array
    g: &mut [f64],
    // Search direction array
    s: &[f64],
    // Step size
    stp: *mut lbfgsfloatval_t,
    // Variable vector of previous step
    xp: &[f64],
    // Gradient vector of previous step
    gp: &[f64],
    // work array?
    wp: &mut [f64],
    // callback struct
    cd: *mut callback_data_t,
    // LBFGS parameter
    _param: &LbfgsParam,
) -> libc::c_int {
    let mut i: libc::c_int = 0;
    let mut width: f64 = 0.5f64;
    let mut finit: f64 = *f;

    // Check the input parameters for errors.
    if *stp <= 0.0f64 {
        return LBFGSERR_INVALIDPARAMETERS as libc::c_int;
    }

    // quick wrapper
    let n = n as usize;
    // let s = unsafe { ::std::slice::from_raw_parts(s, n) };
    // let x = unsafe { ::std::slice::from_raw_parts_mut(x, n) };
    // let g = unsafe { ::std::slice::from_raw_parts_mut(g, n) };
    // let xp = unsafe { ::std::slice::from_raw_parts(xp, n) };
    // let gp = unsafe { ::std::slice::from_raw_parts(gp, n) };
    // let wp = unsafe { ::std::slice::from_raw_parts_mut(wp, n) };

    // Choose the orthant for the new point.
    // FIXME: float == 0.0??
    for i in 0..n {
        wp[i] = if xp[i] == 0.0 { -gp[i] } else { xp[i] };
    }

    let mut count = 0;
    // FIXME: review
    // quick wrapper
    let param = &_param.linesearch;
    loop {
        // Update the current point.
        // veccpy(x, xp, n);
        x.veccpy(xp);
        // vecadd(x, s, *stp, n);
        x.vecadd(s, *stp);

        // The current point is projected onto the orthant.
        owlqn_project(
            x,
            wp,
            _param.orthantwise_start as usize,
            _param.orthantwise_end as usize,
        );

        // Evaluate the function and gradient values.
        *f = (*cd).proc_evaluate.expect("non-null function pointer")(
            (*cd).instance,
            x.as_ptr(),
            g.as_mut_ptr(),
            (*cd).n,
            *stp,
        );

        // Compute the L1 norm of the variables and add it to the object value.
        let norm = x.owlqn_x1norm(
            _param.orthantwise_start as usize,
            _param.orthantwise_end as usize,
        );
        *f += norm * _param.orthantwise_c;

        count += 1;

        let mut dgtest = 0.0f64;
        for i in 0..n {
            dgtest += (x[i] - xp[i]) * gp[i];
        }

        if *f <= finit + param.ftol * dgtest {
            // The sufficient decrease condition.
            return count as i32;
        }

        if *stp < param.min_step {
            // The step is the minimum value.
            return LBFGSERR_MINIMUMSTEP as libc::c_int;
        }

        if *stp > param.max_step {
            // The step is the maximum value.
            return LBFGSERR_MAXIMUMSTEP as libc::c_int;
        }
        if param.max_linesearch <= count {
            // Maximum number of iteration.
            return LBFGSERR_MAXIMUMLINESEARCH as libc::c_int;
        }

        *stp *= width
    }
}

unsafe extern "C" fn owlqn_pseudo_gradient(
    pg: &mut [f64],
    x: &[f64],
    g: &[f64],
    n: usize,
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

    for i in end..n {
        pg[i] = g[i];
    }
}
// old:1 ends here

// problem

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*problem][problem:1]]
#[repr(C)]
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
}

pub trait Evaluate {
    /// Evaluate function value `fx` and gradient `gx` at `x`
    fn eval(&mut self) -> Result<()>;
}
// problem:1 ends here

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

/// Store optimization progress data
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ProgressData<'a> {
    /// The current values of variables
    pub arr_x: &'a [f64],
    /// The current gradient values of variables.
    pub grd_x: &'a [f64],
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

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LBFGS<F, G>
where
    F: FnMut(&mut Problem) -> Result<()>,
    G: FnMut(&ProgressData) -> bool,
{
    pub param: LbfgsParam,
    evaluate: Option<F>,
    progress: Option<G>,
}

impl<F, G> Default for LBFGS<F, G>
where
    F: FnMut(&mut Problem) -> Result<()>,
    G: FnMut(&ProgressData) -> bool,
{
    fn default() -> Self {
        LBFGS {
            param: LbfgsParam::default(),
            evaluate: None,
            progress: None,
        }
    }
}
// common:1 ends here

// old

// [[file:~/Workspace/Programming/rust-libs/lbfgs/lbfgs.note::*old][old:1]]
#[no_mangle]
pub unsafe extern "C" fn lbfgs(
    x: &mut [f64],
    ptr_fx: &mut f64,
    mut proc_evaluate: lbfgs_evaluate_t,
    mut proc_progress: lbfgs_progress_t,
    mut instance: *mut libc::c_void,
    param: &LbfgsParam,
) -> Result<i32> {
    // FIXME: make param immutable
    let mut param = param.clone();

    let mut ls: libc::c_int = 0;

    let m = param.m;
    let mut ys = 0.;
    let mut yy = 0.;
    let mut linesearch: line_search_proc = Some(line_search_morethuente);

    // Construct a callback data.
    let n = x.len();
    let mut cd: callback_data_t = tag_callback_data {
        n: 0,
        instance: 0 as *mut libc::c_void,
        proc_evaluate: None,
        proc_progress: None,
    };

    cd.n = n as i32;
    cd.instance = instance;
    cd.proc_evaluate = proc_evaluate;
    cd.proc_progress = proc_progress;

    // Check the input parameters for errors.
    param.validate()?;

    // FIXME: make param immutable
    if param.orthantwise_start < 0 || (n as i32) < param.orthantwise_start {
        bail!("LBFGSERR_INVALID_ORTHANTWISE_START");
    }
    if param.orthantwise_end < 0 {
        param.orthantwise_end = n as i32
    }
    if (n as i32) < param.orthantwise_end {
        bail!("LBFGSERR_INVALID_ORTHANTWISE_END");
    }

    use self::LineSearchAlgorithm::*;
    if param.orthantwise_c != 0.0 {
        match param.linesearch.algorithm {
            // FIXME: review below
            // Only the backtracking method is available.
            _ => linesearch = Some(line_search_backtracking_owlqn),
        }
    } else {
        match param.linesearch.algorithm {
            MoreThuente => linesearch = Some(line_search_morethuente),
            BacktrackingArmijo | BacktrackingWolfe | BacktrackingStrongWolfe => {
                linesearch = Some(line_search_backtracking)
            }
            _ => {
                bail!("LBFGSERR_INVALID_LINESEARCH");
            }
        }
    }

    // Allocate working space.
    let mut xp = vec![0.0; n];
    let mut g = vec![0.0; n];
    let mut gp = g.clone();
    let mut d = vec![0.0; n];
    let mut w = vec![0.0; n];

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
    let mut xnorm = 0.;
    // let mut x = arr_x.as_mut_ptr();

    fx = cd.proc_evaluate.expect("non-null function pointer")(
        cd.instance,
        x.as_mut_ptr(),
        g.as_mut_ptr(),
        cd.n,
        0.0,
    );
    if 0.0 != param.orthantwise_c {
        // Compute the L1 norm of the variable and add it to the object value.
        // xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
        xnorm = x.owlqn_x1norm(param.orthantwise_start as usize, param.orthantwise_end as usize);

        fx += xnorm * param.orthantwise_c;
        owlqn_pseudo_gradient(
            &mut pg,
            &x,
            &g,
            n,
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
    // vec2norm(&mut xnorm, x, n);
    xnorm = x.vec2norm().max(1.0);
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
    info!("start lbfgs loop...");
    loop {
        // Store the current position and gradient vectors.
        xp.veccpy(&x);
        gp.veccpy(&g);

        // Search for an optimal step.
        if param.orthantwise_c == 0.0 {
            ls = linesearch.expect("non-null function pointer")(
                n as i32, x, &mut fx, &mut g, &d, &mut step, &xp, &gp, &mut w, &mut cd, &param,
            )
        } else {
            ls = linesearch.expect("non-null function pointer")(
                n as i32, x, &mut fx, &mut g, &d, &mut step, &xp, &pg, &mut w, &mut cd, &param,
            );
            owlqn_pseudo_gradient(
                &mut pg,
                &x,
                &g,
                n,
                param.orthantwise_c,
                param.orthantwise_start as usize,
                param.orthantwise_end as usize,
            );
        }
        if ls < 0 {
            error!("line search failed, revert to the previous point!");
            // Revert to the previous point.
            x.veccpy(&xp);
            g.veccpy(&gp);

            ret = ls;
            break;
        }
        // Compute x and g norms.
        xnorm = x.vec2norm();
        gnorm = if param.orthantwise_c == 0.0 {
            g.vec2norm()
        } else {
            pg.vec2norm()
        };

        // Report the progress.
        if cd.proc_progress.is_some() {
            ret = cd.proc_progress.expect("non-null function pointer")(
                cd.instance,
                x.as_ptr(),
                g.as_ptr(),
                fx,
                xnorm,
                gnorm,
                step,
                cd.n,
                k as i32,
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
                    ret = LBFGS_STOP as libc::c_int;
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
            ret = LBFGSERR_MAXIMUMITERATION as libc::c_int;
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
        ys = it.y.vecdot(&it.s);
        yy = it.y.vecdot(&it.y);

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
        if param.orthantwise_c == 0.0f64 {
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
// old:1 ends here
