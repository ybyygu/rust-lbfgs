//!       Limited memory BFGS (L-BFGS).
//
//  Copyright (c) 1990, Jorge Nocedal
//  Copyright (c) 2007-2010 Naoaki Okazaki
//  Copyright (c) 2018-2022 Wenping Guo
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
//
// I would like to thank the original author, Jorge Nocedal, who has been
// distributing the effieicnt and explanatory implementation in an open source
// licence.

use crate::common::*;
use crate::orthantwise::*;

use crate::core::{Problem, Progress, Report};
use crate::line::*;
use crate::math::LbfgsMath;

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
    pub orthantwise: Option<Orthantwise>,

    /// A factor for scaling initial step size.
    pub initial_inverse_hessian: f64,

    /// The maximum allowed step size for each optimization step, useful for
    /// preventing wild step.
    pub max_step_size: f64,

    /// Powell damping
    pub damping: bool,

    /// Constrains the step size to prevent wild steps, which may lead
    /// to evaluation failure. If false, the step size will be set as
    /// 1.
    pub constrain_step_size: bool,
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
            orthantwise: None,
            linesearch: LineSearch::default(),
            initial_inverse_hessian: 1.0,
            max_step_size: 1.0,
            damping: false,
            constrain_step_size: true,
        }
    }
}

/// L-BFGS optimizer.
#[derive(Default, Debug, Clone)]
pub struct Lbfgs {
    param: LbfgsParam,
}

impl Lbfgs {
    /// Set scaled gradient norm for converence test
    ///
    /// This parameter determines the accuracy with which the solution is to be
    /// found. A minimization terminates when
    ///
    /// ||g|| < epsilon * max(1, ||x||),
    ///
    /// where ||.|| denotes the Euclidean (L2) norm. The default value is 1e-5.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        assert!(epsilon.is_sign_positive(), "Invalid parameter epsilon specified.");

        self.param.epsilon = epsilon;

        self
    }

    /// Set initial step size for optimization. The default value is 1.0.
    pub fn with_initial_step_size(mut self, b: f64) -> Self {
        assert!(
            b.is_sign_positive(),
            "Invalid beta parameter for scaling the initial step size."
        );

        self.param.initial_inverse_hessian = b;

        self
    }

    /// Set the maximum allowed step size for optimization. The default value is 1.0.
    pub fn with_max_step_size(mut self, s: f64) -> Self {
        assert!(s.is_sign_positive(), "Invalid max_step_size parameter.");

        self.param.max_step_size = s;

        self
    }

    /// Enable Powell damping.
    pub fn with_damping(mut self, damped: bool) -> Self {
        self.param.damping = damped;

        self
    }

    /// Set orthantwise parameters. See [Orthantwise] for parameters.
    pub fn with_orthantwise(mut self, c: f64, start: usize, end: impl Into<Option<usize>>) -> Self {
        assert!(
            c.is_sign_positive(),
            "Invalid parameter orthantwise c parameter specified."
        );
        warn!("Only the backtracking line search is available for OWL-QN algorithm.");

        self.param.orthantwise = Orthantwise {
            c,
            start,
            end: end.into(),
            ..Default::default()
        }
        .into();

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
    /// value is 0.1. This parameter should be greater than the ftol parameter
    /// (1e-4) and smaller than 1.0.
    pub fn with_linesearch_gtol(mut self, gtol: f64) -> Self {
        assert!(
            gtol >= 0.0 && gtol < 1.0 && gtol > self.param.linesearch.ftol,
            "Invalid parameter gtol specified."
        );

        self.param.linesearch.gtol = gtol;

        self
    }

    /// Try to follow gradient only during optimization, by allowing object
    /// value rises, which removes the sufficient decrease condition constrain
    /// in line search. This option also implies Powell damping and
    /// BacktrackingStrongWolfe line search for improving robustness.
    pub fn with_gradient_only(mut self) -> Self {
        self.param.linesearch.gradient_only = true;
        self.param.damping = true;
        self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingStrongWolfe;

        self
    }

    /// Set the max number of iterations for line search.
    pub fn with_max_linesearch(mut self, n: usize) -> Self {
        self.param.linesearch.max_linesearch = n;

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
            "BacktrackingArmijo" => self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingArmijo,
            "BacktrackingStrongWolfe" => self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingStrongWolfe,
            "BacktrackingWolfe" | "Backtracking" => {
                self.param.linesearch.algorithm = LineSearchAlgorithm::BacktrackingWolfe
            }
            _ => unimplemented!(),
        }

        self
    }
}

impl Lbfgs {
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
    pub fn minimize<E, G>(self, x: &mut [f64], eval_fn: E, mut prgr_fn: G) -> Result<Report>
    where
        E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
        G: FnMut(&Progress) -> bool,
    {
        let mut state = self.build(x, eval_fn)?;

        info!("start lbfgs loop...");
        for _ in 0.. {
            if state.is_converged() {
                break;
            }
            let prgr = state.get_progress();
            let cancel = prgr_fn(&prgr);
            if cancel {
                info!("The minimization process has been canceled.");
                break;
            }
            state.propagate()?;
        }

        // Return the final value of the objective function.
        Ok(state.report())
    }
}

/// L-BFGS optimization state allowing iterative propagation
pub struct LbfgsState<'a, E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// LBFGS parameters
    vars: LbfgsParam,

    /// Define how to evaluate gradient and value
    prbl: Option<Problem<'a, E>>,
    end: usize,
    step: f64,
    k: usize,
    lm_arr: Vec<IterationData>,
    pf: Vec<f64>,
    ncall: usize,
}

impl Lbfgs {
    /// Build LBFGS state struct for iteration.
    pub fn build<'a, E>(self, x: &'a mut [f64], eval_fn: E) -> Result<LbfgsState<'a, E>>
    where
        E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
    {
        // Initialize the limited memory.
        let param = &self.param;
        let lm_arr = (0..param.m).map(|_| IterationData::new(x.len())).collect();

        let mut problem = Problem::new(x, eval_fn, param.orthantwise);

        // Evaluate the function value and its gradient.
        problem.evaluate()?;

        // Compute the L1 norm of the variable and add it to the object value.
        problem.update_owlqn_gradient();

        // Compute the search direction with current gradient.
        problem.update_search_direction();

        // Compute the initial step:
        let h0 = param.initial_inverse_hessian;
        let step = problem.search_direction().vec2norminv() * h0;

        // Apply Powell damping or not
        let damping = param.damping;
        if damping {
            info!("Powell damping Enabled.");
        }

        let fx = problem.fx;
        // FIXME: not correct
        let mut pf = vec![0.0; self.param.past];
        if self.param.past > 0 {
            pf[0] = fx;
        }
        let state = LbfgsState {
            vars: self.param.clone(),
            prbl: Some(problem),
            end: 0,
            step,
            k: 0,
            lm_arr,
            ncall: 0,
            pf,
        };

        Ok(state)
    }
}

impl<'a, E> LbfgsState<'a, E>
where
    E: FnMut(&[f64], &mut [f64]) -> Result<f64>,
{
    /// Check if stopping critera met. Panics if not initialized.
    pub fn is_converged(&mut self) -> bool {
        // Monitor the progress.
        let prgr = self.get_progress();
        // FIXME: work around mut access limitation of self.pf
        let mut pf = self.pf.clone();
        let converged = satisfying_stop_conditions(&self.vars, prgr, &mut pf);
        self.pf = pf;
        converged
    }

    /// Report minimization progress. Panics if not initialized yet.
    pub fn report(&self) -> Report {
        Report::new(self.prbl.as_ref().expect("problem for report"))
    }

    /// Propagate in next LBFGS step. Return optimization progress on success.
    /// Panics if not initialized.
    pub fn propagate(&mut self) -> Result<Progress> {
        self.k += 1;

        // special case: already converged at the first point
        if self.k == 1 {
            let progress = self.get_progress();
            return Ok(progress);
        }

        // Store the current position and gradient vectors.
        let problem = self.prbl.as_mut().expect("problem for propagate");
        problem.save_state();

        // Search for an optimal step.
        self.ncall = self
            .vars
            .linesearch
            .find(problem, &mut self.step)
            .context("Failure during line search")?;

        problem.update_owlqn_gradient();

        // Update LBFGS iteration data.
        let it = &mut self.lm_arr[self.end];
        let gamma = it.update(
            &problem.x,
            &problem.xp,
            &problem.gx,
            &problem.gp,
            self.step,
            self.vars.damping,
        )?;

        // Compute the steepest direction
        problem.update_search_direction();
        let d = problem.search_direction_mut();

        // Apply LBFGS recursion procedure.
        self.end = lbfgs_two_loop_recursion(&mut self.lm_arr, d, gamma, self.vars.m, self.k - 1, self.end);

        // Now the search direction d is ready.
        let dnorm = d.vec2norm();
        ensure!(dnorm.is_sign_positive(), "invalid norm value: {dnorm}, dvector = {d:?}");
        // Constrains the step size to prevent wild steps.
        if self.vars.constrain_step_size {
            self.step = self.vars.max_step_size.min(dnorm) / dnorm;
        } else {
            self.step = 1.0;
        }

        // Constrain the search direction for orthant-wise updates.
        problem.constrain_search_direction();

        let mut progress = self.get_progress();

        Ok(progress)
    }

    fn get_progress(&self) -> Progress {
        let problem = self.prbl.as_ref().expect("problem for progress");
        Progress::new(&problem, self.k, self.ncall, self.step)
    }
}

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

/// Internal iternation data for L-BFGS
#[derive(Clone)]
struct IterationData {
    alpha: f64,

    s: Vec<f64>,

    y: Vec<f64>,

    /// vecdot(y, s)
    ys: f64,
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

    /// Updates L-BFGS correction pairs, returns Cholesky factor \gamma for
    /// scaling the initial inverse Hessian matrix $H_k^0$
    ///
    /// # Arguments
    ///
    /// * x, xp: current position, and previous position
    /// * gx, gp: current gradient and previous gradient
    /// * step: step size along search direction
    /// * damping: applying Powell damping to the gradient difference `y` helps
    ///   stabilize L-BFGS from numerical noise in function value and gradient
    ///
    fn update(&mut self, x: &[f64], xp: &[f64], gx: &[f64], gp: &[f64], step: f64, damping: bool) -> Result<f64> {
        // Update vectors s and y:
        // s_{k} = x_{k+1} - x_{k} = \alpha * d_{k}.
        // y_{k} = g_{k+1} - g_{k}.
        self.s.vecdiff(x, xp);
        let d = self.s.vec2norm();
        ensure!(d != 0.0, "x not changed with step {step}\n x = {xp:?}");
        self.y.vecdiff(gx, gp);

        // Compute scalars ys and yy:
        // ys = y^t \cdot s = 1 / \rho.
        // yy = y^t \cdot y.
        // Notice that yy is used for scaling the intial inverse hessian matrix H_0 (Cholesky factor).
        let ys = self.y.vecdot(&self.s);
        let yy = self.y.vecdot(&self.y);
        ensure!(yy != 0.0, "gx not changed\n g = {gx:?}");
        self.ys = ys;

        // Al-Baali2014JOTA: Damped Techniques for the Limited Memory BFGS
        // Method for Large-Scale Optimization. J. Optim. Theory Appl. 2014,
        // 161 (2), 688–699.
        //
        // Nocedal suggests an equivalent value of 0.8 for sigma2 (Damped BFGS
        // updating)
        let sigma2 = 0.6;
        let sigma3 = 3.0;
        if damping {
            debug!("Applying Powell damping, sigma2 = {}, sigma3 = {}", sigma2, sigma3);

            // B_k * Sk = B_k * (x_k + step*d_k - x_k) = B_k * step * d_k = -g_k * step
            let mut bs = gp.to_vec();
            bs.vecscale(-step);
            // s_k^T * B_k * s_k
            let sbs = self.s.vecdot(&bs);

            if ys < (1.0 - sigma2) * sbs {
                trace!("damping case1");
                let theta = sigma2 * sbs / (sbs - ys);
                bs.vecscale(1.0 - theta);
                bs.vecadd(&self.y, theta);
                self.y.veccpy(&bs);
            } else if ys > (1.0 + sigma3) * sbs {
                trace!("damping case2");
                let theta = sigma3 * sbs / (ys - sbs);
                bs.vecscale(1.0 - theta);
                bs.vecadd(&self.y, theta);
            } else {
                trace!("damping case3");
            }
        }

        Ok(ys / yy)
    }
}

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
        warn!("max iterations reached!");
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
    if !pf.is_empty() {
        // We don't test the stopping criterion while k < past.
        if dbg!(past) <= dbg!(k) {
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
    }

    false
}
