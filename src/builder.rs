// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*imports][imports:1]]
use crate::core::*;

use crate::lbfgs::*;
// imports:1 ends here

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*traits][traits:1]]
pub trait EvaluateLbfgs {
    /// Evaluate gradient `gx` at positions `x`, returns function value on success.
    fn evaluate(&mut self, x: &[f64], gx: &mut [f64]) -> Result<f64>;
}

impl<T> EvaluateLbfgs for T
where
    T: FnMut(&[f64], &mut [f64]) -> Result<f64>
{
    /// Evaluate gradient `gx` at positions `x`, returns function value on success.
    fn evaluate(&mut self, x: &[f64], gx: &mut [f64]) -> Result<f64> {
        self(x, gx)
    }
}
// traits:1 ends here

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*base][base:1]]
pub struct Lbfgs<'a> {
    pub(crate) vars: LbfgsParam,
    /// Define how to evaluate gradient and value
    eval: Option<Box<dyn EvaluateLbfgs + 'a>>,
    end: usize,
}

impl<'a> Default for Lbfgs<'a> {
    fn default() -> Self {
        Self {
            vars: LbfgsParam::default(),
            eval: None,
            end: 0,
        }
    }
}
// base:1 ends here

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*parameters][parameters:1]]
use crate::line::*;

/// Create lbfgs optimizer with epsilon convergence
impl<'a> Lbfgs<'a>
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

        self.vars.epsilon = epsilon;

        self
    }

    /// Set initial step size for optimization. The default value is 1.0.
    pub fn with_initial_step_size(mut self, b: f64) -> Self {
        assert!(
            b.is_sign_positive(),
            "Invalid beta parameter for scaling the initial step size."
        );

        self.vars.initial_inverse_hessian = b;

        self
    }

    /// Set the maximum allowed step size for optimization. The default value is 1.0.
    pub fn with_max_step_size(mut self, s: f64) -> Self {
        assert!(s.is_sign_positive(), "Invalid max_step_size parameter.");

        self.vars.max_step_size = s;

        self
    }

    /// Enable Powell damping.
    pub fn with_damping(mut self, damped: bool) -> Self {
        self.vars.damping = damped;

        self
    }

    /// Set orthantwise parameters
    pub fn with_orthantwise(mut self, c: f64, start: usize, end: usize) -> Self {
        assert!(
            c.is_sign_positive(),
            "Invalid parameter orthantwise c parameter specified."
        );
        warn!("Only the backtracking line search is available for OWL-QN algorithm.");

        self.vars.orthantwise = true;
        self.vars.owlqn.c = c;
        self.vars.owlqn.start = start as i32;
        self.vars.owlqn.end = end as i32;

        self
    }

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 1e-4. This parameter should be greater
    /// than zero and smaller than 0.5.
    pub fn with_linesearch_ftol(mut self, ftol: f64) -> Self {
        assert!(ftol >= 0.0, "Invalid parameter ftol specified.");
        self.vars.linesearch.ftol = ftol;

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
            gtol >= 0.0 && gtol < 1.0 && gtol > self.vars.linesearch.ftol,
            "Invalid parameter gtol specified."
        );

        self.vars.linesearch.gtol = gtol;

        self
    }

    /// Try to follow gradient only during optimization, by allowing object
    /// value rises, which removes the sufficient decrease condition constrain
    /// in line search. This option also implies Powell damping and
    /// BacktrackingStrongWolfe line search for improving robustness.
    pub fn with_gradient_only(mut self) -> Self {
        self.vars.linesearch.gradient_only = true;
        self.vars.damping = true;
        self.vars.linesearch.algorithm = LineSearchAlgorithm::BacktrackingStrongWolfe;

        self
    }

    /// Set the max number of iterations for line search.
    pub fn with_max_linesearch(mut self, n: usize) -> Self {
        self.vars.linesearch.max_linesearch = n;

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

        self.vars.linesearch.xtol = xtol;
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

        self.vars.linesearch.min_step = min_step;
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
        self.vars.max_iterations = niter;
        self
    }

    /// The maximum allowed number of evaluations of function value and
    /// gradients. This number could be larger than max_iterations since line
    /// search procedure may involve one or more evaluations.
    ///
    /// Setting this parameter to zero continues an optimization process until a
    /// convergence or error. The default value is 0.
    pub fn with_max_evaluations(mut self, neval: usize) -> Self {
        self.vars.max_evaluations = neval;
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

        self.vars.past = past;
        self.vars.delta = delta;
        self
    }

    /// Select line search algorithm
    ///
    /// The default is "MoreThuente" line search algorithm.
    pub fn with_linesearch_algorithm(mut self, algo: &str) -> Self {
        match algo {
            "MoreThuente" => self.vars.linesearch.algorithm = LineSearchAlgorithm::MoreThuente,
            "BacktrackingArmijo" => {
                self.vars.linesearch.algorithm = LineSearchAlgorithm::BacktrackingArmijo
            }
            "BacktrackingStrongWolfe" => {
                self.vars.linesearch.algorithm = LineSearchAlgorithm::BacktrackingStrongWolfe
            }
            "BacktrackingWolfe" | "Backtracking" => {
                self.vars.linesearch.algorithm = LineSearchAlgorithm::BacktrackingWolfe
            }
            _ => unimplemented!(),
        }

        self
    }
}
// parameters:1 ends here

// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*evaluation][evaluation:1]]
impl<'a> Lbfgs<'a> {
    pub fn evaluate_with<P: 'a>(mut self, p: P) -> Self
    where
        P: EvaluateLbfgs,
    {
        self.eval = Some(Box::new(p));
        self
    }
}
// evaluation:1 ends here
