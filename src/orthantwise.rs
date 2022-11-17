//! Represents Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
//! algorithm, which minimizes the objective function F(x) combined
//! with the L1 norm |x| of the variables, {F(x) + C |x|}.
//!
//! As the L1 norm |x| is not differentiable at zero, the library
//! modifies function and gradient evaluations from a client program
//! suitably; a client program thus have only to return the function
//! value F(x) and gradients G(x) as usual.

use crate::math::*;

/// Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) algorithm
#[derive(Copy, Clone, Debug)]
pub struct Orthantwise {
    /// Coefficient for the L1 norm of variables.
    ///
    /// This parameter is the coefficient for the |x|, i.e., C. The
    /// default value is 1.
    pub c: f64,

    /// Start index for computing L1 norm of the variables.
    ///
    /// This parameter b (0 <= b < N) specifies the index number from
    /// which the library computes the L1 norm of the variables x,
    ///
    /// |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
    ///
    /// In other words, variables x_1, ..., x_{b-1} are not used for
    /// computing the L1 norm. Setting b (0 < b < N), one can protect
    /// variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
    /// regression) from being regularized. The default value is zero.
    pub start: usize,

    /// End index for computing L1 norm of the variables.
    ///
    /// This parameter e (0 < e <= N) specifies the index number at
    /// which the library stops computing the L1 norm of the variables
    /// x.
    pub end: Option<usize>,
}

impl Default for Orthantwise {
    fn default() -> Self {
        Orthantwise {
            c: 1.0,
            start: 0,
            end: None,
        }
    }
}

impl Orthantwise {
    /// a dirty wrapper for start and end parameters in orthantwise optimization
    fn start_end(&self, x: &[f64]) -> (usize, usize) {
        let start = self.start;
        let n = x.len();
        // do not panic when end parameter is too large
        let end = self.end.unwrap_or(n).min(n);
        assert!(start < end, "invalid start for orthantwise: {start} (end = {end})");

        (start, end)
    }

    /// Compute the L1 norm of the variable x.
    pub(crate) fn x1norm(&self, x: &[f64]) -> f64 {
        let (start, end) = self.start_end(x);

        let mut s = 0.0;
        for i in start..end {
            s += self.c * x[i].abs();
        }

        s
    }

    /// Compute the psuedo-gradient.
    pub(crate) fn pseudo_gradient(&self, pg: &mut [f64], x: &[f64], g: &[f64]) {
        let (start, end) = self.start_end(x);

        for i in 0..start {
            pg[i] = g[i];
        }

        // Compute the psuedo-gradient (see Eq 4)
        let c = self.c;
        for i in start..end {
            // Differentiable.
            if x[i] != 0.0 {
                pg[i] = g[i] + x[i].signum() * c;
            } else {
                let right_partial = g[i] + c;
                let left_partial = g[i] - c;
                if right_partial < 0.0 {
                    pg[i] = right_partial;
                } else if left_partial > 0.0 {
                    pg[i] = left_partial;
                } else {
                    pg[i] = 0.0;
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
    pub(crate) fn project(&self, x: &mut [f64], xp: &[f64], gp: &[f64]) {
        let (start, end) = self.start_end(xp);

        for i in start..end {
            let sign = if xp[i] == 0.0 { -gp[i] } else { xp[i] };
            if x[i] * sign <= 0.0 {
                x[i] = 0.0
            }
        }
    }

    pub(crate) fn constrain(&self, d: &mut [f64], pg: &[f64]) {
        let (start, end) = self.start_end(pg);

        for i in start..end {
            if d[i] * pg[i] >= 0.0 {
                d[i] = 0.0;
            }
        }
        assert_ne!(d.vec2norm(), 0.0, "invalid direction vector after constraints: {d:?}");
    }
}
