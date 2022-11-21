use approx::*;

/// Multidimensional Rosenbrock test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
///
/// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
///
/// # Reference
///
/// - https://en.wikipedia.org/wiki/Rosenbrock_function
#[test]
fn test_lbfgs_rosenbrock() {
    use liblbfgs::{default_evaluate, default_progress, lbfgs, Progress};

    const N: usize = 100;

    // Initialize the variables
    let mut x = [0.0 as f64; N];
    for i in (0..N).step_by(2) {
        x[i] = -1.2;
        x[i + 1] = 1.0;
    }

    let prb = lbfgs()
        .minimize(&mut x, default_evaluate(), default_progress())
        .expect("lbfgs minimize");

    // Iteration 37:
    // fx = 0.0000000000000012832127771605377, x[0] = 0.9999999960382451, x[1] = 0.9999999917607568
    // xnorm = 9.999999938995018, gnorm = 0.0000009486547293218877, step = 1

    assert_relative_eq!(0.0, prb.fx, epsilon = 1e-4);
    for i in 0..N {
        assert_relative_eq!(1.0, x[i], epsilon = 1e-4);
    }

    // OWL-QN
    let prb = lbfgs()
        .with_orthantwise(1.0, 0, 99)
        .minimize(&mut x, default_evaluate(), default_progress())
        .expect("lbfgs owlqn minimize");

    // Iteration 171:
    // fx = 43.50249999999999, x[0] = 0.2500000069348678, x[1] = 0.057500004213084016
    // xnorm = 1.8806931246657475, gnorm = 0.00000112236896804755, step = 1

    assert_relative_eq!(43.5025, prb.fx, epsilon = 1e-4);
    assert_relative_eq!(0.2500, x[0], epsilon = 1e-4);
    assert_relative_eq!(0.0575, x[1], epsilon = 1e-4);
}

#[test]
fn test_lbfgs_booth() {
    use liblbfgs::{default_progress, lbfgs, Progress};

    const N: usize = 2;

    // 1. Initialize the variables
    let mut x = [-1.2, 1.0];

    // 2. Defining how to evaluate function and gradient
    let evaluate = |x: &[f64], gx: &mut [f64]| {
        let x1 = x[0];
        let x2 = x[1];
        let mut fx = (x1 + 2.0 * x2 - 7.0).powi(2) + (2.0 * x1 + x2 - 5.0).powi(2);
        gx[0] = 10.0 * x1 + 8.0 * x2 - 34.0;
        gx[1] = 8.0 * x1 + 10.0 * x2 - 38.0;

        Ok(fx)
    };

    let _ = lbfgs()
        .minimize(&mut x, evaluate, default_progress())
        .expect("lbfgs minimize");

    assert_relative_eq!(x[0], 1.0, epsilon = 1E-6);
    assert_relative_eq!(x[1], 3.0, epsilon = 1E-6);
}
