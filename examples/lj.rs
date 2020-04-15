// [[file:~/Workspace/Programming/gosh-rs/lbfgs/lbfgs.note::*lj.rs][lj.rs:1]]
use vecfx::*;

#[derive(Clone, Copy, Debug)]
pub struct LennardJones {
    /// Energy constant of the Lennard-Jones potential
    pub epsilon: f64,
    /// Distance constant of the Lennard-Jones potential
    pub sigma: f64,
}

impl Default for LennardJones {
    fn default() -> Self {
        LennardJones {
            epsilon: 1.0,
            sigma: 1.0,
        }
    }
}

impl LennardJones {
    // vij
    fn pair_energy(&self, r: f64) -> f64 {
        let s6 = f64::powi(self.sigma / r, 6);
        4.0 * self.epsilon * (f64::powi(s6, 2) - s6)
    }

    // dvij
    fn pair_gradient(&self, r: f64) -> f64 {
        let s6 = f64::powi(self.sigma / r, 6);

        24.0 * self.epsilon * (s6 - 2.0 * f64::powi(s6, 2)) / r
    }

    /// Evaluate energy and forces
    pub fn evaluate(&self, positions: &[[f64; 3]], forces: &mut [[f64; 3]]) -> f64 {
        let n = positions.len();
        debug_assert_eq!(n, forces.len(), "positions.len() != forces.len()");

        // initialize with zeros
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..3 {
                forces[i][j] = 0.0;
            }
        }

        // calculate energy and forces
        for i in 0..n {
            for j in 0..i {
                let r = positions[i].vecdist(&positions[j]);
                energy += self.pair_energy(r);

                let g = self.pair_gradient(r);
                for k in 0..3 {
                    let dr = positions[j][k] - positions[i][k];
                    forces[i][k] += 1.0 * g * dr / r;
                    forces[j][k] += -1.0 * g * dr / r;
                }
            }
        }

        energy
    }
}

fn main() {
    use liblbfgs::{lbfgs, Progress};
    let mut lj = LennardJones::default();

    // LJ38
    let mut positions = [
    50.27754123,     50.04898929,     50.13164926,
    49.54021264,     50.20208324,     49.33142540,
    50.36795885,     50.91366213,     49.53932153,
    49.71004612,     49.12254218,     50.30950773,
    51.59992702,     50.93225616,     49.75465016,
    49.54096326,     50.01499737,     50.89785480,
    49.27589571,     49.40276757,     51.78288271,
    50.84293978,     51.39416756,     50.37578558,
    50.85011555,     49.18468518,     49.88733738,
    50.48954837,     48.31866393,     50.39752223,
    49.54570033,     50.99619360,     50.39370291,
    50.83198382,     49.97838559,     49.15111253,
    48.68259959,     51.76333286,     50.71665986,
    48.78703201,     50.74137148,     48.70964574,
    50.46846546,     50.64822201,     51.05913458,
    50.32617974,     51.72202742,     51.25600177,
    48.98719425,     48.57072315,     50.86630469,
    49.41532317,     48.01960200,     49.86479222,
    50.33160898,     51.97929717,     49.55667898,
    50.05747126,     48.65423592,     51.34871762,
    49.47506697,     52.40222682,     50.09998700,
    50.08941533,     50.76191893,     48.51033008,
    49.66096522,     52.51096459,     51.72590563,
    49.26806275,     51.61276429,     49.24438196,
    51.33222714,     48.66903395,     51.03264292,
    48.82443599,     48.10619401,     51.83279474,
    49.32661434,     51.45251656,     51.52447147,
    48.60626152,     50.67040668,     51.00622736,
    48.66876297,     50.02851267,     49.95976353,
    48.45421774,     49.58111803,     50.93806524,
    49.23169585,     49.11782673,     49.33972073,
    50.51551634,     47.49882067,     49.67074204,
    50.53443706,     49.65001264,     51.43455000,
    51.64327906,     48.46431354,     49.94896031,
    51.54629763,     49.42747135,     51.83066872,
    49.51504504,     50.47172337,     51.84485663,
    49.10265035,     52.00399992,     52.61150826,
    49.66462610,     47.60012985,     50.93620680];

    lbfgs().minimize(
        &mut positions,
        |x: &[f64], gx: &mut [f64]| {
            let energy = lj.evaluate(x.as_3d(), gx.as_mut_3d());
            gx.vecscale(-1.0);
            Ok(energy)
        },
        |prgr| {
            println!("Iteration {}, Evaluation: {}", &prgr.niter, &prgr.neval);
            println!(
                "  xnorm = {}, gnorm = {}, step = {}",
                &prgr.xnorm, &prgr.gnorm, &prgr.step
            );
            println!("");
            false
        },
    );
}
// lj.rs:1 ends here
