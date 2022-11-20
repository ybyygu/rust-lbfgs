//! Backend for lbfgs vector operations

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
}
