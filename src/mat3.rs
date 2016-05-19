use std::ops::*;

use vec3::*;

#[derive(Copy, Clone)]
pub struct Mat3(pub [f32; 9]);

impl Mat3 {
    pub fn transpose(self) -> Mat3 {
        let mut result = MAT_ZERO3;
        for i in 0..3 {
            for j in 0..3 {
                result[(i, j)] = self[(j, i)];
            }
        }
        result
    }
}

pub const MAT_IDENT3 : Mat3 = Mat3([
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
]);

pub const MAT_ZERO3 : Mat3 = Mat3([0.0; 9]);

impl Index<(usize, usize)> for Mat3 {
    type Output = f32;
    fn index<'a>(&'a self, ij : (usize, usize)) -> &'a f32 {
        let Mat3(ref elems) = *self;
        &elems[(ij.0 * 3 + ij.1) as usize]
    }
}

impl IndexMut<(usize, usize)> for Mat3 {
    fn index_mut<'a>(&'a mut self, ij : (usize, usize)) -> &'a mut f32 {
        let &mut Mat3(ref mut elems) = self;
        &mut elems[(ij.0 * 3 + ij.1) as usize]
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;
    fn mul(self, other : Mat3) -> Mat3 {
        let mut result = MAT_ZERO3;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[(i, j)] += self[(i, k)] * other[(k, j)];
                }
            }
        }
        result
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, other : Vec3) -> Vec3 {
        let mut result = ZERO3;
        for i in 0..3 {
            for k in 0..3 {
                result[i] += self[(i, k)] * other[k];
            }
        }
        result
    }
}

impl Mul<f32> for Mat3 {
    type Output = Mat3;
    fn mul(self, other : f32) -> Mat3 {
        let mut res = MAT_ZERO3;
        for i in 0..3 {
            for j in 0..3 {
                res[(i, j)] = self[(i, j)] * other
            }
        }
        res
    }
}

impl Add<Mat3> for Mat3 {
    type Output = Mat3;
    fn add(self, other : Mat3) -> Mat3 {
        let mut res = MAT_ZERO3;
        for i in 0..3 {
            for j in 0..3 {
                res[(i, j)] = self[(i, j)] + other[(i, j)]
            }
        }
        res
    }
}
