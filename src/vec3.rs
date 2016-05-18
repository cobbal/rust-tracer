use std::ops::*;
use std::fmt;

#[derive(Clone)]
pub struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }

}

pub fn vec3(x : f32, y : f32, z : f32) -> Vec3 {
    return Vec3{ x: x, y: y, z: z};
}

pub fn dot<'a>(a : &'a Vec3, b : &'a Vec3) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

pub fn cross<'a>(a : &'a Vec3, b : &'a Vec3) -> Vec3 {
    return vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

impl Vec3 {
    pub fn len<'a>(&'a self) -> f32 {
        return dot(self, self).sqrt();
    }

    pub fn unit(&self) -> Vec3 {
        let l = self.len();
        return self / l;
    }

    pub fn fmap_mut(&mut self, f : &Fn(f32) -> f32) {
        self.x = f(self.x);
        self.y = f(self.y);
        self.z = f(self.z);
    }

    pub fn fmap<F : Fn(f32) -> f32>(&self, f : F) -> Vec3 {
        return vec3(f(self.x), f(self.y), f(self.z));
    }
}

macro_rules! elemOp {
    ($T:ident, $p:ident, $t0:ty, $t1:ty) => (impl<'a, 'b> $T<$t1> for $t0 {
        type Output = Vec3;
        fn $p(self, other : $t1) -> Vec3 {
            return Vec3 { x: self.x.$p(other.x),
                          y: self.y.$p(other.y),
                          z: self.z.$p(other.z) };
        }
    });
}

macro_rules! allElemOp {
    ($T:ident, $p:ident) => {
        elemOp!($T, $p, Vec3, Vec3);
        elemOp!($T, $p, &'a Vec3, Vec3);
        elemOp!($T, $p, Vec3, &'b Vec3);
        elemOp!($T, $p, &'a Vec3, &'b Vec3);
    };
}

macro_rules! elemAssign {
    ($T:ident, $p:ident, $t:ty) => (impl<'a> $T<$t> for Vec3 {
        fn $p(&mut self, other : $t) {
            self.x.$p(other.x);
            self.y.$p(other.y);
            self.z.$p(other.z);
        }
    });
}

macro_rules! allElemAssign {
    ($T:ident, $p:ident) => {
        elemAssign!($T, $p, Vec3);
        elemAssign!($T, $p, &'a Vec3);
    };
}

allElemOp!(Add, add);
allElemOp!(Sub, sub);
allElemOp!(Mul, mul);
allElemAssign!(AddAssign, add_assign);
allElemAssign!(SubAssign, sub_assign);
allElemAssign!(MulAssign, mul_assign);

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other : f32) -> Vec3 {
        return Vec3 { x: self.x * other,
                      y: self.y * other,
                      z: self.z * other };
    }
}

impl<'a> Mul<f32> for &'a Vec3 {
    type Output = Vec3;
    fn mul(self, other : f32) -> Vec3 {
        return Vec3 { x: self.x * other,
                      y: self.y * other,
                      z: self.z * other };
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other : f32) -> Vec3 {
        return self * (1.0 / other);
    }
}

impl<'a> Div<f32> for &'a Vec3 {
    type Output = Vec3;
    fn div(self, other : f32) -> Vec3 {
        return self * (1.0 / other);
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, other : f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}


impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other : Vec3) -> Vec3 {
        return Vec3 { x: self * other.x,
                      y: self * other.y,
                      z: self * other.z };
    }
}

impl<'a> Mul<&'a Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other : &'a Vec3) -> Vec3 {
        return Vec3 { x: self * other.x,
                      y: self * other.y,
                      z: self * other.z };
    }
}

impl<'a> Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        return Vec3 { x: -self.x,
                      y: -self.y,
                      z: -self.z };
    }
}

impl<'a> Neg for &'a Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        return Vec3 { x: -self.x,
                      y: -self.y,
                      z: -self.z };
    }
}

pub enum Vec3Index { X, Y, Z, R, G, B }

impl Index<Vec3Index> for Vec3 {
    type Output = f32;
    fn index<'a>(&'a self, i : Vec3Index) -> &'a f32 {
        return match i {
            Vec3Index::X => &self.x, Vec3Index::R => &self.x,
            Vec3Index::Y => &self.y, Vec3Index::G => &self.y,
            Vec3Index::Z => &self.z, Vec3Index::B => &self.z,
        }
    }
}


impl IndexMut<Vec3Index> for Vec3 {
    fn index_mut<'a>(&'a mut self, i : Vec3Index) -> &'a mut f32 {
        return match i {
            Vec3Index::X => &mut self.x, Vec3Index::R => &mut self.x,
            Vec3Index::Y => &mut self.y, Vec3Index::G => &mut self.y,
            Vec3Index::Z => &mut self.z, Vec3Index::B => &mut self.z,
        }
    }
}
