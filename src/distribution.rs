use random::*;
use vec3::*;
use mat3::*;
use std::f32::consts::PI;

pub trait Meas<T> {
    fn pdf(&self, x : &T) -> f32;
    fn sample(&self, rng : &mut Rng) -> T;
}

pub fn random_cosine_direction(rng : &mut Rng) -> Vec3 {
    let r1 : f32 = rng.gen();
    let r2 : f32 = rng.gen();
    let z = (1.0 - r2).sqrt();
    let phi = 2.0 * PI * r1;
    let x = phi.cos() * r2.sqrt();
    let y = phi.sin() * r2.sqrt();
    vec3(x, y, z)
}

pub struct CosineDist(Mat3);

impl CosineDist {
    pub fn new(w : Vec3) -> Self {
        CosineDist(onb_from_w(w))
    }
}

impl Meas<Vec3> for CosineDist {
    fn pdf(&self, x : &Vec3) -> f32 {
        let cosine = dot(x.unit(), self.0 * ivec3(0, 0, 1));
        cosine.max(0.0) / PI
    }

    fn sample(&self, rng : &mut Rng) -> Vec3 {
        self.0 * random_cosine_direction(rng)
    }
}

pub struct MZero;

impl<T> Meas<T> for MZero {
    fn pdf(&self, x : &T) -> f32 {
        0.0
    }
    fn sample(&self, rng : &mut Rng) -> T {
        panic!("can't sample from mzero");
    }
}
