use vec3::*;

use rand;
pub use rand::distributions::{IndependentSample, Range};
pub use rand::SeedableRng;
pub use rand::Rng as RngTrait;

pub type RngSeed = [u32; 4];
pub type Rng = rand::XorShiftRng;

pub fn rand_in_ball(rng : &mut Rng) -> Vec3 {
    loop {
        let p = 2.0 * rng.gen::<Vec3>() - vec3(1.0, 1.0, 1.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}

pub fn rand_in_disk(rng : &mut Rng) -> Vec3 {
    loop {
        let p = ivec3(2, 2, 0) * rng.gen::<Vec3>() - vec3(1.0, 1.0, 0.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}
