use vec3::*;

pub use rand::distributions::{Distribution, Uniform};
pub use rand::SeedableRng;
pub use rand::Rng as RngTrait;
pub use rand::RngCore;
pub use rand::seq::SliceRandom;
pub use rand_chacha::ChaChaRng;
pub use rand_xorshift::XorShiftRng;

pub fn rand_in_ball(rng : &mut RngCore) -> Vec3 {
    loop {
        let p = 2.0 * rng.gen::<Vec3>() - vec3(1.0, 1.0, 1.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}

pub fn rand_in_disk(rng : &mut RngCore) -> Vec3 {
    loop {
        let p = ivec3(2, 2, 0) * rng.gen::<Vec3>() - vec3(1.0, 1.0, 0.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}
