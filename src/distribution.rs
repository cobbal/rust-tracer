use random::*;
use vec3::*;
use mat3::*;
use std::f32::consts::PI;
use std::fmt::*;

// traits

#[derive(Debug)]
pub struct Weighted<T>(pub T, pub f32);

pub trait MassAndSample<T> {
    fn total_mass(&self) -> f32;
    fn sample(&self, rng : &mut Rng) -> Weighted<T>;
}

pub trait Meas<T> : MassAndSample<T> + Debug {
    fn pdf(&self, x : &T) -> f32;
}

macro_rules! items {
    ($($item:item)*) => ($($item)*);
}

macro_rules! mimpl {
    {[$($TVs:tt)*] Meas[$MT:tt] for $T:ty {
        total_mass(&$self0:tt) $body0:tt
        sample(&$self1:tt, $rng:ident) $body1:tt
        pdf(&$self2:tt, $x:ident) $body2:tt
    }} => {
        items! {
            impl<$($TVs)*> MassAndSample<$MT> for $T {
                fn total_mass(&$self0) -> f32 $body0
                fn sample(&$self1, $rng : &mut Rng) -> Weighted<$MT> $body1
            }
            impl<$($TVs)*> Meas<$MT> for $T {
                fn pdf(&$self2, $x : &$MT) -> f32 $body2
            }
        }
    };
}

// useful measures

pub fn random_cosine_direction(rng : &mut Rng) -> Vec3 {
    let r1 : f32 = rng.gen();
    let r2 : f32 = rng.gen();
    let z = (1.0 - r2).sqrt();
    let phi = 2.0 * PI * r1;
    let x = phi.cos() * r2.sqrt();
    let y = phi.sin() * r2.sqrt();
    vec3(x, y, z)
}

#[derive(Clone, Debug)]
pub struct CosineDist(Mat3);

impl CosineDist {
    pub fn new(w : Vec3) -> Self {
        CosineDist(onb_from_w(w))
    }
}


mimpl!{[] Meas[Vec3] for CosineDist {
    total_mass(&self) { 1.0 }

    sample(&self, rng) {
        Weighted(self.0 * random_cosine_direction(rng), 1.0)
    }

    pdf(&self, x) {
        let cosine = dot(x.unit(), self.0 * ivec3(0, 0, 1));
        cosine.max(0.0) / PI
    }

}}

#[derive(Clone, Debug)]
pub struct HemisphereDist(pub Vec3);

mimpl!{[] Meas[Vec3] for HemisphereDist {
    total_mass(&self) { 1.0 }

    sample(&self, rng) {
        let mut v = rand_in_ball(rng).unit();
        if dot(v, self.0) < 0.0 {
            v = -v;
        }
        Weighted(v, 1.0)
    }

    pdf(&self, x) {
        if dot(*x, self.0) > 0.0 {
            1.0 / (2.0 * PI)
        } else {
            0.0
        }
    }
}}


#[derive(Clone, Debug)]
pub struct MZero;
mimpl!{[T] Meas[T] for MZero {
    total_mass(&self) { 0.0 }
    sample(&self, rng) {
        panic!("can't sample from MZero");
    }
    pdf(&self, x) { 0.0 }
}}

#[derive(Clone, Debug)]
pub struct Dirac<T>(pub T);
mimpl!{[T : Clone + Debug] Meas[T] for Dirac<T> {
    total_mass(&self) { 1.0 }
    sample(&self, rng) { Weighted(self.0.clone(), 1.0) }
    pdf(&self, x) { 0.0 /* HACK */ }
}}

#[derive(Clone, Debug)]
pub struct MPlus<A, B> {
    pub a : A,
    pub b : B,
    pub sampling_bias : f32
}

impl<A, B> MPlus<A, B> {
    pub fn new(a : A, b : B, sampling_bias : f32) -> Self {
        MPlus {a: a, b: b, sampling_bias: sampling_bias}
    }
}

impl<T, A, B> MassAndSample<T> for MPlus<A, B> where A : MassAndSample<T>, B : MassAndSample<T> {
    fn total_mass(&self) -> f32 {
        self.a.total_mass() + self.b.total_mass()
    }

    fn sample(&self, rng : &mut Rng) -> Weighted<T> {

        if rng.gen::<f32>() < self.sampling_bias {
            let Weighted(x, p) = self.a.sample(rng);
            Weighted(x, p / self.sampling_bias)
        } else {
            let Weighted(x, p) = self.b.sample(rng);
            Weighted(x, p / (1.0 - self.sampling_bias))
        }
    }
}

impl <T, A, B> Meas<T> for MPlus<A, B> where A : Meas<T>, B : Meas<T> {
    fn pdf(&self, x : &T) -> f32 {
        self.a.pdf(x) + self.b.pdf(x)
    }
}

impl<T, A> MassAndSample<T> for Weighted<A> where A : MassAndSample<T> {
    fn total_mass(&self) -> f32 {
        self.0.total_mass() * self.1
    }

    fn sample(&self, rng : &mut Rng) -> Weighted<T> {
        let mut res = self.0.sample(rng);
        res.1 *= self.1;
        res
    }
}

impl<T, A> Meas<T> for Weighted<A> where A : Meas<T> {
    fn pdf(&self, x : &T) -> f32 {
        self.0.pdf(x) * self.1
    }
}

// impl<T, A> MassAndSample<T> for Box<A> where A : MassAndSample<T> {
//     fn total_mass(&self) -> f32 { (*self).total_mass() }
//     fn sample(&self, rng : &mut Rng) -> Weighted<T> { (*self).sample(rng) }
// }
