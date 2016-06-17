use ray::*;
use object::*;
use texture::*;
use distribution::*;
use utils::*;
use vec3::*;

use std::sync::Arc;
use std::f32::consts::PI;

pub struct ScatterMeas {
    pub alb : Vec3,
    pub origin : Vec3,
    pub out_dir : Box<Meas<Vec3>>,
}

pub enum MaterialInteraction {
    Scatter(ScatterMeas),
    Emit(Color),
    Absorb,
}
pub use self::MaterialInteraction::*;

pub trait Material : Send + Sync {
    fn scatter(&self, r_in : &Ray, hit : &HitRecord) -> MaterialInteraction;
    fn radiosity(&self) -> Option<Vec3> { None }
    fn emit(&self, normal : Vec3) -> Option<Box<Meas<Vec3>>> { None }
}

pub struct Lambertian<T : Texture>(pub T);

impl<T : Texture> Lambertian<T> {
    fn scatter_pdf(&self, r_in : &Ray, hit : &HitRecord, scattered : &Ray) -> f32 {
        let cosine = dot(hit.normal, scattered.direction.unit());
        cosine.max(0.0) / PI
    }
}

impl<T : Texture> Material for Lambertian<T> {
    fn scatter(
        &self, r_in : &Ray, hit : &HitRecord
    ) -> MaterialInteraction {

        return Scatter(ScatterMeas {
            alb: self.0.tex_lookup(hit.uv, &hit.p),
            origin: hit.p,
            out_dir: box CosineDist::new(hit.normal),
        });
    }
}

// struct LambertianScatterMeas {
//     normal : Vec3,
// }

// mimpl!{[] Meas[Vec3] for LambertianScatterMeas {
//     total_mass(&self) { 1.0 }

//     sample(&self, rng) {
//         let w = self.normal;
//         let uvw = onb_from_w(w);
//         Weighted(uvw * random_cosine_direction(rng), 1.0)
//     }

//     pdf(&self, x) {
//         let cosine = dot(self.normal, x.unit());
//         cosine.max(0.0) / PI
//     }
// }}

// pub struct Metal {
//     pub albedo : Vec3,
//     pub fuzz : f32,
// }

// impl Material for Metal {
//     fn scatter(&self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord)
//                -> Option<(Ray, Vec3)> {
//         let dir = r_in.direction.unit();
//         let scattered = reflect(dir, hit.normal) + self.fuzz * rand_in_ball(rng);
//         if dot(scattered, hit.normal) <= 0.0 {
//             return None
//         }
//         return Some((ray(hit.p, scattered, r_in.time),
//                      self.albedo));
//     }
// }

// pub struct Dielectric(pub f32);

// impl Material for Dielectric {
//     fn scatter(
//         &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
//     ) -> Option<(Ray, Vec3)> {
//         let &Dielectric(ref_idx) = self;

//         let dir = r_in.direction.unit();
//         let reflected = reflect(dir, hit.normal);
//         let attenuation = vec3(1.0, 1.0, 1.0);

//         let (outward_normal, ni_over_nt, cosine) =
//             if dot(dir, hit.normal) > 0.0 {
//                 (-1.0 * &hit.normal,
//                  ref_idx,
//                  ref_idx * dot(dir, hit.normal))
//             } else {
//                 (1.0 * &hit.normal,
//                  1.0 / ref_idx,
//                  -dot(dir, hit.normal))
//             };

//         let out_dir =
//             match refract(r_in.direction, outward_normal, ni_over_nt) {
//                 Some(refracted) => {
//                     let reflect_prob = schlick(cosine, ref_idx);
//                     if rng.gen::<f32>() < reflect_prob {
//                         reflected
//                     } else {
//                         refracted
//                     }
//                 },
//                 None => reflected
//             };
//         return Some((ray(hit.p, out_dir, r_in.time), attenuation));

//     }
// }

pub struct DiffuseLight<T>(pub Arc<T>);

impl Material for DiffuseLight<Vec3> {
    fn scatter(&self, r_in : &Ray, hit : &HitRecord) -> MaterialInteraction {
        Emit(*self.0)
    }

    fn radiosity(&self) -> Option<Vec3> {
        Some(*self.0 * PI)
    }

    fn emit(&self, normal : Vec3) -> Option<Box<Meas<Vec3>>> {
        Some(box CosineDist::new(normal))
    }
}

// pub fn reflect(v : Vec3, n : Vec3) -> Vec3 {
//     return v - 2.0 * dot(v, n) * n;
// }

// pub fn refract(v : Vec3, n : Vec3, ni_over_nt : f32) -> Option<Vec3> {
//     let uv = v.unit();
//     let dt = dot(uv, n);
//     let disc = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
//     if disc > 0.0 {
//         return Some(ni_over_nt * (uv - n * dt) - n * disc.sqrt());
//     } else {
//         return None;
//     }
// }

// pub fn schlick(cosine : f32, ref_idx : f32) -> f32 {
//     let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
//     r0 = r0 * r0;
//     return r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0);
// }

// pub struct MixtureMaterial<M1 : Material, M2 : Material> {
//     pub p : f32,
//     pub m1 : M1,
//     pub m2 : M2,
// }
// impl<M1 : Material, M2 : Material> Material for MixtureMaterial<M1, M2> {
//     fn scatter(
//         &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
//     ) -> Option<(Ray, Vec3)> {
//         let m : &Material = if rng.gen::<f32>() < self.p {
//             &self.m1
//         } else {
//             &self.m2
//         };
//         m.scatter(rng, r_in, hit)
//     }
// }

// pub struct Isotropic<T : Texture>(pub T);
// impl<T : Texture> Material for Isotropic<T> {
//     fn scatter(
//         &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
//     ) -> Option<(Ray, Vec3)> {
//         let Isotropic(ref albedo) = *self;
//         let mut r = (*r_in).clone();
//         r.origin = hit.p;
//         r.direction = rand_in_ball(rng).unit();
//         Some((r, albedo.tex_lookup(hit.uv, &hit.p)))
//     }
// }

// pub struct Tinted<T : Texture>(pub T);
// impl<T : Texture> Material for Tinted<T> {
//     fn scatter(
//         &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
//     ) -> Option<(Ray, Vec3)> {
//         let Tinted(ref tint) = *self;
//         let mut r = (*r_in).clone();
//         r.origin = hit.p;
//         Some((r, tint.tex_lookup(hit.uv, &hit.p)))
//     }
// }
