#![feature(associated_consts)]
#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(const_fn)]
#![feature(custom_derive)]
#![allow(unused_variables)]

mod vec3;
use vec3::*;
use vec3::Vec3Index::*;

mod mat3;
use mat3::*;

use std::fs::File;
use std::path::Path;
use std::io;
use std::io::Write;
use std::cmp::{max, min, Ordering};
use std::f32;
use std::f32::consts::PI;
use std::rc::Rc;

extern crate image;
extern crate rand;

use rand::distributions::{IndependentSample, Range};
use rand::Rng as RngTrait;
type Rng = rand::XorShiftRng;

fn rand_in_ball(rng : &mut Rng) -> Vec3 {
    loop {
        let p = 2.0 * vec3(rng.gen(), rng.gen(), rng.gen()) - vec3(1.0, 1.0, 1.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}

fn rand_in_disk(rng : &mut Rng) -> Vec3 {
    loop {
        let p = 2.0 * vec3(rng.gen(), rng.gen(), 0.0) - vec3(1.0, 1.0, 0.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}

fn write_buffer(nx : u32, ny : u32, ns : u32, flimg : &mut [&mut [[f64; 3]]]) {
    let mut img = image::DynamicImage::new_rgb8(nx, ny);

    for y in 0..ny {
        for x in 0..nx {
            let mut col = flimg[y as usize][x as usize].clone();
            col[0] = (col[0] / ns as f64).max(0.0).min(1.0).sqrt();
            col[1] = (col[1] / ns as f64).max(0.0).min(1.0).sqrt();
            col[2] = (col[2] / ns as f64).max(0.0).min(1.0).sqrt();

            let ir = (255.99 * col[0]) as u8;
            let ig = (255.99 * col[1]) as u8;
            let ib = (255.99 * col[2]) as u8;
            img.as_mut_rgb8().unwrap()[(x, y)] = image::Rgb([ir, ig, ib])
        }
    }
    let ref mut fout = File::create(&Path::new("trace.png")).unwrap();
    let _ = img.save(fout, image::PNG);
}

#[allow(dead_code)]
fn random_scene(rng : &mut Rng) -> Box<Hitable> {
    let mut list : Vec<Box<Hitable>> = Vec::new();

    let chexture = CheckerTex {
        even: ConstantTex(vec3(0.2, 0.3, 0.1)),
        odd: ConstantTex(vec3(0.9, 0.9, 0.9)),
    };
    list.push(sphere(1000.0, vec3(0.0, -1000.0, 0.0),
                     Rc::new(Lambertian(chexture))));

    for a in -11..11 {
        for b in -11..11 {
            if rng.gen::<f32>() < 0.0 {
                continue;
            }

            let choose_mat = rng.gen::<f32>();
            let center = vec3(a as f32 + 0.9 * rng.gen::<f32>(),
                              0.2,
                              b as f32 + 0.9 * rng.gen::<f32>());
            if (&center - vec3(4.0, 0.2, 0.0)).len() > 0.9 {
                let sph =
                    if choose_mat < 0.8 { // diffuse
                        let col = vec3(rng.gen(), rng.gen(), rng.gen())
                            * vec3(rng.gen(), rng.gen(), rng.gen());
                        moving_sphere(0.2,
                               center, center + vec3(0.0, 0.5 * rng.gen::<f32>(), 0.0),
                               0.0, 1.0,
                               Rc::new(Lambertian(ConstantTex(col))))
                    } else if choose_mat < 0.95 { //metal
                        let col = (vec3(1.0, 1.0, 1.0)
                                   + vec3(rng.gen(), rng.gen(), rng.gen())) / 2.0;
                        sphere(0.2, center,
                                          Rc::new(Metal{albedo: col, fuzz: 0.5 * rng.gen::<f32>()}))
                    } else {
                        sphere(0.2, center,
                                          Rc::new(Dielectric(1.5)))
                    };
                list.push(sph);
            }
        }
    }

    list.push(sphere(
        1.0, ivec3(0, 1, 0),
        Rc::new(Dielectric(1.5))));
    list.push(sphere(
        1.0, ivec3(-4, 1, 0),
        Rc::new(Lambertian(ConstantTex(vec3(0.4, 0.2, 0.1))))));
    list.push(sphere(
        1.0, ivec3(4, 1, 0),
        Rc::new(Metal{albedo: vec3(0.7, 0.6, 0.5),
                      fuzz: 0.0})));

    return into_bvh(rng, list, (0.0, 1.0));
}

fn main() {


    let x = AABB{min: ZERO3, max: ZERO3};
    let r = ray(ZERO3, ZERO3, 0.0);
    x.hit(&r, (0.0, 1.0));


    // rand::weak_rng();
    let seed = rand::thread_rng().gen::<[u32; 4]>();
    println!("let seed = {:?};", seed);
    let mut rng : Rng = rand::SeedableRng::from_seed(seed);

    let nx = 500;
    let ny = 500;
    let ns = 100000;

    // let world = random_scene(&mut rng);
    // let world = simple_light(&mut rng);
    // let world = cornell_box(&mut rng);
    let world = the_next_week(&mut rng);

    let lookfrom = ivec3(278, 278, -800);
    let lookfrom = ivec3(478, 278, -600);
    let lookat = ivec3(278, 278, 0);
    let dist_to_focus = 10.0;
    let aperture = 0.0;
    let vfov = 40.0;

    let cam = camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0),
                     vfov, nx as f32 / ny as f32,
                     aperture, dist_to_focus,
                     0.0, 1.0);


    let log_tick_maybe = |s : i32| -> bool {
        let pcent = |n : i32| (100 * n / ns as i32);
        let pc_last = pcent(s - 1);
        let pc_now = pcent(s);
        // if pc_now % 1 == 0 && pc_now != pc_last {
        if true {
            print!("{} ", s);
            io::stdout().flush().unwrap();
            return true;
        }
        return false;
    };

    // http://stackoverflow.com/a/36376568/73681
    let mut flimg_raw = vec![[0.0; 3]; (nx * ny) as usize];
    let mut grid_base : Vec<_> =
        flimg_raw.as_mut_slice().chunks_mut(nx as usize).collect();
    let mut flimg : &mut [&mut [[f64; 3]]] = grid_base.as_mut_slice();

    for s in 0..ns {
        if log_tick_maybe(s as i32) {
            write_buffer(nx, ny, s, flimg);
        }

        for y in 0..ny {
            for x in 0..nx {

                let u = (x as f32 + rng.gen::<f32>()) / nx as f32;
                let v = ((ny - y - 1) as f32 + rng.gen::<f32>()) / ny as f32;

                let r = cam.get_ray(&mut rng, u, v);

                let col = color(&mut rng, r, &*world);
                flimg[y as usize][x as usize][0] += col[R] as f64;
                flimg[y as usize][x as usize][1] += col[G] as f64;
                flimg[y as usize][x as usize][2] += col[B] as f64;
            }
        }
    }
    write_buffer(nx, ny, ns, flimg);
}

#[derive(Clone,Debug)]
struct Ray {
    origin : Vec3,
    direction : Vec3,
    time : f32,
}

fn ray(origin : Vec3, direction : Vec3, time : f32) -> Ray {
    return Ray { origin: origin, direction: direction, time: time };
}

impl Ray {
    fn at(&self, t : f32) -> Vec3 {
        return &self.origin + &(t * &self.direction);
    }
}

#[inline(never)]
fn color(rng : &mut Rng, r0 : Ray, world : &Hitable) -> Vec3 {
    let mut accumulator = vec3(0.0, 0.0, 0.0);
    let mut attenuation = vec3(1.0, 1.0, 1.0);
    let mut r = r0;
    for ttl in 0..50 {
        match world.hit(rng, &r, (0.001, f32::INFINITY)) {
            Some(hit) => {
                let emitted = hit.material.emitted(hit.uv, &hit.p);
                accumulator += emitted * attenuation;
                match hit.material.scatter(rng, &r, &hit) {
                    Some((scattered, local_att)) => {
                        r = scattered;
                        attenuation *= local_att;
                        continue;
                    },
                    None => {
                        break;
                    }
                }
            },
            None => {
                break;
                // let unit_direction = r.direction.unit();
                // let t = 0.5 * unit_direction[Y] + 1.0;
                // accumulator *= (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
                // break;
            }
        }
        panic!()
    }
    return accumulator;
}

struct HitRecord<'a> {
    t : f32,
    p : Vec3,
    normal : Vec3,
    material : &'a Material,
    uv : (f32, f32),
}

trait Hitable {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord>;
    fn bounding_box(&self, time : (f32, f32)) -> AABB;
}

#[derive(Clone)]
struct Sphere {
    material : Rc<Material>,
    center0 : Vec3, dcenter : Vec3,
    radius : f32,
}


impl Sphere {
    fn center(&self, time : f32) -> Vec3 {
        return &self.center0 + time * &self.dcenter
    }
}


fn moving_sphere(radius : f32,
          center0 : Vec3, center1 : Vec3,
          time0 : f32, time1 : f32,
          material : Rc<Material>,
) -> Box<Sphere> {
    let dcenter = (&center1 - &center0) / (&time1 - &time0);
    box Sphere {
        material: material,
        center0: &center0 - time0 * &dcenter,
        dcenter: dcenter,
        radius: radius,
    }
}

fn sphere(
    radius : f32, center : Vec3, material : Rc<Material>
) -> Box<Sphere> {
    moving_sphere(radius, center, center, 0.0, 1.0, material)
}

impl Hitable for Sphere {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
        let center = self.center(r.time);
        let oc = &r.origin - &center;
        let a = dot(r.direction, r.direction);
        let b = dot(oc, r.direction);
        let c = dot(oc, oc) - self.radius * self.radius;
        let disc = b * b - a * c;
        if disc > 0.0 {
            let t0 = (-b - disc.sqrt()) / a;
            let t1 = (-b + disc.sqrt()) / a;
            for t in &[t0, t1] {
                if dist.0 <= *t && *t < dist.1 {
                    let p = r.at(*t);
                    let normal = (&p - &center) / self.radius;

                    let phi = normal[Z].atan2(normal[X]);
                    let theta = normal[Y].asin();

                    let u = 1.0 - (phi + PI) / (2.0 * PI);
                    let v = (theta + PI / 2.0) / PI;

                    return Some(HitRecord {
                        t: *t, p: p, normal: normal, material: &*self.material,
                        uv: (u, v),
                    });
                }
            }
        }
        return None
    }


    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        let r = self.radius;
        return AABB {
            min: self.center0 - vec3(r, r, r),
            max: self.center0 + vec3(r, r, r),
        };
    }
}
#[allow(dead_code)]
struct Camera {
    origin : Vec3,
    lower_left_corner : Vec3,
    horizontal : Vec3,
    vertical : Vec3,
    u : Vec3, v : Vec3, w : Vec3,
    lens_radius : f32,
    time0 : f32,
    time1 : f32,
}

fn camera(lookfrom : Vec3, lookat : Vec3, vup : Vec3,
          vfov : f32, aspect : f32, aperture : f32, focus_dist : f32,
          time0 : f32, time1 : f32)
    -> Camera {
    // vfov is top to bottom in degrees

    // let focus_dist = (&lookat - &lookfrom).len();

    let theta = vfov.to_radians();
    let half_height = (theta / 2.0).tan();
    let half_width = aspect * half_height;

    let w = (&lookfrom - &lookat).unit();
    let u = cross(&vup, &w).unit();
    let v = cross(&w, &u);

    return Camera {
        origin: lookfrom,
        lower_left_corner: lookfrom
            - half_width * focus_dist * &u
            - half_height * focus_dist * &v
            - focus_dist * &w,
        horizontal: 2.0 * half_width * focus_dist * &u,
        vertical: 2.0 * half_height * focus_dist * &v,
        u: u, v: v, w: w,
        lens_radius: aperture / 2.0,
        time0 : time0,
        time1 : time1,
    };
}


impl Camera {
    fn get_ray(&self, rng : &mut Rng, s : f32, t : f32) -> Ray {
        let rd = self.lens_radius * rand_in_disk(rng);
        let offset = &self.u * rd[X] + &self.v * rd[Y];

        let dir = &self.lower_left_corner
            + s * &self.horizontal
            + t * &self.vertical
            - &self.origin - &offset;
        let time = self.time0 + rng.gen::<f32>() * (self.time1 - self.time0);
        return ray(&self.origin + offset, dir, time);
    }
}

trait Material {
    fn scatter(&self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord)
               -> Option<(Ray, Vec3)>;
    fn emitted(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        ZERO3
    }
}

struct Lambertian<T : Texture>(T);

impl<T : Texture> Material for Lambertian<T> {
    fn scatter(
        &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
    ) -> Option<(Ray, Vec3)> {
        let Lambertian(ref albedo) = *self;
        let target = hit.p + hit.normal + rand_in_ball(rng);
        let direction = target - hit.p;

        return Some((ray(hit.p, direction, r_in.time),
                     albedo.tex_lookup(hit.uv, &hit.p)));
    }
}

fn reflect(v : Vec3, n : Vec3) -> Vec3 {
    return v - 2.0 * dot(v, n) * n;
}

fn refract(v : Vec3, n : Vec3, ni_over_nt : f32) -> Option<Vec3> {
    let uv = v.unit();
    let dt = dot(uv, n);
    let disc = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if disc > 0.0 {
        return Some(ni_over_nt * (uv - n * dt) - n * disc.sqrt());
    } else {
        return None;
    }
}

fn schlick(cosine : f32, ref_idx : f32) -> f32 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0);
}

struct Metal {
    albedo : Vec3,
    fuzz : f32,
}

impl Material for Metal {
    fn scatter(&self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord)
               -> Option<(Ray, Vec3)> {
        let dir = r_in.direction.unit();
        let scattered = reflect(dir, hit.normal) + self.fuzz * rand_in_ball(rng);
        if dot(scattered, hit.normal) <= 0.0 {
            return None
        }
        return Some((ray(hit.p, scattered, r_in.time),
                     self.albedo));
    }
}

struct Dielectric(f32);

impl Material for Dielectric {
    fn scatter(
        &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
    ) -> Option<(Ray, Vec3)> {
        let &Dielectric(ref_idx) = self;

        let dir = r_in.direction.unit();
        let reflected = reflect(dir, hit.normal);
        let attenuation = vec3(1.0, 1.0, 1.0);

        let (outward_normal, ni_over_nt, cosine) =
            if dot(dir, hit.normal) > 0.0 {
                (-1.0 * &hit.normal,
                 ref_idx,
                 ref_idx * dot(dir, hit.normal))
            } else {
                (1.0 * &hit.normal,
                 1.0 / ref_idx,
                 -dot(dir, hit.normal))
            };

        let out_dir =
            match refract(r_in.direction, outward_normal, ni_over_nt) {
                Some(refracted) => {
                    let reflect_prob = schlick(cosine, ref_idx);
                    if rng.gen::<f32>() < reflect_prob {
                        reflected
                    } else {
                        refracted
                    }
                },
                None => reflected
            };
        return Some((ray(hit.p, out_dir, r_in.time), attenuation));

    }
}

#[derive(Clone, Copy)]
struct AABB {
    min : Vec3,
    max : Vec3,
}

impl AABB {
    #[inline(never)]
    fn hit(self, r : &Ray, dist : (f32, f32)) -> bool {
        for d in 0..3 {
            let mut t0 = (self.min[d] - r.origin[d]) / r.direction[d];
            let mut t1 = (self.max[d] - r.origin[d]) / r.direction[d];
            if r.direction[d] < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            let tmin = if t0 > dist.0 { t0 } else { dist.0 };
            let tmax = if t1 < dist.1 { t1 } else { dist.1 };
            if tmax <= tmin {
                return false;
            }
        }
        return true;
    }

    fn union(&self, other : &AABB) -> AABB {
        return AABB {
            min: Vec3::fmap2(f32::min, &self.min, &other.min),
            max: Vec3::fmap2(f32::max, &self.max, &other.max),
        };
    }
}

struct BVHNode {
    left : Box<Hitable>,
    right : Box<Hitable>,
    bbox : AABB,
}

impl Hitable for BVHNode {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
        if self.bbox.hit(r, dist) {
            let l_hit = self.left.hit(rng, r, dist);
            let r_hit = self.right.hit(rng, r, dist);
            return match (l_hit, r_hit) {
                (None, h) => h,
                (h, None) => h,
                (Some(lh), Some(rh)) =>
                    if lh.t < rh.t {
                        Some(lh)
                    } else {
                        Some(rh)
                    }
            }
        } else {
            return None
        }
    }

    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        return self.bbox;
    }
}


fn into_bvh_det(
    rng_fn : &mut FnMut() -> usize,
    // rng : &mut Rng,
    mut v : Vec<Box<Hitable>>, time : (f32, f32)
) -> Box<Hitable> {
    let axis = rng_fn();

    v.sort_by(&|a : &Box<Hitable>, b : &Box<Hitable>| {
        let va = (*a).bounding_box(time).min[axis];
        let vb = (*b).bounding_box(time).min[axis];
        return va.partial_cmp(&vb).unwrap_or(Ordering::Equal);
    });

    let result = match v.len() {
        0 => panic!("can't create empty BVH"),
        1 => v.pop().unwrap(),
        n => {
            let vr = v.drain(n/2..).collect();
            let left = into_bvh_det(rng_fn, v, time);
            let right = into_bvh_det(rng_fn, vr, time);
            let lbb = left.bounding_box(time);
            let rbb = right.bounding_box(time);
            box BVHNode {
                left: left,
                right: right,
                bbox: lbb.union(&rbb)
            }
        }
    };

    return result;
}

fn into_bvh(
    rng : &mut Rng, v : Vec<Box<Hitable>>, time : (f32, f32)
) -> Box<Hitable> {
    let range = Range::new(0, 3);
    into_bvh_det(
        &mut || range.ind_sample(rng),
        v, time)
}


trait Texture {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3;
}

struct ConstantTex(Vec3);

impl Texture for ConstantTex {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        let ConstantTex(v) = *self;
        return v;
    }
}

struct CheckerTex<T0 : Texture, T1 : Texture> {
    odd : T0,
    even : T1,
}

impl<T0 : Texture, T1 : Texture> Texture for CheckerTex<T0, T1> {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        let sines = p.fmap(|a| (10.0 * a).sin());

        if sines[X] * sines[Y] * sines[Z] < 0.0 {
            return self.odd.tex_lookup(uv, p);
        } else {
            return self.even.tex_lookup(uv, p);
        }
    }
}

fn perlin_interp(cube : [[[Vec3; 2]; 2]; 2], uvw : Vec3) -> f32 {
    let hermite = uvw.fmap(&|a| a * a * (3.0 - 2.0 * a));

    let mut accum = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let one = ivec3(1, 1, 1);
                let ijk = uvec3(i, j, k);
                let weight_v = uvw - ijk;
                let vint = ijk * hermite + (one - ijk) * (one - hermite);
                accum += vint[X] * vint[Y] * vint[Z]
                    * dot(cube[i][j][k], weight_v)
            }
        }
    }
    accum
}

struct PerlinNoise {
    n : usize,
    ranvec : Vec<Vec3>,
    perm_x : Vec<usize>,
    perm_y : Vec<usize>,
    perm_z : Vec<usize>,
}

impl PerlinNoise {
    fn new(rng : &mut Rng, n : usize) -> PerlinNoise {
        let mut pn = PerlinNoise {
            n: n,
            ranvec: vec![ZERO3; n],
            perm_x: vec![0; n],
            perm_y: vec![0; n],
            perm_z: vec![0; n],
        };

        for i in 0..n {
            pn.ranvec[i] = (2.0 * vec3(rng.gen(), rng.gen(), rng.gen())
                            - ivec3(1, 1, 1)).unit();
            pn.perm_x[i] = i;
            pn.perm_y[i] = i;
            pn.perm_z[i] = i;
        }

        for a in &mut[&mut pn.perm_x, &mut pn.perm_y, &mut pn.perm_z] {
            for i in (0..n).rev() {
                let j = Range::new(0, i + 1).ind_sample(rng);
                // slow swap here, since it may well alias
                let tmp = a[i];
                a[i] = a[j];
                a[j] = tmp
            }
        }

        pn
    }

    fn noise(&self, p : Vec3) -> f32 {
        fn floor_mod(f : f32, n : usize) -> usize {
            let nf = n as f32;
            // calculate real mod from fake remainder function
            (((f % nf) + nf) % nf) as usize
        }

        let uvw = p.fmap(&|a : f32| a - a.floor());

        let i = floor_mod(p[X], self.n);
        let j = floor_mod(p[Y], self.n);
        let k = floor_mod(p[Z], self.n);

        let mut cube = [[[ZERO3; 2]; 2]; 2];
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    cube[di][dj][dk] =
                        self.ranvec[(self.perm_x[(i + di) % self.n] ^
                                     self.perm_y[(j + dj) % self.n] ^
                                     self.perm_z[(k + dk) % self.n])]
                }
            }
        }

        perlin_interp(cube, uvw)
    }

    fn turb(&self, mut p : Vec3, depth : usize) -> f32 {
        let mut accum = 0.0;
        let mut weight = 1.0;
        for i in 0..depth {
            accum += weight * self.noise(p);
            weight *= 0.5;
            p = p * 2.0;
        }
        accum.abs()
    }
}

#[derive(Clone)]
struct PerlinTexture {
    noise : Rc<PerlinNoise>,
    scale : f32,
}

impl Texture for PerlinTexture {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        // return vec3(1.0, 1.0, 1.0) * 0.5 * (1.0 + self.noise.turb(self.scale * p, 7));
        ONE3 * 0.5 * (1.0 + (self.scale * p[X] + 5.0 * self.noise.turb(*p, 7)).sin())
        //return vec3(1,1,1)*0.5*(1 + sin(scale*p.x() + 5*noise.turb(scale*p))) ;

    }
}

struct ImageTexture {
    im : image::RgbImage,
}

#[allow(dead_code)]
fn image_texture(d_img : &image::DynamicImage) -> ImageTexture {
    ImageTexture {
        im: d_img.to_rgb()
    }
}

impl Texture for ImageTexture {
    fn tex_lookup(&self, mut uv : (f32, f32), p : &Vec3) -> Vec3 {
        let nx = self.im.width();
        let ny = self.im.height();

        uv.1 = 1.0 - uv.1;
        let i = min(max(0, (uv.0 * nx as f32) as i32) as u32, nx - 1);
        let j = min(max(0, (uv.1 * ny as f32) as i32) as u32, ny - 1);
        let pix = self.im.get_pixel(i, j);
        vec3(pix[0] as f32 / 255.0,
             pix[1] as f32 / 255.0,
             pix[2] as f32 / 255.0)
    }
}

struct DiffuseLight<T : Texture>(T);

impl<T : Texture> Material for DiffuseLight<T> {
    fn scatter(
        &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
    ) -> Option<(Ray, Vec3)> {
        None
    }

    fn emitted(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        let DiffuseLight(ref emit) = *self;
        emit.tex_lookup(uv, p)
    }
}

macro_rules! aarect {
    ($XYRect:ident,
     [$x:ident, $X:ident],
     [$y:ident, $Y:ident],
     [$z:ident, $Z:ident]) => (
        struct $XYRect {
            material : Rc<Material>,
            $x : (f32, f32),
            $y : (f32, f32),
            $z : f32,
        }

        impl $XYRect {
            fn new(
                $x : (f32, f32), $y : (f32, f32), $z : f32, mat : &Rc<Material>, flip : bool
            ) -> Box<Hitable> {
                let r = $XYRect {$x: $x, $y: $y, $z: $z, material: mat.clone()};
                if flip { box FlipNormals(r) } else { box r }
            }
        }

        impl Hitable for $XYRect {
            fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
                let t = (self.$z - r.origin[$Z]) / r.direction[$Z];
                if !(dist.0 < t && t < dist.1) {
                    return None;
                }
                let $x = r.origin[$X] + t * r.direction[$X];
                let $y = r.origin[$Y] + t * r.direction[$Y];
                if !(self.$x.0 < $x && $x < self.$x.1 &&
                     self.$y.0 < $y && $y < self.$y.1) {
                    None
                } else {
                    let u = ($x - self.$x.0) / (self.$x.1 - self.$x.0);
                    let v = ($y - self.$y.0) / (self.$y.1 - self.$y.0);
                    let mut p = ZERO3;
                    p[$X] = $x;
                    p[$Y] = $y;
                    p[$Z] = self.$z;
                    let mut normal = ZERO3;
                    normal[$Z] = 1.0;
                    Some(HitRecord {
                        t: t, p: p, normal: normal, material: &*self.material,
                        uv: (u, v)
                    })
                }
            }

            fn bounding_box(&self, time : (f32, f32)) -> AABB {
                let mut min = ZERO3;
                let mut max = ZERO3;
                min[$X] = self.$x.0;
                min[$Y] = self.$y.0;
                min[$Z] = self.$z - 0.001;
                max[$X] = self.$x.1;
                max[$Y] = self.$y.1;
                max[$Z] = self.$z + 0.001;
                AABB{min: min, max: max}
            }
        })
}

aarect!(XYRect, [x, X], [y, Y], [z, Z]);
aarect!(YZRect, [y, Y], [z, Z], [x, X]);
aarect!(XZRect, [x, X], [z, Z], [y, Y]);

#[allow(dead_code)]
fn simple_light(rng : &mut Rng) -> Box<Hitable> {
    let noise = Rc::new(PerlinNoise::new(rng, 256));
    let ntex = PerlinTexture {
        noise: noise,
        scale: 4.0,
    };
    let mut list : Vec<Box<Hitable>> = Vec::new();

    let marble_material = Rc::new(Lambertian(ntex.clone()));

    // floor
    list.push(sphere(
        1000.0, ivec3(0, -1000, 0),
        marble_material.clone(),
    ));

    // marble sphere
    list.push(sphere(
        2.0, ivec3(0, 2, 0),
        marble_material.clone(),
    ));

    let light_material = Rc::new(DiffuseLight(ConstantTex(ivec3(4, 4, 4))));

    // sphere light
    list.push(sphere(
        2.0, ivec3(0, 7, 0),
        light_material.clone()));

    // rectangular light
    list.push(box XYRect {
        x: (3.0, 5.0),
        y: (1.0, 3.0),
        z: -2.0,
        material: light_material.clone()
    });

    return into_bvh(rng, list, (0.0, 1.0));
}

#[allow(dead_code)]
fn cornell_box(rng : &mut Rng) -> Box<Hitable> {
    let mut list : Vec<Box<Hitable>> = Vec::new();

    let red : Rc<Material> = Rc::new(Lambertian(ConstantTex(vec3(0.65, 0.05, 0.05))));
    let white : Rc<Material> = Rc::new(Lambertian(ConstantTex(0.73 * ONE3)));
    let green : Rc<Material> = Rc::new(Lambertian(ConstantTex(vec3(0.12, 0.45, 0.15))));
    let light : Rc<Material> = Rc::new(DiffuseLight(ConstantTex(7.0 * ONE3)));

    list.push(YZRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &green, true));
    list.push(YZRect::new((0.0, 555.0), (0.0, 555.0), 0.0, &red, false));
    list.push(XZRect::new((113.0, 443.0), (127.0, 432.0), 554.0, &light, false));
    list.push(XZRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &white, true));
    list.push(XZRect::new((0.0, 555.0), (0.0, 555.0), 0.0, &white, false));
    list.push(XYRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &white, true));

    let b1 = translate(ivec3(130, 0, 65),
                       rotate(ivec3(0, 1, 0), -18.0,
                              cube(ivec3(0, 0, 0), ivec3(165, 165, 165), &white)));
    let b2 = translate(ivec3(265, 0, 295),
                       rotate(ivec3(0, 1, 0), 15.0,
                              cube(ivec3(0, 0, 0), ivec3(165, 330, 165), &white)));

    list.push(box ConstantMedium {
        boundary: b1,
        density: 0.01,
        phase_function: box Isotropic(ConstantTex(ivec3(1, 1, 1))),
    });
    list.push(box ConstantMedium {
        boundary: b2,
        density: 0.01,
        phase_function: box Isotropic(ConstantTex(ivec3(0, 0, 0))),
    });

    return into_bvh(rng, list, (0.0, 1.0));
}

struct FlipNormals<H : Hitable>(H);

impl<H : Hitable> Hitable for FlipNormals<H> {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
        let FlipNormals(ref inner) = *self;
        inner.hit(rng, r, dist).map(|mut hit| {
            hit.normal = -1.0 * hit.normal;
            return hit;
        })
    }

    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        let FlipNormals(ref inner) = *self;
        return inner.bounding_box(time);
    }
}

fn cube(p0 : Vec3, p1 : Vec3, mat : &Rc<Material>) -> Box<Hitable> {
    into_bvh_det(&mut || 0, vec![
        XYRect::new((p0[X], p1[X]), (p0[Y], p1[Y]), p1[Z], mat, false),
        XYRect::new((p0[X], p1[X]), (p0[Y], p1[Y]), p0[Z], mat, true),
        XZRect::new((p0[X], p1[X]), (p0[Z], p1[Z]), p1[Y], mat, false),
        XZRect::new((p0[X], p1[X]), (p0[Z], p1[Z]), p0[Y], mat, true),
        YZRect::new((p0[Y], p1[Y]), (p0[Z], p1[Z]), p1[X], mat, false),
        YZRect::new((p0[Y], p1[Y]), (p0[Z], p1[Z]), p0[X], mat, true),
    ], (0.0, 1.0))
}

struct Translate {
    inner : Box<Hitable>,
    offset : Vec3,
}

fn translate(offset : Vec3, inner : Box<Hitable>) -> Box<Translate> {
    box Translate {
        inner: inner,
        offset: offset,
    }
}

impl Hitable for Translate {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
        let mut r = (*r).clone();
        r.origin -= self.offset;
        self.inner.hit(rng, &r, dist).map(|mut hit| {
            hit.p += self.offset;
            hit
        })
    }

    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        let mut inner_box = self.inner.bounding_box(time);
        inner_box.min += self.offset;
        inner_box.max += self.offset;
        inner_box
    }
}

struct Rotate {
    inner : Box<Hitable>,
    mat : Mat3,
}

fn rotate(axis : Vec3, angle : f32, inner : Box<Hitable>) -> Box<Rotate> {
    let ct = (-angle).to_radians().cos();
    let st = (-angle).to_radians().sin();
    let u = axis.unit();
    let tensor_matrix = Mat3([
        u[X] * u[X], u[X] * u[Y], u[X] * u[Z],
        u[Y] * u[X], u[Y] * u[Y], u[Y] * u[Z],
        u[Z] * u[X], u[Z] * u[Y], u[Z] * u[Z],
    ]);
    let cross_matrix = Mat3([
        0.0  , -u[Z], u[Y] ,
        u[Z] , 0.0  , -u[X],
        -u[Y], u[X] , 0.0  ,
    ]);

    let mat = MAT_IDENT3 * ct + cross_matrix * st + tensor_matrix * (1.0 - ct);

    let Mat3(foo) = mat;

    println!("{:?}", foo);

    box Rotate {
        inner: inner,
        mat: mat,
    }
}

impl Hitable for Rotate {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
        let mut r = (*r).clone();
        r.origin = self.mat * r.origin;
        r.direction = self.mat * r.direction;
        self.inner.hit(rng, &r, dist).map(|mut hit| {
            let inv = self.mat.transpose();
            hit.p = inv * hit.p;
            hit.normal = inv * hit.normal;
            hit
        })
    }

    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        let mut b = self.inner.bounding_box(time);
        b.min = self.mat * b.min;
        b.max = self.mat * b.max;
        for i in 0..3 {
            if b.max[i] < b.min[i] {
                std::mem::swap(&mut b.max[i], &mut b.min[i]);
            }
        }
        b
    }
}

struct ConstantMedium {
    boundary : Box<Hitable>,
    density : f32,
    phase_function : Box<Material>,
}

impl Hitable for ConstantMedium {
    fn hit(
        &self, rng : &mut Rng, r : &Ray, dist : (f32, f32)
    ) -> Option<HitRecord> {
        let db = rng.gen::<f32>() < 0.00001;
        let db = false;
        let mhit1 = self.boundary.hit(rng, r, (f32::MIN, f32::MAX));
        if let Some(mut hit1) = mhit1 {
            let mhit2 = self.boundary.hit(rng, r, (hit1.t + 0.0001, f32::MAX));
            if let Some(mut hit2) = mhit2 {
                if db {println!("\nt0, t1 = {}, {}", hit1.t, hit2.t)}
                hit1.t = hit1.t.max(dist.0);
                hit2.t = hit2.t.min(dist.1);
                if hit1.t >= hit2.t {
                    return None
                }
                hit1.t = hit1.t.max(0.0);

                let dist_inside = (hit2.t - hit1.t) * r.direction.len();
                let hit_dist = -(1.0 / self.density) * rng.gen::<f32>().ln();
                if db {println!("hit_dist = {}", hit_dist)}
                if hit_dist < dist_inside {
                    let t = hit1.t + hit_dist / r.direction.len();
                    if db {println!("hit.t = {}", t)}
                    let p = r.at(t);
                    if db {println!("hit.p = {}", p)}
                    let normal = ivec3(1, 0, 0);
                    return Some(HitRecord {
                        t: t, p: p,
                        normal: normal, material: &*self.phase_function,
                        uv: (0.0, 0.0),
                    })
                }
            }
        }
        return None
    }

    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        self.boundary.bounding_box(time)
    }
}

struct Isotropic<T : Texture>(T);

impl<T : Texture> Material for Isotropic<T> {
    fn scatter(
        &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
    ) -> Option<(Ray, Vec3)> {
        let Isotropic(ref albedo) = *self;
        let mut r = (*r_in).clone();
        r.origin = hit.p;
        r.direction = rand_in_ball(rng).unit();
        Some((r, albedo.tex_lookup(hit.uv, &hit.p)))
    }
}

fn the_next_week(rng : &mut Rng) -> Box<Hitable> {
    let mut list : Vec<Box<Hitable>> = Vec::new();
    let mut boxlist : Vec<Box<Hitable>> = Vec::new();
    let mut boxlist2 : Vec<Box<Hitable>> = Vec::new();

    let white : Rc<Material> =
        Rc::new(Lambertian(ConstantTex(0.73 * ONE3)));
    let ground : Rc<Material> =
        Rc::new(Lambertian(ConstantTex(vec3(0.48, 0.83, 0.53))));
    let red : Rc<Material> =
        Rc::new(Lambertian(ConstantTex(ivec3(1, 0, 0))));
    let light : Rc<Material> =
        Rc::new(DiffuseLight(ConstantTex(7.0 * ONE3)));

    let nb = 20;
    for i in 0..nb {
        for j in 0..nb {
            let w = 100.0;
            let x0 = -1000.0 + w * i as f32;
            let z0 = -1000.0 + w * j as f32;
            let y0 = 0.0;
            let x1 = x0 + w;
            let mut y1 = 100.0 * (rng.gen::<f32>() + 0.01);
            let z1 = z0 + w;

            if i == 12 && j == 10 {
                y1 = 100.0;
            }

            boxlist.push(cube(vec3(x0, y0, z0), vec3(x1, y1, z1), &ground));
        }
    }

    list.push(into_bvh(rng, boxlist, (0.0, 1.0)));
    let bb = list[0].bounding_box((0.0, 1.0));
    println!("{}, {}", bb.min, bb.max);
    list.push(XZRect::new((123.0, 423.0), (147.0, 412.0), 554.0, &light, false));
    let center = ivec3(400, 400, 200);
    list.push(moving_sphere(50.0, center, center + ivec3(30, 0, 0), 0.0, 1.0,
                     Rc::new(Lambertian(ConstantTex(vec3(0.7, 0.3, 0.1))))));
    list.push(sphere(50.0, ivec3(260, 150, 45),
                                Rc::new(Dielectric(1.5))));
    list.push(sphere(50.0, ivec3(0, 150, 145),
                                Rc::new(Metal{
                                    albedo: vec3(0.8, 0.8, 0.9),
                                    fuzz: 10.0,
                                })));
    let boundary = sphere(70.0, ivec3(360, 150, 145),
                                     Rc::new(Dielectric(1.5)));
    list.push(boundary.clone());
    list.push(box ConstantMedium {
        boundary: boundary,
        density: 0.2,
        phase_function: box Isotropic(ConstantTex(vec3(0.2, 0.4, 0.9))),
    });
    let boundary = sphere(5000.0, ivec3(0, 0, 0),
                                     Rc::new(Dielectric(1.5)));
    list.push(box ConstantMedium {
        boundary: boundary,
        density: 0.0001,
        phase_function: box Isotropic(ConstantTex(ivec3(1, 1, 1))),
    });

    let img = image::open(&Path::new("earth.png")).unwrap();
    let earth = Rc::new(Lambertian(image_texture(&img)));
    list.push(sphere(100.0, ivec3(400, 200, 400), earth));

    let marble = Rc::new(Lambertian(
        PerlinTexture {
            noise: Rc::new(PerlinNoise::new(rng, 256)),
            scale: 0.1,
        }));
    list.push(sphere(80.0, ivec3(220, 280, 300), marble));

    let nspheres = 1000;
    for i in 0..nspheres {
        let center = 165.0 * vec3(rng.gen(), rng.gen(), rng.gen());
        boxlist2.push(sphere(10.0, center, white.clone()));
    }
    list.push(translate(ivec3(-100, 270, 395),
                        rotate(ivec3(0, 1, 0), 15.0,
                               into_bvh(rng, boxlist2, (0.0, 1.0)))));

    return into_bvh(rng, list, (0.0, 1.0));
}
