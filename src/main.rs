#![feature(box_syntax)]
#![allow(unused_variables)]

#![feature(alloc_system)]
extern crate alloc_system;


mod vec3;
use vec3::*;
use vec3::Vec3Index::*;

use std::fs::File;
use std::path::Path;
use std::io;
use std::io::Write;

use std::f32;

extern crate image;
extern crate rand;

use rand::Rng as RngTrait;
type Rng = rand::XorShiftRng;

fn rand_in_ball(rng : &mut Rng) -> Vec3 {
    loop {
        let p = 2.0 * vec3(rng.gen(), rng.gen(), rng.gen()) - vec3(1.0, 1.0, 1.0);
        if dot(&p, &p) < 1.0 {
            return p;
        }
    }
}

fn rand_in_disk(rng : &mut Rng) -> Vec3 {
    loop {
        let p = 2.0 * vec3(rng.gen(), rng.gen(), 0.0) - vec3(1.0, 1.0, 0.0);
        if dot(&p, &p) < 1.0 {
            return p;
        }
    }
}


fn write_buffer(nx : u32, ny : u32, ns : u32, flimg : &mut [&mut [Vec3]]) {
    let mut img = image::DynamicImage::new_rgb8(nx, ny);

    for y in 0..ny {
        for x in 0..nx {
            let mut col = flimg[y as usize][x as usize].clone();
            col /= ns as f32;
            col.fmap_mut(&|x : f32| x.max(0.0).min(1.0));
            col.fmap_mut(&|x : f32| x.sqrt());

            let ir = (255.99 * col[R]) as u8;
            let ig = (255.99 * col[G]) as u8;
            let ib = (255.99 * col[B]) as u8;
            img.as_mut_rgb8().unwrap()[(x, y)] = image::Rgb([ir, ig, ib])
        }
    }
    let ref mut fout = File::create(&Path::new("trace.png")).unwrap();
    let _ = img.save(fout, image::PNG);
}

fn random_scene(rng : &mut Rng) -> Vec<Box<Hitable>> {
    let mut list : Vec<Box<Hitable>> = Vec::new();

    list.push(box Sphere{center: vec3(0.0, -1000.0, 0.0), radius: 1000.0,
                         material: box Lambertian{albedo: vec3(0.5, 0.5, 0.5)}});

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.gen::<f32>();
            let center = vec3(a as f32 + 0.9 * rng.gen::<f32>(),
                              0.2,
                              b as f32 + 0.9 * rng.gen::<f32>());
            if (&center - vec3(4.0, 0.2, 0.0)).len() > 0.9 {
                let mat : Box<Material> =
                    if choose_mat < 0.8 { // diffuse
                        let col = vec3(rng.gen(), rng.gen(), rng.gen())
                            * vec3(rng.gen(), rng.gen(), rng.gen());
                        box Lambertian{albedo: col}
                    } else if choose_mat < 0.95 { //metal
                        let col = (vec3(1.0, 1.0, 1.0)
                                   + vec3(rng.gen(), rng.gen(), rng.gen())) / 2.0;
                        box Metal{albedo: col, fuzz: 0.5 * rng.gen::<f32>()}
                    } else {
                        box Dielectric{ref_idx: 1.5}
                    };
                list.push(box Sphere{center: center, radius: 0.2, material: mat});
            }
        }
    }

    list.push(box Sphere{center: vec3(0.0, 1.0, 0.0), radius: 1.0,
                         material: box Dielectric{ref_idx: 1.5}});
    list.push(box Sphere{center: vec3(-4.0, 1.0, 0.0), radius: 1.0,
                         material: box Lambertian{albedo: vec3(0.4, 0.2, 0.1)}});
    list.push(box Sphere{center: vec3(4.0, 1.0, 0.0), radius: 1.0,
                         material: box Metal{albedo: vec3(0.7, 0.6, 0.5),
                                             fuzz: 0.0}});

    return list;
}

fn main() {
    let mut rng = rand::weak_rng();

    let nx = 200;
    let ny = 100;
    let ns = 10;

    // let world : Vec<Box<Hitable>> = vec![
    //     box Sphere{ center: vec3(0.0, 0.0, -1.0), radius: 0.5,
    //                 material: box Lambertian{ albedo: vec3(0.1, 0.2, 0.5) } },
    //     box Sphere{ center: vec3(0.0, -100.5, -1.0), radius: 100.0,
    //                 material: box Lambertian{ albedo: vec3(0.8, 0.8, 0.0) } },

    //     box Sphere{ center: vec3(1.0, 0.0, -1.0), radius: 0.5,
    //                 material: box Metal{ albedo: vec3(0.8, 0.6, 0.2),
    //                                      fuzz: 0.0 } },
    //     box Sphere{ center: vec3(-1.0, 0.0, -1.0), radius: 0.5,
    //                 material: box Dielectric{ ref_idx: 1.5 } },
    //     box Sphere{ center: vec3(-1.0, 0.0, -1.0), radius: -0.45,
    //                 material: box Dielectric{ ref_idx: 1.5 } },
    // ];

    let world = random_scene(&mut rng);

    let cam = camera(vec3(13.0, 2.0, 3.0),
                     vec3(0.0, 0.0, 0.0),
                     vec3(0.0, 1.0, 0.0),
                     20.0, nx as f32 / ny as f32, 0.1, 10.0);


    let mut counter = 0;
    let log_tick_maybe = |counter : i32| -> bool {
        let pcent = |n : i32| (100 * n / ns as i32);
        let pc_last = pcent(counter - 1);
        let pc_now = pcent(counter);
        if pc_now % 1 == 0 && pc_now != pc_last {
            print!("{} ", pc_now);
            io::stdout().flush().unwrap();
            return true;
        }
        return false;
    };

    // http://stackoverflow.com/a/36376568/73681
    let mut flimg_raw = vec![vec3(0.0, 0.0, 0.0); (nx * ny) as usize];
    let mut grid_base : Vec<_> =
        flimg_raw.as_mut_slice().chunks_mut(nx as usize).collect();
    let mut flimg : &mut [&mut [Vec3]] = grid_base.as_mut_slice();

    for s in 0..ns {
        for y in 0..ny {
            for x in 0..nx {

                let u = (x as f32 + rng.gen::<f32>()) / nx as f32;
                let v = ((ny - y - 1) as f32 + rng.gen::<f32>()) / ny as f32;

                let r = cam.get_ray(&mut rng, u, v);

                flimg[y as usize][x as usize] += color(&mut rng, r, &world);
            }
        }

        counter += 1;
        if log_tick_maybe(counter) {
            write_buffer(nx, ny, s + 1, flimg);
        }
    }

    log_tick_maybe(counter);

    write_buffer(nx, ny, ns, flimg);

    // write_buffer(&img);
}

struct Ray {
    origin : Vec3,
    direction : Vec3,
}

fn ray(origin : Vec3, direction : Vec3) -> Ray {
    return Ray { origin: origin, direction: direction };
}

impl Ray {
    fn at(&self, t : f32) -> Vec3 {
        return &self.origin + &(t * &self.direction);
    }
}

#[inline(never)]
fn color(rng : &mut Rng, r0 : Ray, world : &Hitable) -> Vec3 {
    let mut accumulator = vec3(1.0, 1.0, 1.0);
    let mut r = r0;
    for ttl in 0..50 {
        match world.hit(&r, 0.001, f32::INFINITY) {
            Some(hit) => {
                match hit.material.scatter(rng, &r, &hit) {
                    Some((scattered, attenuation)) => {
                        r = scattered;
                        accumulator *= attenuation;
                        continue;
                    },
                    None => {
                        accumulator = vec3(0.0, 0.0, 0.0);
                        break;
                    }
                }
            },
            None => {
                let unit_direction = r.direction.unit();
                let t = 0.5 * unit_direction[Y] + 1.0;
                accumulator *= (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
                break;
            }
        }
    }
    return accumulator;
}

struct HitRecord<'a> {
    t : f32,
    p : Vec3,
    normal : Vec3,
    material : &'a Material,
}

trait Hitable {
    fn hit(&self, r : &Ray, t_min : f32, t_max : f32)
           -> Option<HitRecord>;
}

struct Sphere {
    material : Box<Material>,
    center : Vec3,
    radius : f32,
}

impl Hitable for Sphere {
    fn hit(&self, r : &Ray, t_min : f32, t_max : f32)
           -> Option<HitRecord> {
        let oc = &r.origin - &self.center;
        let a = dot(&r.direction, &r.direction);
        let b = dot(&oc, &r.direction);
        let c = dot(&oc, &oc) - self.radius * self.radius;
        let disc = b * b - a * c;
        if disc > 0.0 {
            let t0 = (-b - disc.sqrt()) / a;
            let t1 = (-b + disc.sqrt()) / a;
            for t in &[t0, t1] {
                if t_min <= *t && *t < t_max {
                    let p = r.at(*t);
                    let normal = (&p - &self.center) / self.radius;
                    return Some(HitRecord {
                        t: *t, p: p, normal: normal, material: &*self.material
                    });
                }
            }
        }
        return None
    }
}

impl Hitable for Vec<Box<Hitable>> {
    fn hit<'a>(&'a self, r : &Ray, t_min : f32, t_max : f32) -> Option<HitRecord> {
        let mut closest = None;
        for t in self {
            closest = match (closest, t.hit(r, t_min, t_max)) {
                (None, onew) => onew,
                (old, None) => old,
                (Some(old), Some(new)) =>
                    if new.t < old.t {
                        Some(new)
                    } else {
                        Some(old)
                    }
            }
        }
        return closest;
    }
}

struct Camera {
    origin : Vec3,
    lower_left_corner : Vec3,
    horizontal : Vec3,
    vertical : Vec3,
    u : Vec3, v : Vec3, // w : Vec3,
    lens_radius : f32,
}

fn camera(lookfrom : Vec3, lookat : Vec3, vup : Vec3,
          vfov : f32, aspect : f32, aperture : f32, focus_dist : f32)
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
        origin: lookfrom.clone(),
        lower_left_corner: lookfrom
            - half_width * focus_dist * &u
            - half_height * focus_dist * &v
            - focus_dist * &w,
        horizontal: 2.0 * half_width * focus_dist * &u,
        vertical: 2.0 * half_height * focus_dist * &v,
        u: u, v: v, // w: w,
        lens_radius: aperture / 2.0,
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
        return ray(&self.origin + offset, dir);
    }
}

trait Material {
    fn scatter(&self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord)
               -> Option<(Ray, Vec3)>;
}

struct Lambertian {
    albedo : Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord)
               -> Option<(Ray, Vec3)> {
        let target = &hit.p + &hit.normal + rand_in_ball(rng);
        let direction = &target - &hit.p;
        return Some((ray(hit.p.clone(), direction), self.albedo.clone()));


    }
}

fn reflect(v : &Vec3, n : &Vec3) -> Vec3 {
    return v - 2.0 * dot(v, n) * n;
}

fn refract(v : &Vec3, n : &Vec3, ni_over_nt : f32) -> Option<Vec3> {
    let uv = v.unit();
    let dt = dot(&uv, n);
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
        let scattered = reflect(&dir, &hit.normal) + self.fuzz * rand_in_ball(rng);
        if dot(&scattered, &hit.normal) <= 0.0 {
            return None
        }
        return Some((ray(hit.p.clone(), scattered),
                     self.albedo.clone()));
    }
}

struct Dielectric {
    ref_idx : f32,
}

impl Material for Dielectric {
    fn scatter(&self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord)
               -> Option<(Ray, Vec3)> {
        let dir = r_in.direction.unit();
        let reflected = reflect(&dir, &hit.normal);
        let attenuation = vec3(1.0, 1.0, 1.0);

        let (outward_normal, ni_over_nt, cosine) =
            if dot(&dir, &hit.normal) > 0.0 {
                (-1.0 * &hit.normal,
                 self.ref_idx,
                 self.ref_idx * dot(&dir, &hit.normal))
            } else {
                (1.0 * &hit.normal,
                 1.0 / self.ref_idx,
                 -dot(&dir, &hit.normal))
            };

        let out_dir =
            match refract(&r_in.direction, &outward_normal, ni_over_nt) {
                Some(refracted) => {
                    let reflect_prob = schlick(cosine, self.ref_idx);
                    if rng.gen::<f32>() < reflect_prob {
                        reflected
                    } else {
                        refracted
                    }
                },
                None => reflected
            };
        return Some((ray(hit.p.clone(), out_dir), attenuation));

    }
}
