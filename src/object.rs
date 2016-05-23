use vec3::*;
use mat3::*;
use vec3::Vec3Index::*;
use material::*;
use random::*;
use ray::*;

use std;
use std::f32;
use std::sync::Arc;
use std::f32::consts::PI;
use std::cmp::{max, min, Ordering};

pub struct HitRecord<'a> {
    pub t : f32,
    pub p : Vec3,
    pub normal : Vec3,
    pub material : &'a Material,
    pub uv : (f32, f32),
}

pub trait Object : Sync + Send {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord>;
    fn bounding_box(&self, time : (f32, f32)) -> AABB;
}

#[derive(Clone)]
pub struct Sphere {
    pub material : Arc<Material>,
    pub center0 : Vec3,
    pub dcenter : Vec3,
    pub radius : f32,
}

fn uv_for_sphere_normal(normal : Vec3) -> (f32, f32) {
    let phi = normal[Z].atan2(normal[X]);
    let theta = normal[Y].asin();

    let u = 1.0 - (phi + PI) / (2.0 * PI);
    let v = (theta + PI / 2.0) / PI;

    (u, v)
}

impl Sphere {
    pub fn center(&self, time : f32) -> Vec3 {
        return &self.center0 + time * &self.dcenter
    }
}

pub fn moving_sphere(radius : f32,
          center0 : Vec3, center1 : Vec3,
          time0 : f32, time1 : f32,
          material : Arc<Material>,
) -> Box<Sphere> {
    let dcenter = (&center1 - &center0) / (&time1 - &time0);
    box Sphere {
        material: material,
        center0: &center0 - time0 * &dcenter,
        dcenter: dcenter,
        radius: radius,
    }
}

pub fn sphere(
    radius : f32, center : Vec3, material : Arc<Material>
) -> Box<Sphere> {
    moving_sphere(radius, center, center, 0.0, 1.0, material)
}

impl Object for Sphere {
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

                    return Some(HitRecord {
                        t: *t, p: p, normal: normal, material: &*self.material,
                        uv: uv_for_sphere_normal(normal)
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


#[derive(Clone, Copy)]
pub struct AABB {
    pub min : Vec3,
    pub max : Vec3,
}

impl AABB {
    #[inline(never)]
    pub fn hit(self, r : &Ray, dist : (f32, f32)) -> bool {
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

    pub fn union(&self, other : &AABB) -> AABB {
        return AABB {
            min: Vec3::fmap2(f32::min, &self.min, &other.min),
            max: Vec3::fmap2(f32::max, &self.max, &other.max),
        };
    }
}

pub struct BVHNode {
    pub left : Box<Object>,
    pub right : Box<Object>,
    pub bbox : AABB,
}

impl Object for BVHNode {
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
    mut v : Vec<Box<Object>>, time : (f32, f32)
) -> Box<Object> {
    let axis = rng_fn();

    v.sort_by(&|a : &Box<Object>, b : &Box<Object>| {
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

pub fn into_bvh(
    rng : &mut Rng, v : Vec<Box<Object>>, time : (f32, f32)
) -> Box<Object> {
    let range = Range::new(0, 3);
    into_bvh_det(
        &mut || range.ind_sample(rng),
        v, time)
}

pub struct ConstantMedium {
    pub boundary : Box<Object>,
    pub density : f32,
    pub phase_function : Box<Material>,
}

impl Object for ConstantMedium {
    fn hit(
        &self, rng : &mut Rng, r : &Ray, dist : (f32, f32)
    ) -> Option<HitRecord> {
        let mhit1 = self.boundary.hit(rng, r, (f32::MIN, f32::MAX));
        if let Some(mut hit1) = mhit1 {
            let mhit2 = self.boundary.hit(rng, r, (hit1.t + 0.0001, f32::MAX));
            if let Some(mut hit2) = mhit2 {
                hit1.t = hit1.t.max(dist.0);
                hit2.t = hit2.t.min(dist.1);
                if hit1.t >= hit2.t {
                    return None
                }
                hit1.t = hit1.t.max(0.0);

                let dist_inside = (hit2.t - hit1.t) * r.direction.len();
                let hit_dist = -(1.0 / self.density) * rng.gen::<f32>().ln();
                if hit_dist < dist_inside {
                    let t = hit1.t + hit_dist / r.direction.len();
                    let p = r.at(t);
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

pub struct FlipNormals<H : Object>(H);

impl<H : Object> Object for FlipNormals<H> {
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

pub fn cube(p0 : Vec3, p1 : Vec3, mat : &Arc<Material>) -> Box<Object> {
    into_bvh_det(&mut || 0, vec![
        XYRect::new((p0[X], p1[X]), (p0[Y], p1[Y]), p1[Z], mat, false),
        XYRect::new((p0[X], p1[X]), (p0[Y], p1[Y]), p0[Z], mat, true),
        XZRect::new((p0[X], p1[X]), (p0[Z], p1[Z]), p1[Y], mat, false),
        XZRect::new((p0[X], p1[X]), (p0[Z], p1[Z]), p0[Y], mat, true),
        YZRect::new((p0[Y], p1[Y]), (p0[Z], p1[Z]), p1[X], mat, false),
        YZRect::new((p0[Y], p1[Y]), (p0[Z], p1[Z]), p0[X], mat, true),
    ], (0.0, 1.0))
}

pub struct Translate {
    pub inner : Box<Object>,
    pub offset : Vec3,
}

pub fn translate(offset : Vec3, inner : Box<Object>) -> Box<Translate> {
    box Translate {
        inner: inner,
        offset: offset,
    }
}

impl Object for Translate {
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

pub struct Rotate {
    pub inner : Box<Object>,
    pub mat : Mat3,
}

pub fn rotate(axis : Vec3, angle : f32, inner : Box<Object>) -> Box<Rotate> {
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

    box Rotate {
        inner: inner,
        mat: mat,
    }
}

impl Object for Rotate {
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

macro_rules! aarect {
    ($XYRect:ident,
     [$x:ident, $X:ident],
     [$y:ident, $Y:ident],
     [$z:ident, $Z:ident]) => (
        pub struct $XYRect {
            pub material : Arc<Material>,
            pub $x : (f32, f32),
            pub $y : (f32, f32),
            pub $z : f32,
        }

        impl $XYRect {
            pub fn new(
                $x : (f32, f32), $y : (f32, f32), $z : f32, mat : &Arc<Material>, flip : bool
            ) -> Box<Object> {
                let r = $XYRect {$x: $x, $y: $y, $z: $z, material: mat.clone()};
                if flip { box FlipNormals(r) } else { box r }
            }
        }

        impl Object for $XYRect {
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

pub struct Sky;

impl Object for Sky {
    fn hit(&self, rng : &mut Rng, r : &Ray, dist : (f32, f32)) -> Option<HitRecord> {
        if dist.1 < f32::INFINITY {
            None
        } else {
            // we play fast and loose with floating point, because we are madmen
            let dir = r.direction;
            Some(HitRecord {
                t: f32::INFINITY,
                p: dir * f32::INFINITY,
                normal: dir,
                material: self,
                uv: (0.0, (1.0 + dir.unit()[Y]) / 2.0),
            })
        }
    }

    fn bounding_box(&self, time : (f32, f32)) -> AABB {
        return AABB {
            min: -f32::INFINITY * ONE3,
            max: f32::INFINITY * ONE3,
        }
    }
}

//eh, why not?
impl Material for Sky {
    fn scatter(
        &self, rng : &mut Rng, r_in : &Ray, hit : &HitRecord
    ) -> Option<(Ray, Vec3)> {
        None
    }

    fn emitted(&self, (u, v) : (f32, f32), p : &Vec3) -> Vec3 {
        let t = 0.5 + v;
        (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0)
    }

}
