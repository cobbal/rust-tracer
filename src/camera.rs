use vec3::*;
use vec3::Vec3Index::*;
use random::*;
use ray::*;

#[derive(Clone)]
pub struct Camera {
    origin : Vec3,
    upper_left_corner : Vec3,
    horizontal : Vec3,
    vertical : Vec3,
    u : Vec3,
    v : Vec3,
    w : Vec3,
    lens_radius : f32,
    time0 : f32,
    time1 : f32,
}

pub fn camera(
    lookfrom : Vec3, lookat : Vec3, vup : Vec3,
    vfov : f32, aspect : f32, aperture : f32, focus_dist : f32,
    time0 : f32, time1 : f32
) -> Camera {
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
        upper_left_corner: lookfrom
            - half_width * focus_dist * &u
            + half_height * focus_dist * &v
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
    pub fn get_ray(&self, rng : &mut Rng, s : f32, t : f32) -> Ray {
        let rd = self.lens_radius * rand_in_disk(rng);
        let offset = &self.u * rd[X] + &self.v * rd[Y];

        let dir = &self.upper_left_corner
            + s * &self.horizontal
            - t * &self.vertical
            - &self.origin - &offset;
        let time = self.time0 + rng.gen::<f32>() * (self.time1 - self.time0);
        return ray(&self.origin + offset, dir, time);
    }
}
