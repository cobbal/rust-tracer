use vec3::*;
use vec3::Vec3Index::*;
use camera::*;
use object::*;
use random::*;
use texture::*;
use material::*;

use std::sync::Arc;
use std::path::Path;
use image;

pub struct RenderTask {
    pub camera : Camera,
    pub target_size : (u32, u32),
    pub samples : u32,
    pub world : Box<Object>,
}

#[allow(dead_code)]
pub fn one_weekend(rng : &mut RngCore) -> RenderTask {
    let mut list : Vec<Box<Object>> = Vec::new();

    let chexture = CheckerTex {
        even: ConstantTex(vec3(0.2, 0.3, 0.1)),
        odd: ConstantTex(vec3(0.9, 0.9, 0.9)),
    };
    list.push(sphere(1000.0, vec3(0.0, -1000.0, 0.0),
                     Arc::new(Lambertian(chexture))));

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
                        let col = rng.gen::<Vec3>() * rng.gen::<Vec3>();
                        moving_sphere(0.2,
                               center, center + vec3(0.0, 0.5 * rng.gen::<f32>(), 0.0),
                               0.0, 1.0,
                               Arc::new(Lambertian(ConstantTex(col))))
                    } else if choose_mat < 0.95 { //metal
                        let col = (vec3(1.0, 1.0, 1.0) + rng.gen::<Vec3>()) / 2.0;
                        sphere(0.2, center,
                               Arc::new(Metal{albedo: col, fuzz: 0.5 * rng.gen::<f32>()}))
                    } else {
                        sphere(0.2, center,
                               Arc::new(Dielectric(1.5)))
                    };
                list.push(sph);
            }
        }
    }

    list.push(sphere(
        1.0, ivec3(0, 1, 0),
        Arc::new(Dielectric(1.5))));
    list.push(sphere(
        1.0, ivec3(-4, 1, 0),
        Arc::new(Lambertian(ConstantTex(vec3(0.4, 0.2, 0.1))))));
    list.push(sphere(
        1.0, ivec3(4, 1, 0),
        Arc::new(Metal{albedo: vec3(0.7, 0.6, 0.5),
                      fuzz: 0.0})));

    list.push(box Sky);

    let world = into_bvh(rng, list, (0.0, 1.0));
    let size = (500, 500);

    let lookfrom = ivec3(13, 2, 3);
    let lookat = ivec3(0, 0, 0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let vfov = 20.0;
    let aspect = size.0 as f32 / size.1 as f32;

    let camera = camera(lookfrom, lookat, ivec3(0, 1, 0),
           vfov, aspect,
           aperture, dist_to_focus,
           0.0, 1.0);

    RenderTask {
        world: world,
        camera: camera,
        target_size: size,
        samples: 10000,
    }
}

#[allow(dead_code)]
pub fn the_next_week(rng : &mut RngCore) -> RenderTask {
    let mut list : Vec<Box<Object>> = Vec::new();
    let mut boxlist : Vec<Box<Object>> = Vec::new();
    let mut boxlist2 : Vec<Box<Object>> = Vec::new();

    let white : Arc<Material> =
        Arc::new(Lambertian(ConstantTex(0.73 * ONE3)));
    let ground : Arc<Material> =
        Arc::new(Lambertian(ConstantTex(vec3(0.48, 0.83, 0.53))));
    let red : Arc<Material> =
        Arc::new(Lambertian(ConstantTex(ivec3(1, 0, 0))));

    let nb = 20;
    let gi = -10..10;
    let gj = -5..10;
    let w = 100.0;
    for i in gi.clone() {
        for j in gj.clone() {
            let x0 = w * i as f32;
            let z0 = w * j as f32;
            let y0 = 0.0;
            let x1 = x0 + w;
            let mut y1 = 100.0 * (rng.gen::<f32>() + 0.01);
            let z1 = z0 + w;

            // make sure we see the caustics
            if i == 2 && j == 0 {
                y1 = 90.0;
            }
            if i == 2 && j == -1 {
                y1 = 70.0;
            }
            if i == 3 && j == 1 {
                y1 = 80.0;
            }
            if i == 4 && j == 1 {
                y1 = 30.0;
            }

            boxlist.push(cube(vec3(x0, y0, z0), vec3(x1, y1, z1), &ground));
        }
    }

    let complicated_geom = true;

    if complicated_geom {
        //complicated ground
        list.push(into_bvh(rng, boxlist, (0.0, 1.0)));
    } else {
        // simple ground
        list.push(XZRect::new((w * gi.start as f32, w * gi.end as f32),
                              (w * gj.start as f32, w * gj.end as f32),
                              80.0,
                              &ground, false));
    }
    {
        let center = ivec3(273, 554, 279);
        let size = ivec3(150, 0, 132);
        let adjust = 1.0;
        let light : Arc<Material> =
            Arc::new(DiffuseLight(
                ConstantTex(7.0 * ONE3 / adjust / adjust)));
        let min = center - size * adjust;
        let max = center + size * adjust;

        list.push(XZRect::new((min[X], max[X]), (min[Z], max[Z]), min[Y],
                              &light, false));
    }
    let center = ivec3(400, 400, 200);
    list.push(moving_sphere(50.0, center, center + ivec3(30, 0, 0), 0.0, 1.0,
                     Arc::new(Lambertian(ConstantTex(vec3(0.7, 0.3, 0.1))))));
    list.push(sphere(50.0, ivec3(260, 150, 45),
                                Arc::new(Dielectric(1.5))));
    list.push(sphere(50.0, ivec3(0, 150, 145),
                                Arc::new(Metal{
                                    albedo: vec3(0.8, 0.8, 0.9),
                                    fuzz: 1.0,
                                })));

    let boundary = sphere(70.0, ivec3(360, 150, 145),
                          Arc::new(Dielectric(1.3)));
    list.push(boundary.clone());
    let medium_color = vec3(0.2, 0.4, 0.9);
    let iso = Isotropic(ConstantTex(medium_color));
    let tint = Tinted(ConstantTex(medium_color));
    let mix = MixtureMaterial{p: 0.1, m1: iso, m2: tint};
    list.push(box ConstantMedium {
        boundary: boundary,
        density: 0.1,
        phase_function: box mix,
    });
    let boundary = sphere(5000.0, ivec3(0, 0, 0),
                                     Arc::new(Dielectric(1.5)));
    list.push(box ConstantMedium {
        boundary: boundary,
        density: 0.0001,
        phase_function: box Isotropic(ConstantTex(ivec3(1, 1, 1))),
    });

    let img = image::open(&Path::new("earth.png")).unwrap();
    let earth = Arc::new(Lambertian(image_texture(&img)));
    list.push(sphere(100.0, ivec3(400, 200, 400), earth));

    let marble = Arc::new(Lambertian(PerlinTexture::new(rng, 0.1)));
    list.push(sphere(80.0, ivec3(220, 280, 300), marble));

    for i in 0..500 {
        let rvec : Vec3 = rng.gen();
        let rvec = (rand_in_ball(rng).unit() + ivec3(1, 1, 1)) / 2.0;
        let center = rvec * 165.0;
        boxlist2.push(sphere(10.0, center, white.clone()));
    }

    if complicated_geom {
        list.push(translate(ivec3(-100, 270, 395),
                        // rotate(ivec3(0, 1, 0), 15.0,
                        into_bvh(rng, boxlist2, (0.0, 1.0))));
    }

    let world = into_bvh(rng, list, (0.0, 1.0));

    let size = (1920, 1080);

    let lookfrom = ivec3(478, 278, -600);
    let lookat = ivec3(278, 278, 0);
    let dist_to_focus = 10.0;
    let aperture = 0.0;
    let vfov = 40.0;
    let aspect = size.0 as f32 / size.1 as f32;

    let camera = camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0),
           vfov, aspect,
           aperture, dist_to_focus,
           0.0, 1.0);


    RenderTask {
        world: world,
        camera: camera,
        target_size: size,
        samples: 100000,
    }
}

#[allow(dead_code)]
pub fn simple_light(rng : &mut RngCore) -> RenderTask {
    let noise = Arc::new(PerlinNoise::new(rng, 256));
    let ntex = PerlinTexture::new(rng, 4.0);
    let mut list : Vec<Box<Object>> = Vec::new();

    let marble_material = Arc::new(Lambertian(ntex.clone()));

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

    let light_material = Arc::new(DiffuseLight(ConstantTex(ivec3(4, 4, 4))));

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

    let world = into_bvh(rng, list, (0.0, 1.0));

    let size = (200, 200);

    let lookfrom = ivec3(13, 2, 3);
    let lookat = ivec3(0, 0, 0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let vfov = 20.0;
    let aspect = size.0 as f32 / size.1 as f32;

    let camera = camera(lookfrom, lookat, ivec3(0, 1, 0),
           vfov, aspect,
           aperture, dist_to_focus,
           0.0, 1.0);

    RenderTask {
        world: world,
        camera: camera,
        target_size: size,
        samples: 1000,
    }
}

#[allow(dead_code)]
pub fn cornell_box(rng : &mut RngCore) -> RenderTask {
    let mut list : Vec<Box<Object>> = vec![];

    let red : Arc<Material> = Arc::new(Lambertian(ConstantTex(vec3(0.65, 0.05, 0.05))));
    let white : Arc<Material> = Arc::new(Lambertian(ConstantTex(0.73 * ONE3)));
    let green : Arc<Material> = Arc::new(Lambertian(ConstantTex(vec3(0.12, 0.45, 0.15))));
    let light : Arc<Material> = Arc::new(DiffuseLight(ConstantTex(7.0 * ONE3)));

    list.push(YZRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &green, true));
    list.push(YZRect::new((0.0, 555.0), (0.0, 555.0), 0.0, &red, false));
    list.push(XZRect::new((113.0, 443.0), (127.0, 432.0), 554.0, &light, false));
    list.push(XZRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &white, true));
    list.push(XZRect::new((0.0, 555.0), (0.0, 555.0), 0.0, &white, false));
    list.push(XYRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &white, true));

    let b1 = cube(ivec3(0, 0, 0), ivec3(165, 165, 165), &white);
    let b2 = cube(ivec3(0, 0, 0), ivec3(165, 330, 165), &white);

    let mist1 = box ConstantMedium {
        boundary: b1,
        density: 0.01,
        phase_function: box Isotropic(ConstantTex(ivec3(1, 1, 1))),
    };
    let mist2 = box ConstantMedium {
        boundary: b2,
        density: 0.01,
        phase_function: box Isotropic(ConstantTex(ivec3(0, 0, 0))),
    };

    let xmist1 = translate(ivec3(130, 0, 65),
                           rotate(ivec3(0, 1, 0), -18.0,
                                  mist1));
    let xmist2 = translate(ivec3(265, 0, 295),
                           rotate(ivec3(0, 1, 0), 12.0,
                                  mist2));

    list.push(xmist1);
    list.push(xmist2);

    let size = (500, 500);

    let camera = camera_setup(ivec3(278, 278, -800))
        .look_at(ivec3(278, 278, 0))
        .focus_dist(10.0)
        .vfov(40.0)
        .aspect(size)
        .to_camera();

    let world = into_bvh(rng, list, (0.0, 1.0));

    RenderTask {
        world: world,
        camera: camera,
        target_size: size,
        samples: 1000,
    }
}

#[allow(dead_code)]
pub fn two_spheres(rng : &mut RngCore) -> RenderTask {
    let mut list : Vec<Box<Object>> = vec![];

    let chexture = Arc::new(Lambertian(CheckerTex {
        even: ConstantTex(vec3(0.2, 0.3, 0.1)),
        odd: ConstantTex(vec3(0.9, 0.9, 0.9)),
    }));

    list.push(sphere(10.0, ivec3(0, -10, 0), chexture.clone()));
    list.push(sphere(10.0, ivec3(0, 10, 0), chexture.clone()));
    list.push(box Sky);

    let size = (200, 100);
    RenderTask {
        world: into_bvh(rng, list, (0.0, 1.0)),
        camera: camera_setup(ivec3(13, 2, 3))
            .aspect(size)
           .to_camera(),
        target_size: size,
        samples: 1000,
    }
}

#[allow(dead_code)]
pub fn lambertian_test(rng : &mut RngCore, angle : f32) -> RenderTask {
    let mut list : Vec<Box<Object>> = vec![];

    let light : Arc<Material> = Arc::new(DiffuseLight(ConstantTex(5.0 * ONE3)));
    let red : Arc<Material> = Arc::new(Lambertian(ConstantTex(vec3(0.9, 0.3, 0.3))));

    list.push(sphere(1.0, ivec3(0, 0, 10), light));
    list.push(XYRect::new((-1.0, 1.0), (-1.0, 1.0), 0.0, &red, false));

    RenderTask {
        world: into_bvh(rng, list, (0.0, 1.0)),
        camera: camera_setup(
            9.0 * vec3(0.0,
                       angle.to_radians().sin(),
                       angle.to_radians().cos()))
            .to_camera(),
        target_size: (500, 500),
        samples: 1000,
    }
}
