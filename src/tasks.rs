use vec3::*;
use camera::*;
use object::*;
use random::*;
use texture::*;
use material::*;

use std::sync::Arc;

pub struct RenderTask {
    pub camera : Camera,
    pub target_size : (u32, u32),
    pub samples : u32,
    pub world : Box<Object>,
}
struct CornellBox;

impl CornellBox {
    fn box_only(rng : &mut Rng, light_brightness : f32, (lw, ld) : (f32, f32)) -> Vec<Box<Object>> {
        let mut list = vec![];

        let center = vec3(278.0, 554.0, 279.5);
        let size = vec3(lw, 0.0, ld);
        let light : Arc<Material> =
            Arc::new(DiffuseLight(Arc::new(light_brightness * ONE3)));
        let min = center - size;
        let max = center + size;

        let red : Arc<Material> = Arc::new(Lambertian(ConstantTex(vec3(0.65, 0.05, 0.05))));
        let white : Arc<Material> = Arc::new(Lambertian(ConstantTex(0.73 * ONE3)));
        let green : Arc<Material> = Arc::new(Lambertian(ConstantTex(vec3(0.12, 0.45, 0.15))));

        list.push(YZRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &green, true));
        list.push(YZRect::new((0.0, 555.0), (0.0, 555.0), 0.0, &red, false));
        list.push(XZRect::new((min[X], max[X]), (min[Z], max[Z]), center[Y], &light, false));
        list.push(XZRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &white, true));
        list.push(XZRect::new((0.0, 555.0), (0.0, 555.0), 0.0, &white, false));
        list.push(XYRect::new((0.0, 555.0), (0.0, 555.0), 555.0, &white, true));

        list
    }

    fn cube1(rng : &mut Rng, mat : &Arc<Material>) -> Box<Object> {
        translate(ivec3(130, 0, 65),
                  rotate(ivec3(0, 1, 0), -18.0,
                         cube(ivec3(0, 0, 0), ivec3(165, 165, 165), mat)))
    }

    fn cube2(rng : &mut Rng, mat : &Arc<Material>) -> Box<Object> {
        translate(ivec3(265, 0, 295),
                  rotate(ivec3(0, 1, 0), 15.0,
                         cube(ivec3(0, 0, 0), ivec3(165, 330, 165), mat)))
    }
}

#[allow(dead_code)]
pub fn cornell_box(rng : &mut Rng) -> RenderTask {
    let mut list : Vec<Box<Object>> = vec![];

    let white : Arc<Material> = Arc::new(Lambertian(ConstantTex(0.73 * ONE3)));

    list.append(&mut CornellBox::box_only(rng, 15.0, (65.0, 52.5)));

    list.push(CornellBox::cube1(rng, &white));
    list.push(CornellBox::cube2(rng, &white));

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
        samples: 100,
    }
}
