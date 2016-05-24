use vec3::*;
use vec3::Vec3Index::*;
use random::*;

use image;

use std::sync::Arc;
use image::math::utils::clamp;

pub trait Texture : Send + Sync {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3;
}

pub struct ConstantTex(pub Vec3);

impl Texture for ConstantTex {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        let ConstantTex(v) = *self;
        return v;
    }
}

pub struct CheckerTex<T0 : Texture, T1 : Texture> {
    pub odd : T0,
    pub even : T1,
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

pub struct PerlinNoise {
    n : usize,
    ranvec : Vec<Vec3>,
    perm_x : Vec<usize>,
    perm_y : Vec<usize>,
    perm_z : Vec<usize>,
}

impl PerlinNoise {
    pub fn new(rng : &mut Rng, n : usize) -> PerlinNoise {
        let mut pn = PerlinNoise {
            n: n,
            ranvec: vec![ZERO3; n],
            perm_x: vec![0; n],
            perm_y: vec![0; n],
            perm_z: vec![0; n],
        };

        for i in 0..n {
            pn.ranvec[i] = (2.0 * rng.gen::<Vec3>()
                            - ivec3(1, 1, 1)).unit();
            pn.perm_x[i] = i;
            pn.perm_y[i] = i;
            pn.perm_z[i] = i;
        }

        rng.shuffle(&mut pn.perm_x[..]);
        rng.shuffle(&mut pn.perm_y[..]);
        rng.shuffle(&mut pn.perm_z[..]);

        pn
    }

    pub fn noise(&self, p : Vec3) -> f32 {
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

    pub fn turb(&self, mut p : Vec3, depth : usize) -> f32 {
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
pub struct PerlinTexture {
    noise : Arc<PerlinNoise>,
    scale : f32,
}

impl PerlinTexture {
    pub fn new(rng : &mut Rng, scale : f32) -> PerlinTexture {
        PerlinTexture {
            noise: Arc::new(PerlinNoise::new(rng, 256)),
            scale: scale,
        }
    }
}

impl Texture for PerlinTexture {
    fn tex_lookup(&self, uv : (f32, f32), p : &Vec3) -> Vec3 {
        // return vec3(1.0, 1.0, 1.0) * 0.5 * (1.0 + self.noise.turb(self.scale * p, 7));
        ONE3 * 0.5 * (1.0 + (self.scale * p[X] + 5.0 * self.noise.turb(self.scale * p, 7)).sin())
        //return vec3(1,1,1)*0.5*(1 + sin(scale*p.x() + 5*noise.turb(scale*p))) ;

    }
}

pub struct ImageTexture {
   pub  im : image::RgbImage,
}

#[allow(dead_code)]
pub fn image_texture(d_img : &image::DynamicImage) -> ImageTexture {
    ImageTexture {
        im: d_img.to_rgb()
    }
}

impl Texture for ImageTexture {
    fn tex_lookup(&self, mut uv : (f32, f32), p : &Vec3) -> Vec3 {
        let nx = self.im.width();
        let ny = self.im.height();

        uv.1 = 1.0 - uv.1;
        let i = clamp((uv.0 * nx as f32) as i32, 0, nx as i32 - 1);
        let j = clamp((uv.1 * ny as f32) as i32, 0, ny as i32 - 1);
        let pix = self.im.get_pixel(i as u32, j as u32);
        vec3(pix[0] as f32 / 255.0,
             pix[1] as f32 / 255.0,
             pix[2] as f32 / 255.0)
    }
}
