use vec3::*;

use std;
use std::ops::{Index, IndexMut};
use std::fs::File;
use std::path::Path;
use std::io::Write;

use image;
use image::{Luma};
use image::math::utils::clamp;

use num::{Float};

type Image32 = image::ImageBuffer<Luma<f32>, Vec<f32>>;

pub struct RenderTarget {
    pub size : (u32, u32),
    pub buf : Vec<[f64; 3]>,
    pub samples : u32,
}

impl RenderTarget {
    pub fn new(size : (u32, u32)) -> RenderTarget {
        RenderTarget {
            size: size,
            buf: vec![[0.0; 3]; (size.0 * size.1) as usize],
            samples: 0,
        }
    }

    #[allow(dead_code)]
    pub fn write_png(&mut self, filename : &str) {
        let (nx, ny) = self.size;
        let ns = self.samples;
        let mut img = image::DynamicImage::new_rgb8(nx, ny);
        let bloom = self.bloom();

        for y in 0..ny {
            for x in 0..nx {
                let mut col = self[(x, y)].clone();

                for e in 0..3 {
                    col[e] /= ns as f64;
                    col[e] += bloom.get_pixel(x, y).data[0] as f64;
                    col[e] = clamp(col[e], 0.0, 1.0);
                    col[e] = col[e].sqrt();
                }

                // col[0] = clamp(col[0] / ns as f64, 0.0, 1.0).sqrt();
                // col[1] = clamp(col[1] / ns as f64, 0.0, 1.0).sqrt();
                // col[2] = clamp(col[2] / ns as f64, 0.0, 1.0).sqrt();

                let ir = (255.99 * col[0]) as u8;
                let ig = (255.99 * col[1]) as u8;
                let ib = (255.99 * col[2]) as u8;
                img.as_mut_rgb8().unwrap()[(x, y)]
                    = image::Rgb([ir, ig, ib]);
            }
        }
        let ref mut fout = File::create(&Path::new(filename)).unwrap();
        let _ = img.save(fout, image::PNG);
    }


    #[allow(dead_code)]
    pub fn write_hdr(&self, filename : &str) {
        let ref mut fout = File::create(&Path::new(filename)).unwrap();

        let bytes : &[u8] = unsafe {
            let ptr : *const u8 = std::mem::transmute(&self.buf[0] as *const _);
            std::slice::from_raw_parts(ptr, (8 * self.size.0 * self.size.1 * 3) as usize)
        };

        write!(fout, "{} {} {}\n\n",
               self.size.0, self.size.1, self.samples).unwrap();
        fout.write(bytes).unwrap();
    }

    fn bloom(&mut self) -> Image32 {
        let (nx, ny) = self.size;

        let luma_matrix = vec3(0.299, 0.587, 0.114);

        let lumae : Vec<f32> = self.buf.iter().map(|p| {
            let mut l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
            l /= self.samples as f64;
            if l > 1.0 {
                l as f32
            } else {
                0.0
            }
        }).collect();

        let im = Image32::from_vec(nx, ny, lumae).unwrap();
        image::imageops::blur(&im, 2.5)
    }
}

impl Index<(u32, u32)> for RenderTarget {
    type Output = [f64; 3];
    fn index(&self, (x, y) : (u32, u32)) -> &[f64; 3] {
        &self.buf[(y * self.size.0 + x) as usize]
    }
}

impl IndexMut<(u32, u32)> for RenderTarget {
    fn index_mut<'a>(&'a mut self, (x, y) : (u32, u32)) -> &'a mut [f64; 3] {
        &mut self.buf[(y * self.size.0 + x) as usize]
    }
}
