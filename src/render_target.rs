use std;
use std::ops::{Index, IndexMut};
use std::fs::File;
use std::path::Path;
use std::io::Write;

use image;

pub struct RenderTarget {
    pub size : (u32, u32),
    pub buf : Vec<[f64; 3]>,
    pub samples : u32,
}

impl RenderTarget {
    pub fn new(size : (u32, u32)) -> RenderTarget {
        RenderTarget{
            size: size,
            buf: vec![[0.0; 3]; (size.0 * size.1) as usize],
            samples: 0,
        }
    }

    pub fn write_png(&self, filename : &str) {
        let (nx, ny) = self.size;
        let ns = self.samples;
        let mut img = image::DynamicImage::new_rgb8(nx, ny);

        for y in 0..ny {
            for x in 0..nx {
                let mut col = self[(x, y)].clone();
                col[0] = (col[0] / ns as f64).max(0.0).min(1.0).sqrt();
                col[1] = (col[1] / ns as f64).max(0.0).min(1.0).sqrt();
                col[2] = (col[2] / ns as f64).max(0.0).min(1.0).sqrt();

                let ir = (255.99 * col[0]) as u8;
                let ig = (255.99 * col[1]) as u8;
                let ib = (255.99 * col[2]) as u8;
                img.as_mut_rgb8().unwrap()[(x, ny - y - 1)]
                    = image::Rgb([ir, ig, ib]);
            }
        }
        let ref mut fout = File::create(&Path::new(filename)).unwrap();
        let _ = img.save(fout, image::PNG);
    }


    pub fn write_hdr(&self, filename : &str) {
        let ref mut fout = File::create(&Path::new(filename)).unwrap();

        let bytes = unsafe {
        let ptr : *const u8 = std::mem::transmute(&self.buf[0][0] as *const _);
        let len = 8 * self.size.0 * self.size.1 * 3;
            std::slice::from_raw_parts(ptr, len as usize)
        };

        write!(fout, "{} {} {}\n\n",
               self.size.0, self.size.1, self.samples).unwrap();
        fout.write(bytes).unwrap();
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
