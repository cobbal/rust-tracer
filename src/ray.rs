use vec3::*;

#[derive(Clone,Debug)]
pub struct Ray {
    pub origin : Vec3,
    pub direction : Vec3,
    pub time : f32,
}


impl Ray {
    pub fn at(&self, t : f32) -> Vec3 {
        return &self.origin + &(t * &self.direction);
    }

    pub fn new(origin : Vec3, direction : Vec3, time : f32) -> Ray {
        return Ray { origin: origin, direction: direction, time: time };
    }
}
