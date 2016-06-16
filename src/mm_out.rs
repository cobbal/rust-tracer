use vec3::*;
use vec3::Vec3Index::*;
use distribution::*;

pub trait MMOut {
    fn mm_out(&self) -> String;
}

impl<M : MMOut> MMOut for Vec<M> {
    fn mm_out(&self) -> String {
        let mut res = String::from("{");
        for (i, e) in self.iter().enumerate() {
            if i > 0 {
                res.push_str(", ");
            }
            let foo = e.mm_out();
            res.push_str(&foo);
        }
        res.push_str("}");
        res
    }
}

impl<'a> MMOut for &'a MMOut {
    fn mm_out(&self) -> String {
        (*self).mm_out()
    }
}

impl MMOut for Vec3 {
    fn mm_out(&self) -> String {
        format!("{{{}, {}, {}}}", self[X], self[Y], self[Z])
    }
}

impl<M : MMOut> MMOut for Weighted<M> {
    fn mm_out(&self) -> String {
        format!("{{{}, {}}}", self.0.mm_out(), self.1)
    }
}
