use vec3::*;
use std::ops::*;

pub type Color = Vec3;

pub struct Upcast<T>(pub T);

pub fn interpolate<A>(λ : f32, a : A, b : A) -> A
    where A : Mul<f32, Output = A> + Add<A, Output = A>
{
    a * λ + b * (1.0 - λ)
}
