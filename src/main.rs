#![feature(box_syntax, box_patterns)]
#![feature(const_fn)]
#![feature(custom_derive)]
#![feature(associated_type_defaults)]
#![feature(std_panic, panic_handler)]
#![feature(trace_macros)]
#![feature(non_ascii_idents)]

#![allow(unused_variables)]
#![allow(dead_code)]

#[macro_use]
mod distribution;
mod camera;
mod mat3;
mod material;
mod object;
mod random;
mod ray;
mod render_target;
mod tasks;
mod texture;
mod vec3;
mod utils;
mod mm_out;

use material::*;
use object::*;
use random::*;
use ray::*;
use render_target::*;
use tasks::*;
use distribution::*;
use mm_out::*;
use utils::*;
use vec3::*;
use camera::*;

use std::f32;
use std::sync::Arc;
use std::thread;
use std::sync::mpsc::{channel, sync_channel, Sender};
use std::io;
use std::io::{Write};
use std::panic;
use std::process;

extern crate image;
extern crate rand;
extern crate num_cpus;
extern crate byteorder;
extern crate num;

const ENABLE_BLOOM : bool = true;

fn render_overlord(base_rng : &mut Rng, render_task : RenderTask) {
    // make thread panics kill the program, as per
    // http://stackoverflow.com/a/36031130/73681
    let orig_hook = panic::take_hook();
    panic::set_hook(box move |panic_info| {
        // invoke the default hook and exit the process
        orig_hook(panic_info);
        process::exit(1);
    });

    let render_task = Arc::new(render_task);
    let (nx, ny) = render_task.target_size;
    let mut main_target = RenderTarget::new(render_task.target_size);

    enum Task {
        Frame(u32, RngSeed, RenderTarget),
        Kill,
    };

    struct Worker {
        tx : Sender<Task>,
        join_handle : thread::JoinHandle<()>,
    };

    let nworkers = num_cpus::get();
    // let nworkers = 1;
    println!("running with {} threads", nworkers);

    let (task_tx, task_rx) = sync_channel::<(usize, RenderTarget)>(0);

    let mut workers : Vec<Worker> = vec![];

    for i in 0..nworkers {
        let task_tx = task_tx.clone();
        let (wtx, wrx) = channel::<Task>();
        let render_task = render_task.clone();
        let thread = thread::spawn(move || {
            let mut worker_target = RenderTarget::new(render_task.target_size);
            loop {
                task_tx.send((i, worker_target)).unwrap();
                match wrx.recv().unwrap() {
                    Task::Kill => {
                        println!("thread {} exiting", i);
                        break;
                    },
                    Task::Frame(s, seed, returned_target) => {
                        worker_target = returned_target;
                        let mut rng = Rng::from_seed(seed);
                        render_a_frame(&mut rng, &render_task, &mut worker_target);
                    },
                }
            }
        });

        workers.push(Worker {
            tx: wtx,
            join_handle: thread,
        })
    }

    let copy_to_main_target =
        |main : &mut RenderTarget, source : &mut RenderTarget| {
            for y in 0..ny {
                for x in 0..nx {
                    for e in 0..3 {
                        main[(x, y)][e] += source[(x, y)][e];
                        source[(x, y)][e] = 0.0;
                    }
                }
            }
            main.samples += source.samples;
            source.samples = 0;
        };

    for s in 0..render_task.samples {
        // mix up the seed some or else the generator duplicates itself
        let mut seed : RngSeed = base_rng.gen();
        let a_big_prime_number = 2780564309;
        for i in 1..seed.len() {
            seed[i] += base_rng.gen::<u32>() * a_big_prime_number;
        }

        let (i, mut worker_target) = task_rx.recv().unwrap();

        let prev_samp = main_target.samples;

        let copy_cutoff = if s < 10 * nworkers as u32 { 0 } else { 20 };

        if worker_target.samples >= copy_cutoff {
            copy_to_main_target(&mut main_target, &mut worker_target);
        }

        let mut nsave = nworkers as u32;
        if nsave < 10 { nsave = 10 };

        if prev_samp / nsave != main_target.samples / nsave {
            main_target.write_png("trace.png", ENABLE_BLOOM);
            // main_target.write_hdr("raw.rgb");

            print!("{} ", main_target.samples);
            io::stdout().flush().unwrap();
        }
        workers[i].tx.send(Task::Frame(s, seed, worker_target)).unwrap();
    }

    for i in 0..nworkers {
        let (i, mut worker_target) = task_rx.recv().unwrap();
        copy_to_main_target(&mut main_target, &mut worker_target);
        workers[i].tx.send(Task::Kill).unwrap();
    }

    for h in workers.into_iter().map(|w| w.join_handle) {
        h.join().unwrap()
    }

    main_target.write_png("trace.png", ENABLE_BLOOM);
    main_target.write_hdr("raw.rgb");
}

#[derive(Clone, Debug)]
struct LensPoint {
    camera : Camera,
    origin : Vec3,
    time : f32,
}

mimpl!{[] Meas[Vec3] for LensPoint {
    total_mass(&self) {
        self.camera.total_mass()
    }

    sample(&self, rng) {
        let st = (rng.gen(), rng.gen());
        Weighted(self.camera.get_ray_from(rng, st, self.origin).direction, 1.0)
    }

    pdf(&self, x) {
        self.camera.pdf(&Ray::new(self.origin, *x, self.time))
    }
}}

fn render_a_frame(rng : &mut Rng, task : &RenderTask, target : &mut RenderTarget) {
    let (nx, ny) = task.target_size;
    for y in 0..ny {
        for x in 0..nx {
            let u = (x as f32 + rng.gen::<f32>()) / nx as f32;
            let v = (y as f32 + rng.gen::<f32>()) / ny as f32;

            let r = task.camera.get_ray(rng, (u, v));
            let camera_point = Dirac(r.direction);
            let camera_fuzzy = LensPoint {
                origin: r.origin,
                camera: task.camera.clone(),
                time: r.time,
            };
            let origin = RaySendingPoint {
                point: r.origin,
                directions: box camera_point,
                is_dirac: true,
            };
            let Weighted(light, mut light_factor) = task.world.sample(rng);
            light_factor *= task.world.total_mass();

            let dest = RaySendingPoint {
                point: light.0.origin,
                directions: light.0.out_dir,
                is_dirac: false,
            };

            let col = bitrace(rng,
                              origin, dest,
                              light_factor * light.0.alb,
                              r.time, &*task.world);

            for i in 0..3 {
                target[(x, y)][i] += col[i] as f64;
            }
        }
    }
    target.samples += 1;
}

struct RaySendingPoint {
    point : Vec3,
    directions : Box<Meas<Vec3>>,
    is_dirac : bool,
}

fn bitrace(
    rng : &mut Rng,
    origin : RaySendingPoint,
    destination : RaySendingPoint,
    light_color : Vec3,
    time : f32,
    world : &Object,
) -> Color {
    let light_terminate_prob : f32 = 0.2;
    let mut ll = 1.0;
    let ll = &mut ll;

    let factor = |ll : &mut f32, p : f32| {
        // println!("factor({})", p);
        *ll *= p;
    };

    let ray : Ray;

    let res = if rng.gen::<f32>() < light_terminate_prob {
        factor(ll, 1.0 / light_terminate_prob);

        let dir = destination.point - origin.point;
        ray = Ray::new(origin.point, dir, time);

        match trace1(rng, ray.clone(), world) {
            None => ZERO3,
            Some((t, mi)) => {
                if t < 1.0 - 0.0001 {
                    ZERO3
                    // TODO: better shadows
                } else {
                    factor(ll, origin.directions.pdf(&dir));
                    factor(ll, destination.directions.pdf(&-dir));
                    factor(ll, 1.0 / dot(dir, dir));
                    light_color
                }
            }
        }
    } else {
        factor(ll, 1.0 / (1.0 - light_terminate_prob));
        let Weighted(dir, dir_weight) = origin.directions.sample(rng);
        factor(ll, dir_weight);
        ray = Ray::new(origin.point, dir, time);
        match trace1(rng, ray.clone(), world) {
            None => ZERO3,
            Some((_, Absorb)) => ZERO3,
            Some((_, Emit(color))) =>
                // kinda hacky, but don't know how else to solve it
                if origin.is_dirac { color } else { ZERO3 },
            Some((_, Scatter(scatter_meas))) => {
                let pt = RaySendingPoint {
                    point: scatter_meas.origin,
                    directions: scatter_meas.out_dir,
                    is_dirac: false,
                };
                scatter_meas.alb * bitrace(rng,
                                           pt, destination,
                                           light_color,
                                           time, world)
            }
        }
    };

    (*ll) * res
}

type T1R = Option<(f32, MaterialInteraction)>;

fn trace1(rng : &mut Rng, r : Ray, world : &Object) -> T1R {
    world.hit(rng, &r, (0.001, f32::INFINITY)).map( |hit| {
        (hit.t, hit.material.scatter(&r, &hit))
    })
}

fn test_mplus(rng : &mut Rng) {
    let a = Dirac(3.0);
    let b = Dirac(4.0);
    let c = MPlus::new(a, b, 0.5);

    let mut sum : f32 = 0.0;
    let mut count : f32 = 0.0;

    for i in 0..20 {
        println!("{:?}", c.sample(rng))
    }

    for i in 0..10000 {
        let Weighted(x, w) = c.sample(rng);
        sum += x * w;
        count += 1.0;
    }

    println!("{} / {} = {}", sum, count, sum / count);
}

#[derive(Clone)]
struct CosineSample(CosineDist);
impl MassAndSample<Vec3> for CosineSample {
    fn total_mass(&self) -> f32 { 1.0 }
    fn sample(&self, rng : &mut Rng) -> Weighted<Vec3> {
        let c = rand_in_ball(rng);
        let w = self.0.pdf(&c);
        Weighted(c, w)
    }
}

fn test_dists(rng : &mut Rng) {
    let norm = rand_in_ball(rng);
    let cd = CosineDist::new(norm);
    let cd2 = CosineSample(cd.clone());
    let cd3 = Weighted(MPlus::new(cd.clone(), cd2.clone(), 0.5), 0.5);

    let mut vec : Vec<Weighted<Vec3>> = vec![];
    for i in 0..10000 {
        let Weighted(v, w) = cd3.sample(rng);
        vec.push(Weighted(v.unit(), w));
    }

    let out : Vec<&MMOut> = vec![&norm, &vec];
    println!("{}", out.mm_out());
}

fn test_camera_meas(rng : &mut Rng) {
    let cam = cornell_box(rng).camera;

    let r = cam.get_ray(rng, (0.5, 0.5));
    println!("{:?} {}", r, cam.pdf(&r))
}

fn main() {

    let x = AABB{min: ZERO3, max: ZERO3};
    let r = Ray::new(ZERO3, ZERO3, 0.0);
    x.hit(&r, (0.0, 1.0));

    let seed : RngSeed = rand::thread_rng().gen();
    // println!("let seed = {:?};", seed);
    let mut rng : Rng = Rng::from_seed(seed);

    // return test_mplus(&mut rng);
    // return test_dists(&mut rng);
    // return test_camera_meas(&mut rng);

    let task = cornell_box(&mut rng);

    render_overlord(&mut rng, task);
}
