#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(const_fn)]
#![allow(unused_variables)]

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

use object::*;
use random::*;
use ray::*;
use render_target::*;
use tasks::*;
use vec3::*;


use std::f32;
use std::sync::Arc;
use std::thread;
use std::sync::mpsc::{channel, sync_channel, Sender};
use std::io;
use std::io::{Write};

extern crate image;
extern crate rand;
extern crate num_cpus;
extern crate byteorder;
extern crate num;
extern crate rand_chacha;
extern crate rand_xorshift;

fn render_overlord(base_rng : &mut RngCore, render_task : RenderTask) {

    // std::mem::drop(fs::OpenOptions::new()
    //                .write(true)
    //                .truncate(true)
    //                .create(true)
    //                .open("foo.txt")
    //                .unwrap());

    let render_task = Arc::new(render_task);
    let (nx, ny) = render_task.target_size;
    let mut main_target = RenderTarget::new(render_task.target_size);

    type XSeed = <XorShiftRng as SeedableRng>::Seed;

    enum Task {
        Frame(u32, XSeed, RenderTarget),
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
                        let mut rng = XorShiftRng::from_seed(seed);
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
        let mut seed : XSeed = base_rng.gen();

        let (i, mut worker_target) = task_rx.recv().unwrap();

        let prev_samp = main_target.samples;

        let copy_cutoff = if s < 10 * nworkers as u32 { 0 } else { 20 };

        if worker_target.samples >= copy_cutoff {
            copy_to_main_target(&mut main_target, &mut worker_target);
        }

        let mut nsave = nworkers as u32;
        if nsave < 10 { nsave = 10 };

        if prev_samp / nsave != main_target.samples / nsave {
            main_target.write_img("trace.png");
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

    main_target.write_img("trace.png");
    main_target.write_hdr("raw.rgb");
}

fn render_a_frame(rng : &mut RngCore, task : &RenderTask, target : &mut RenderTarget) {
    let (nx, ny) = task.target_size;
    for y in 0..ny {
        for x in 0..nx {
            let u = (x as f32 + rng.gen::<f32>()) / nx as f32;
            let v = (y as f32 + rng.gen::<f32>()) / ny as f32;

            let r = task.camera.get_ray(rng, u, v);

            let col = ray_trace(rng, r, &*task.world);

            for i in 0..3 {
                target[(x, y)][i] += col[i] as f64;
            }
        }
    }
    target.samples += 1;
}


#[inline(never)]
fn ray_trace(rng : &mut RngCore, r0 : Ray, world : &Object) -> Vec3 {
    let mut accumulator = vec3(0.0, 0.0, 0.0);
    let mut attenuation = vec3(1.0, 1.0, 1.0);
    let mut r = r0;
    let max_ttl = 50;
    for bounce in 0..max_ttl {
        match world.hit(rng, &r, (0.001, f32::INFINITY)) {
            Some(hit) => {
                let emitted = hit.material.emitted(hit.uv, &hit.p);
                accumulator += emitted * attenuation;
                match hit.material.scatter(rng, &r, &hit) {
                    Some((scattered, local_att)) => {
                        r = scattered;
                        attenuation *= local_att;
                        if false {
                            let thresh = 0.25;
                            if
                                attenuation[R] < thresh &&
                                attenuation[G] < thresh &&
                                attenuation[B] < thresh
                            {
                                if rng.gen::<f32>() < thresh {
                                    attenuation = attenuation / thresh;
                                } else {
                                    break;
                                }
                            }
                        }
                        continue;
                    },
                    None => {
                        break;
                    }
                }
            },
            None => {
                break;
            }
        }
    }

    return accumulator;
}

fn main() {
    let seed : u64 = rand::thread_rng().gen();
    println!("let seed = {:?};", seed);
    let mut base_rng : ChaChaRng = ChaChaRng::seed_from_u64(seed);

    let task = the_next_week(&mut base_rng);

    render_overlord(&mut base_rng, task);
}
