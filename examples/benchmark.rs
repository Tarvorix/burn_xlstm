//! Reproducible forward/backward kernel timings; GPU runs include synchronization.
use burn::tensor::{Device, Distribution, Tensor};
use burn_xlstm::functional::{MlstmInput, MlstmState};
use std::{hint::black_box, time::Instant};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Parallel,
    Chunked,
    Recurrent,
}

fn run(input: &MlstmInput, mode: Mode) -> Tensor<4> {
    match mode {
        Mode::Parallel => input.clone().parallel(1e-6),
        Mode::Chunked => input.clone().chunked(None, 64, 1e-6).0,
        Mode::Recurrent => {
            let [batch, heads, time, dim] = input.query.dims();
            let mut state = MlstmState::zeros([batch, heads, dim], &input.query.device());
            let mut outputs = Vec::with_capacity(time);
            for t in 0..time {
                let take = |x: &Tensor<4>| x.clone().slice([0..batch, 0..heads, t..t + 1]);
                let (output, next) = MlstmInput {
                    query: take(&input.query),
                    key: take(&input.key),
                    value: take(&input.value),
                    input: take(&input.input),
                    forget: take(&input.forget),
                }
                .step(state, 1e-6);
                outputs.push(output);
                state = next;
            }
            Tensor::cat(outputs, 2)
        }
    }
}

fn measure(device: &Device, mut work: impl FnMut()) -> f64 {
    // Warmup excludes compilation/startup. Synchronize every measured sample.
    work();
    device.sync().unwrap();
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        device.sync().unwrap();
        let start = Instant::now();
        work();
        device.sync().unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    samples[1]
}

fn main() {
    let gpu = std::env::args().any(|arg| arg == "--wgpu");
    #[cfg(feature = "wgpu")]
    let device = if gpu {
        let mut device = Device::wgpu(burn::tensor::DeviceKind::default());
        device.configure(burn::tensor::FloatDType::F32).unwrap();
        device
    } else {
        Device::flex()
    };
    #[cfg(not(feature = "wgpu"))]
    let device = {
        assert!(!gpu, "enable --features wgpu for GPU benchmarks");
        Device::flex()
    };
    let backend = if gpu { "wgpu" } else { "flex" };
    println!("backend,mode,phase,batch,heads,time,head_dim,chunk,median_ms");
    for time in [128, 512, 2048] {
        for training in [false, true] {
            let device = if training {
                device.clone().autodiff()
            } else {
                device.clone()
            };
            device.seed(123);
            let tensor = |shape| {
                let x = Tensor::random(shape, Distribution::Normal(0.0, 0.3), &device);
                if training { x.require_grad() } else { x }
            };
            let input = MlstmInput {
                query: tensor([2, 4, time, 16]),
                key: tensor([2, 4, time, 16]),
                value: tensor([2, 4, time, 16]),
                input: tensor([2, 4, time, 1]),
                forget: {
                    let x = tensor([2, 4, time, 1]) + 2.0;
                    if training {
                        x.detach().require_grad()
                    } else {
                        x
                    }
                },
            };
            for mode in [Mode::Parallel, Mode::Chunked, Mode::Recurrent] {
                let milliseconds = measure(&device, || {
                    let output = run(&input, mode);
                    if training {
                        let grads = output.square().mean().backward();
                        let checksum = [
                            &input.query,
                            &input.key,
                            &input.value,
                            &input.input,
                            &input.forget,
                        ]
                        .into_iter()
                        .map(|x| x.grad(&grads).unwrap().sum())
                        .reduce(|a, b| a + b)
                        .unwrap();
                        black_box(checksum.into_scalar::<f32>());
                    } else {
                        black_box(output.sum().into_scalar::<f32>());
                    }
                });
                let phase = if training {
                    "forward_backward"
                } else {
                    "forward"
                };
                println!("{backend},{mode:?},{phase},2,4,{time},16,64,{milliseconds:.3}");
            }
        }
    }
}
