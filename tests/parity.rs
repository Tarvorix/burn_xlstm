//! Numerical comparisons with the pinned, unmodified NX-AI implementation.
mod common;

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Device, Gradients, Tensor},
};
use burn_xlstm::{
    components::{CausalConv, HeadLinear, Norm},
    functional::{MlstmInput, MlstmState, SlstmGates, SlstmState},
    stack::{BlockKind, MemoryLayer, Xlstm, XlstmConfig},
};
use common::{close, fixture, tensor};
use serde_json::Value;

fn matrix_input(data: &Value, device: &Device) -> MlstmInput {
    MlstmInput {
        query: tensor(&data["query"], device).require_grad(),
        key: tensor(&data["key"], device).require_grad(),
        value: tensor(&data["value"], device).require_grad(),
        input: tensor(&data["input"], device).require_grad(),
        forget: tensor(&data["forget"], device).require_grad(),
    }
}

#[test]
fn matrix_kernels_and_gradients_match_nxai() {
    let device = Device::flex().autodiff();
    let data = &fixture()["mlstm"];
    let input = matrix_input(data, &device);
    let parallel = input.clone().parallel(1e-6);
    close(parallel.clone(), tensor(&data["parallel"], &device), 2e-6);
    let grads = (parallel * tensor::<4>(&data["loss_weights"], &device))
        .sum()
        .backward();
    for (name, parameter) in [
        ("query", &input.query),
        ("key", &input.key),
        ("value", &input.value),
        ("input", &input.input),
        ("forget", &input.forget),
    ] {
        close(
            parameter.grad(&grads).unwrap(),
            tensor(&data["gradients"][name], &Device::flex()),
            2e-5,
        );
    }
    let mut state = MlstmState::zeros([2, 2, 3], &device);
    let mut outputs = Vec::new();
    for t in 0..5 {
        let slice = |x: &Tensor<4>| x.clone().slice([0..2, 0..2, t..t + 1]);
        let step = MlstmInput {
            query: slice(&input.query),
            key: slice(&input.key),
            value: slice(&input.value),
            input: slice(&input.input),
            forget: slice(&input.forget),
        };
        let (output, next) = step.step(state, 1e-6);
        state = next;
        outputs.push(output);
    }
    let recurrent = Tensor::cat(outputs, 2);
    close(recurrent.clone(), tensor(&data["recurrent"], &device), 2e-6);
    let grads = (recurrent * tensor::<4>(&data["loss_weights"], &device))
        .sum()
        .backward();
    for (name, parameter) in [
        ("query", &input.query),
        ("key", &input.key),
        ("value", &input.value),
        ("input", &input.input),
        ("forget", &input.forget),
    ] {
        close(
            parameter.grad(&grads).unwrap(),
            tensor(&data["recurrent_gradients"][name], &Device::flex()),
            2e-5,
        );
    }
    close(state.memory, tensor(&data["memory"], &device), 2e-6);
    close(state.normalizer, tensor(&data["normalizer"], &device), 2e-6);
    close(state.log_scale, tensor(&data["log_scale"], &device), 2e-6);
}

#[test]
fn scalar_recurrence_states_and_gradients_match_nxai() {
    let device = Device::flex().autodiff();
    let data = &fixture()["slstm"];
    let gates = tensor::<4>(&data["gates"], &device).require_grad();
    let mut state = SlstmState::zeros([2, 2, 3], &device);
    let mut outputs = Vec::new();
    for t in 0..5 {
        let gate = |g| {
            gates
                .clone()
                .slice([t..t + 1, 0..2, g..g + 1, 0..6])
                .reshape([2, 2, 3])
        };
        state = SlstmGates {
            input: gate(0),
            forget: gate(1),
            candidate: gate(2),
            output: gate(3),
        }
        .step(state);
        outputs.push(state.hidden.clone().reshape([2, 1, 6]));
    }
    let output = Tensor::cat(outputs, 1);
    close(output.clone(), tensor(&data["output"], &device), 2e-6);
    let states = Tensor::stack::<4>(
        vec![
            state.hidden,
            state.memory,
            state.normalizer,
            state.log_scale,
        ],
        0,
    )
    .reshape([4, 2, 6]);
    close(states, tensor(&data["state"], &device), 2e-6);
    let grads = (output * tensor::<3>(&data["loss_weights"], &device))
        .sum()
        .backward();
    close(
        gates.grad(&grads).unwrap(),
        tensor(&data["gate_gradients"], &Device::flex()),
        2e-5,
    );
}

// The same layout map is used for loading weights and comparing all parameter
// gradients. This makes transposition/gate ordering explicit and auditable.
struct Transfer<'a> {
    data: &'a Value,
    device: &'a Device,
    gradients: Option<&'a Gradients>,
    count: usize,
}

impl Transfer<'_> {
    fn read<const D: usize>(&self, name: &str) -> Tensor<D> {
        tensor(&self.data[name], self.device)
    }

    fn parameter<const D: usize>(
        &mut self,
        parameter: &mut Param<Tensor<D>>,
        expected: Tensor<D>,
        name: &str,
    ) {
        assert_eq!(parameter.val().dims(), expected.dims(), "{name}");
        if let Some(gradients) = self.gradients {
            let actual = parameter
                .val()
                .grad(gradients)
                .unwrap_or_else(|| panic!("missing gradient for {name}"));
            close(actual, expected, 8e-5);
        } else {
            *parameter = Param::from_tensor(expected);
        }
        self.count += 1;
    }

    fn linear(&mut self, layer: &mut Linear, name: &str) {
        let key = format!("{name}.weight");
        self.parameter(&mut layer.weight, self.read::<2>(&key).transpose(), &key);
        if let Some(bias) = &mut layer.bias {
            let key = format!("{name}.bias");
            self.parameter(bias, self.read(&key), &key);
        }
    }

    fn head(&mut self, layer: &mut HeadLinear, name: &str) {
        let key = format!("{name}.weight");
        self.parameter(&mut layer.weight, self.read(&key), &key);
    }

    fn norm(&mut self, layer: &mut Norm, name: &str) {
        let key = format!("{name}.weight");
        self.parameter(&mut layer.weight, self.read(&key), &key);
    }

    fn conv(&mut self, layer: &mut CausalConv, name: &str) {
        let key = format!("{name}.conv.weight");
        self.parameter(
            &mut layer.weight,
            self.read::<3>(&key).squeeze_dim::<2>(1).transpose(),
            &key,
        );
        let key = format!("{name}.conv.bias");
        self.parameter(&mut layer.bias, self.read(&key), &key);
    }

    fn model(&mut self, model: &mut Xlstm) {
        for (index, block) in model.blocks.iter_mut().enumerate() {
            self.norm(&mut block.norm, &format!("{index}.xlstm_norm"));
            let name = format!("{index}.xlstm");
            match &mut block.memory {
                MemoryLayer::Mlstm(layer) => {
                    self.linear(&mut layer.up, &format!("{name}.proj_up"));
                    self.linear(&mut layer.down, &format!("{name}.proj_down"));
                    self.head(&mut layer.query, &format!("{name}.q_proj"));
                    self.head(&mut layer.key, &format!("{name}.k_proj"));
                    self.head(&mut layer.value, &format!("{name}.v_proj"));
                    if let Some(conv) = &mut layer.conv {
                        self.conv(conv, &format!("{name}.conv1d"));
                    }
                    self.linear(
                        &mut layer.cell.input_gate,
                        &format!("{name}.mlstm_cell.igate"),
                    );
                    self.linear(
                        &mut layer.cell.forget_gate,
                        &format!("{name}.mlstm_cell.fgate"),
                    );
                    self.norm(&mut layer.cell.norm, &format!("{name}.mlstm_cell.outnorm"));
                    let key = format!("{name}.learnable_skip");
                    self.parameter(&mut layer.skip, self.read(&key), &key);
                }
                MemoryLayer::Slstm(layer) => {
                    if let Some(conv) = &mut layer.conv {
                        self.conv(conv, &format!("{name}.conv1d"));
                    }
                    // Reference layer assigns fgate(x) to i and igate(x) to f.
                    self.head(&mut layer.input, &format!("{name}.fgate"));
                    self.head(&mut layer.forget, &format!("{name}.igate"));
                    self.head(&mut layer.candidate, &format!("{name}.zgate"));
                    self.head(&mut layer.output, &format!("{name}.ogate"));
                    self.norm(&mut layer.norm, &format!("{name}.group_norm"));
                    let key = format!("{name}.slstm_cell._recurrent_kernel_");
                    let [_, heads, dim] = layer.cell.bias.val().dims();
                    for (gate, recurrent) in layer.cell.recurrent.iter_mut().enumerate() {
                        self.parameter(
                            &mut recurrent.weight,
                            self.read::<3>(&key).slice([
                                0..heads,
                                gate * dim..(gate + 1) * dim,
                                0..dim,
                            ]),
                            &key,
                        );
                    }
                    let key = format!("{name}.slstm_cell._bias_");
                    self.parameter(
                        &mut layer.cell.bias,
                        self.read::<1>(&key).reshape([4, heads, dim]),
                        &key,
                    );
                }
            }
            if let Some((norm, feedforward)) = &mut block.feedforward {
                self.norm(norm, &format!("{index}.ffn_norm"));
                self.linear(&mut feedforward.up, &format!("{index}.ffn.proj_up"));
                self.linear(&mut feedforward.down, &format!("{index}.ffn.proj_down"));
            }
        }
        self.norm(
            model.post_norm.as_mut().unwrap(),
            &model.blocks.len().to_string(),
        );
    }
}

fn reference_model(device: &Device) -> Xlstm {
    XlstmConfig::new(
        8,
        vec![BlockKind::Mlstm, BlockKind::Slstm, BlockKind::Mlstm],
    )
    .with_num_heads(2)
    .with_mlstm_inner_dim(Some(16))
    .with_feedforward_dim(Some(12))
    .with_conv_kernel(3)
    .init(device)
    .unwrap()
}

#[test]
fn mixed_stack_outputs_and_all_parameter_gradients_match_nxai() {
    stack_parity(&Device::flex().autodiff());
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a working GPU; run explicitly with --features wgpu -- --ignored"]
fn wgpu_stack_outputs_and_all_parameter_gradients_match_nxai() {
    let mut device = Device::wgpu(burn::tensor::DeviceKind::default());
    // Test numerical parity in explicit float32 on both devices.
    device.configure(burn::tensor::FloatDType::F32).unwrap();
    let device = device.autodiff();
    stack_parity(&device);
    shape_sweep(&device);
}

fn stack_parity(device: &Device) {
    let data = &fixture()["stack"];
    check_stack(reference_model(device), data, device, false);
}

fn check_stack(mut model: Xlstm, data: &Value, device: &Device, chunked: bool) {
    eprintln!("reference case seed={}, chunked={chunked}", data["seed"]);
    let scalar_blocks = model
        .blocks
        .iter()
        .filter(|block| matches!(block.memory, MemoryLayer::Slstm(_)))
        .count();
    let mut load = Transfer {
        data: &data["weights"],
        device,
        gradients: None,
        count: 0,
    };
    load.model(&mut model);
    // One recurrent tensor in PyTorch is split into four Burn parameters.
    assert_eq!(
        load.count,
        data["weights"].as_object().unwrap().len() + 3 * scalar_blocks
    );
    let input = tensor::<3>(&data["input"], device).require_grad();
    let output = if chunked {
        let [batch, time, width] = input.dims();
        let mut outputs = Vec::new();
        let mut state = None;
        let mut start = 0;
        while start < time {
            let end = (start + 3).min(time);
            let (output, next) = model
                .forward_with_state(input.clone().slice([0..batch, start..end, 0..width]), state);
            outputs.push(output);
            state = Some(next);
            start = end;
        }
        Tensor::cat(outputs, 1)
    } else {
        model.forward(input.clone())
    };
    close(output.clone(), tensor(&data["output"], device), 1e-5);
    let grads = (output * tensor::<3>(&data["loss_weights"], device))
        .sum()
        .backward();
    close(
        input.grad(&grads).unwrap(),
        tensor(&data["input_gradient"], &Device::flex()),
        8e-5,
    );
    let mut check = Transfer {
        data: &data["parameter_gradients"],
        device: &Device::flex(),
        gradients: Some(&grads),
        count: 0,
    };
    check.model(&mut model);
    assert_eq!(check.count, load.count);
}

#[test]
fn reference_stack_streaming_and_chunking_match_full_sequence() {
    let device = Device::flex();
    let data = &fixture()["stack"];
    let mut model = reference_model(&device);
    Transfer {
        data: &data["weights"],
        device: &device,
        gradients: None,
        count: 0,
    }
    .model(&mut model);
    let input = tensor::<3>(&data["input"], &device);
    let expected = tensor::<3>(&data["output"], &device);
    let mut state = None;
    let mut outputs = Vec::new();
    for time in 0..5 {
        let (output, next) = model.step(input.clone().slice([0..2, time..time + 1, 0..8]), state);
        outputs.push(output);
        state = Some(next);
    }
    close(Tensor::cat(outputs, 1), expected.clone(), 2e-5);
    let (first, state) = model.forward_with_state(input.clone().slice([0..2, 0..2, 0..8]), None);
    let (rest, _) = model.forward_with_state(input.slice([0..2, 2..5, 0..8]), Some(state));
    close(Tensor::cat(vec![first, rest], 1), expected, 2e-5);
}

#[test]
fn randomized_shapes_outputs_and_gradients_match_nxai() {
    shape_sweep(&Device::flex().autodiff());
}

fn shape_sweep(device: &Device) {
    let fixtures: Value = serde_json::from_str(include_str!("fixtures/nxai_sweep.json")).unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        let number = |key: &str| usize::try_from(case[key].as_u64().unwrap()).unwrap();
        let kinds = case["kinds"]
            .as_str()
            .unwrap()
            .chars()
            .map(|kind| match kind {
                'm' => BlockKind::Mlstm,
                's' => BlockKind::Slstm,
                _ => panic!("unknown block kind"),
            })
            .collect();
        let config = XlstmConfig::new(number("width"), kinds)
            .with_num_heads(number("heads"))
            .with_mlstm_inner_dim(Some(2 * number("width")))
            .with_feedforward_dim(Some(3 * number("width") / 2))
            .with_qkv_block_size(number("group"))
            .with_conv_kernel(number("kernel"));
        for size in [None, Some(3)] {
            for chunked in [false, true] {
                check_stack(
                    config
                        .clone()
                        .with_mlstm_chunk_size(size)
                        .init(device)
                        .unwrap(),
                    case,
                    device,
                    chunked,
                );
            }
        }
    }
}
