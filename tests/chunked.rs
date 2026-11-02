//! Chunkwise execution against upstream recurrent outputs, states, and derivatives.
mod common;
use burn::tensor::{Device, Tensor};
use burn_xlstm::functional::{MlstmInput, MlstmState};
use common::{close, fixture, tensor};

fn check(device: &Device) {
    let fixture = fixture();
    let data = &fixture["mlstm"];
    for chunk in [1, 2, 3, 5, 16] {
        let input = MlstmInput {
            query: tensor(&data["query"], device).require_grad(),
            key: tensor(&data["key"], device).require_grad(),
            value: tensor(&data["value"], device).require_grad(),
            input: tensor(&data["input"], device).require_grad(),
            forget: tensor(&data["forget"], device).require_grad(),
        };
        let (output, state) = input.clone().chunked(None, chunk, 1e-6);
        close(output.clone(), tensor(&data["recurrent"], device), 3e-6);
        close(state.memory, tensor(&data["memory"], device), 3e-6);
        close(state.normalizer, tensor(&data["normalizer"], device), 3e-6);
        close(state.log_scale, tensor(&data["log_scale"], device), 3e-6);
        let grads = (output * tensor::<4>(&data["loss_weights"], device))
            .sum()
            .backward();
        for (name, param) in [
            ("query", &input.query),
            ("key", &input.key),
            ("value", &input.value),
            ("input", &input.input),
            ("forget", &input.forget),
        ] {
            close(
                param.grad(&grads).unwrap(),
                tensor(&data["recurrent_gradients"][name], &Device::flex()),
                3e-5,
            );
        }
    }
}

#[test]
fn chunked_outputs_states_and_gradients_match_nxai() {
    check(&Device::flex().autodiff());
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a working GPU"]
fn wgpu_chunked_outputs_states_and_gradients_match_nxai() {
    let mut device = Device::wgpu(burn::tensor::DeviceKind::default());
    device.configure(burn::tensor::FloatDType::F32).unwrap();
    check(&device.autodiff());
}

#[test]
fn chunked_continuation_and_initial_state_gradients_match_steps() {
    let device = Device::flex().autodiff();
    let fixture = fixture();
    let data = &fixture["mlstm"];
    let input = MlstmInput {
        query: tensor(&data["query"], &device),
        key: tensor(&data["key"], &device),
        value: tensor(&data["value"], &device),
        input: tensor(&data["input"], &device),
        forget: tensor(&data["forget"], &device),
    };
    let state = MlstmState {
        memory: Tensor::full([2, 2, 3, 3], 0.2, &device).require_grad(),
        normalizer: Tensor::full([2, 2, 3, 1], -0.3, &device).require_grad(),
        log_scale: Tensor::full([2, 2, 1, 1], 0.7, &device).require_grad(),
    };
    let (chunked, next) = input.clone().chunked(Some(state.clone()), 2, 1e-6);
    let mut recurrent = state.clone();
    let mut outputs = Vec::new();
    for t in 0..5 {
        let take = |x: &Tensor<4>| x.clone().slice([0..2, 0..2, t..t + 1]);
        let (output, next) = MlstmInput {
            query: take(&input.query),
            key: take(&input.key),
            value: take(&input.value),
            input: take(&input.input),
            forget: take(&input.forget),
        }
        .step(recurrent, 1e-6);
        outputs.push(output);
        recurrent = next;
    }
    close(chunked.clone(), Tensor::cat(outputs.clone(), 2), 3e-6);
    close(next.memory.clone(), recurrent.memory.clone(), 3e-6);
    close(next.normalizer.clone(), recurrent.normalizer.clone(), 3e-6);
    close(next.log_scale.clone(), recurrent.log_scale.clone(), 3e-6);
    let chunk_grads =
        (chunked.sum() + next.memory.sum() + next.normalizer.sum() + next.log_scale.sum())
            .backward();
    let recurrent_grads = (Tensor::cat(outputs, 2).sum()
        + recurrent.memory.sum()
        + recurrent.normalizer.sum()
        + recurrent.log_scale.sum())
    .backward();
    for param in [&state.memory, &state.normalizer, &state.log_scale] {
        close(
            param.grad(&chunk_grads).unwrap(),
            param.grad(&recurrent_grads).unwrap(),
            3e-5,
        );
    }
}
