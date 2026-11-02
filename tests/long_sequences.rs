//! Long-horizon stability, continuation, and independent batch/head lanes.
use burn::{
    module::Module,
    tensor::{Device, Tensor, TensorData},
};
use burn_xlstm::{
    functional::{MlstmInput, MlstmState},
    stack::{BlockKind, BlockState, XlstmConfig, XlstmState},
};

#[expect(
    clippy::cast_possible_truncation,
    reason = "bounded deterministic test inputs round to f32"
)]
fn signal<const D: usize>(shape: [usize; D], phase: f64, device: &Device) -> Tensor<D> {
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|i| {
            (f64::from(u32::try_from(i).unwrap())
                .mul_add(0.719, phase)
                .sin()
                * 0.8) as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn close<const D: usize>(actual: Tensor<D>, expected: Tensor<D>, tolerance: f32) {
    assert_eq!(actual.dims(), expected.dims());
    let a = actual.into_data().try_to_vec::<f32>().unwrap();
    let e = expected.into_data().try_to_vec::<f32>().unwrap();
    let mut worst = 0.0_f32;
    for (a, e) in a.into_iter().zip(e) {
        assert!(a.is_finite() && e.is_finite());
        worst = worst.max((a - e).abs() / (1.0 + e.abs()));
    }
    assert!(
        worst <= tolerance,
        "worst scaled error {worst}, tolerance {tolerance}"
    );
}

fn input(time: usize, device: &Device) -> MlstmInput {
    MlstmInput {
        query: signal([2, 2, time, 3], 0.1, device),
        key: signal([2, 2, time, 3], 1.1, device),
        value: signal([2, 2, time, 3], 2.1, device),
        input: signal([2, 2, time, 1], 3.1, device),
        forget: (signal([2, 2, time, 1], 4.1, device) + 2.0),
    }
}

fn recurrent(data: &MlstmInput) -> Tensor<4> {
    let [batch, heads, time, dim] = data.query.dims();
    let mut state = MlstmState::zeros([batch, heads, dim], &data.query.device());
    let mut outputs = Vec::with_capacity(time);
    for t in 0..time {
        let take = |x: &Tensor<4>| x.clone().slice([0..batch, 0..heads, t..t + 1]);
        let (output, next) = MlstmInput {
            query: take(&data.query),
            key: take(&data.key),
            value: take(&data.value),
            input: take(&data.input),
            forget: take(&data.forget),
        }
        .step(state, 1e-6);
        outputs.push(output);
        state = next;
    }
    Tensor::cat(outputs, 2)
}

#[test]
fn long_matrix_sequences_and_gradients_agree_between_execution_paths() {
    // Float32 cumsum differences in the parallel path accumulate rounding with
    // time; this explicit tolerance includes that error, not only one-step error.
    let device = Device::flex();
    for time in [127, 512, 2048] {
        let data = input(time, &device);
        close(data.clone().parallel(1e-6), recurrent(&data), 2e-4);
    }
    let device = device.autodiff();
    let mut data = input(257, &device);
    data.query = data.query.require_grad();
    data.key = data.key.require_grad();
    data.value = data.value.require_grad();
    data.input = data.input.require_grad();
    data.forget = data.forget.require_grad();
    let weights = signal([2, 2, 257, 3], 5.1, &device);
    let parallel = (data.clone().parallel(1e-6) * weights.clone())
        .sum()
        .backward();
    let recurrent = (recurrent(&data) * weights).sum().backward();
    for tensor in [
        &data.query,
        &data.key,
        &data.value,
        &data.input,
        &data.forget,
    ] {
        close(
            tensor.grad(&parallel).unwrap(),
            tensor.grad(&recurrent).unwrap(),
            2e-4,
        );
    }
}

fn states_close(left: XlstmState, right: XlstmState) {
    assert_eq!(left.blocks.len(), right.blocks.len());
    for (left, right) in left.blocks.into_iter().zip(right.blocks) {
        let (left_history, right_history) = match (left, right) {
            (BlockState::Mlstm(a), BlockState::Mlstm(b)) => {
                close(a.cell.memory, b.cell.memory, 2e-5);
                close(a.cell.normalizer, b.cell.normalizer, 2e-5);
                close(a.cell.log_scale, b.cell.log_scale, 2e-5);
                (a.conv, b.conv)
            }
            (BlockState::Slstm(a), BlockState::Slstm(b)) => {
                close(a.cell.hidden, b.cell.hidden, 2e-5);
                close(a.cell.memory, b.cell.memory, 2e-5);
                close(a.cell.normalizer, b.cell.normalizer, 2e-5);
                close(a.cell.log_scale, b.cell.log_scale, 2e-5);
                (a.conv, b.conv)
            }
            _ => panic!("state variant mismatch"),
        };
        match (left_history, right_history) {
            (Some(a), Some(b)) => close(a, b, 2e-5),
            (None, None) => (),
            _ => panic!("convolution history mismatch"),
        }
    }
}

#[test]
fn long_prefill_chunking_and_continuation_preserve_every_state() {
    let device = Device::flex();
    device.seed(811);
    let model = XlstmConfig::new(8, vec![BlockKind::Mlstm, BlockKind::Slstm])
        .with_num_heads(2)
        .with_mlstm_inner_dim(Some(16))
        .with_feedforward_dim(Some(12))
        .with_conv_kernel(7)
        .init(&device)
        .unwrap();
    let input = signal([2, 257, 8], 0.2, &device);
    let (expected, expected_state) = model.forward_with_state(input.clone(), None);
    close(model.forward(input.clone()), expected.clone(), 2e-5);
    let mut state = None;
    let mut outputs = Vec::new();
    let mut start = 0;
    for end in [1, 2, 7, 65, 257] {
        let (output, next) =
            model.forward_with_state(input.clone().slice([0..2, start..end]), state);
        outputs.push(output);
        state = Some(next);
        start = end;
    }
    close(Tensor::cat(outputs, 1), expected, 2e-5);
    states_close(state.clone().unwrap(), expected_state.clone());
    let token = signal([2, 1, 8], 1.2, &device);
    let (a, a_state) = model.step(token.clone(), state);
    let (b, b_state) = model.step(token, Some(expected_state));
    close(a, b, 2e-5);
    states_close(a_state, b_state);
}

#[test]
fn matrix_batches_and_heads_are_independent() {
    let device = Device::flex();
    let data = input(37, &device);
    let combined = data.clone().parallel(1e-6);
    for batch in 0..2 {
        for head in 0..2 {
            let lane = |x: &Tensor<4>| x.clone().slice([batch..batch + 1, head..head + 1]);
            let isolated = MlstmInput {
                query: lane(&data.query),
                key: lane(&data.key),
                value: lane(&data.value),
                input: lane(&data.input),
                forget: lane(&data.forget),
            };
            close(isolated.clone().parallel(1e-6), lane(&combined), 1e-6);
            close(recurrent(&isolated), lane(&combined), 2e-5);
        }
    }
}

#[test]
fn validation_mode_disables_dropout_for_streaming() {
    let device = Device::flex().autodiff();
    let model = XlstmConfig::new(8, vec![BlockKind::Mlstm, BlockKind::Slstm])
        .with_num_heads(2)
        .with_mlstm_inner_dim(Some(16))
        .with_feedforward_dim(Some(12))
        .with_dropout(0.5)
        .init(&device)
        .unwrap();
    let input = signal([2, 13, 8], 0.3, &device);
    let training_difference = (model.forward(input.clone()) - model.forward(input.clone()))
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(
        training_difference > 1e-3,
        "training dropout must be active"
    );
    let model = model.valid();
    close(
        model.forward(input.clone()),
        model.forward(input.clone()),
        0.0,
    );
    close(
        model.forward(input.clone()),
        model.forward_with_state(input, None).0,
        2e-5,
    );
}
