use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    store::ModuleRecord,
    tensor::{Device, Distribution, Int, Tensor, TensorData},
};
use burn_xlstm::{
    functional::{MlstmInput, MlstmState, SlstmGates, SlstmState},
    layers::{MlstmConfig, SlstmConfig},
    lm::XlstmLmConfig,
    stack::{BlockKind, MemoryLayer, XlstmConfig},
};

fn finite<const D: usize>(tensor: Tensor<D>) {
    assert!(
        tensor
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
}

fn near<const D: usize>(a: Tensor<D>, b: Tensor<D>, tolerance: f32) {
    let difference = (a - b).abs().max().into_scalar::<f32>();
    assert!(
        difference.is_finite() && difference <= tolerance,
        "difference {difference}"
    );
}

fn config() -> XlstmConfig {
    XlstmConfig::new(8, vec![BlockKind::Mlstm, BlockKind::Slstm])
        .with_num_heads(2)
        .with_mlstm_inner_dim(Some(16))
        .with_feedforward_dim(Some(12))
        .with_conv_kernel(3)
}

#[test]
fn scalar_extreme_first_gate_has_finite_output_and_gradients() {
    let device = Device::flex().autodiff();
    for logit in [-1000.0, 1000.0] {
        let input = Tensor::full([1, 2, 3], logit, &device).require_grad();
        let candidate = Tensor::full([1, 2, 3], 0.5, &device).require_grad();
        let gates = SlstmGates {
            input: input.clone(),
            forget: Tensor::full([1, 2, 3], 1000.0, &device),
            candidate: candidate.clone(),
            output: Tensor::zeros([1, 2, 3], &device),
        };
        let state = gates.step(SlstmState::zeros([1, 2, 3], &device));
        near(state.hidden.clone(), candidate.clone().tanh() * 0.5, 1e-7);
        let grads = state.hidden.sum().backward();
        finite(input.grad(&grads).unwrap());
        finite(candidate.grad(&grads).unwrap());
    }
}

#[test]
fn matrix_extreme_logits_and_cancellation_stay_finite() {
    let device = Device::flex().autodiff();
    for logit in [-1000.0, 1000.0] {
        let input = Tensor::full([1, 1, 4, 1], logit, &device).require_grad();
        let query = Tensor::from_data(
            TensorData::new([1.0_f32, -1.0].repeat(4), [1, 1, 4, 2]),
            &device,
        )
        .require_grad();
        let data = MlstmInput {
            query: query.clone(),
            key: Tensor::ones([1, 1, 4, 2], &device),
            value: Tensor::ones([1, 1, 4, 2], &device),
            input: input.clone(),
            forget: Tensor::full([1, 1, 4, 1], -1000.0, &device),
        };
        let output = data.clone().parallel(1e-6);
        finite(output.clone());
        let grads = output.sum().backward();
        finite(input.grad(&grads).unwrap());
        finite(query.grad(&grads).unwrap());
        let mut state = MlstmState::zeros([1, 1, 2], &device);
        for time in 0..4 {
            let take = |x: &Tensor<4>| x.clone().slice([0..1, 0..1, time..time + 1]);
            let (output, next) = MlstmInput {
                query: take(&data.query),
                key: take(&data.key),
                value: take(&data.value),
                input: take(&data.input),
                forget: take(&data.forget),
            }
            .step(state, 1e-6);
            finite(output);
            finite(next.memory.clone());
            state = next;
        }
    }
}

#[test]
fn long_scalar_recurrence_remains_finite() {
    let device = Device::flex();
    let mut state = SlstmState::zeros([1, 2, 3], &device);
    for time in 0..1024 {
        let logit = if time % 2 == 0 { 1000.0 } else { -1000.0 };
        state = SlstmGates {
            input: Tensor::full([1, 2, 3], logit, &device),
            forget: Tensor::full([1, 2, 3], 1000.0, &device),
            candidate: Tensor::full([1, 2, 3], 0.2, &device),
            output: Tensor::zeros([1, 2, 3], &device),
        }
        .step(state);
    }
    near(
        state.hidden,
        Tensor::<3>::full([1, 2, 3], 0.2, &device).tanh() * 0.5,
        1e-6,
    );
    finite(state.memory);
    finite(state.normalizer);
}

#[test]
fn changing_future_tokens_cannot_change_prefix() {
    let device = Device::flex();
    let model = config().init(&device).unwrap();
    let input = Tensor::random([2, 7, 8], Distribution::Normal(0.0, 1.0), &device);
    let changed = input
        .clone()
        .slice_assign([0..2, 4..7, 0..8], Tensor::full([2, 3, 8], 50.0, &device));
    let original = model.forward(input).slice([0..2, 0..4, 0..8]);
    let changed = model.forward(changed).slice([0..2, 0..4, 0..8]);
    near(original, changed, 1e-6);
}

#[test]
fn convolution_zero_one_and_longer_than_sequence_stream_correctly() {
    let device = Device::flex();
    for kernel in [0, 1, 7] {
        let model = config().with_conv_kernel(kernel).init(&device).unwrap();
        let input = Tensor::random([2, 3, 8], Distribution::Normal(0.0, 1.0), &device);
        let expected = model.forward_with_state(input.clone(), None).0;
        let mut outputs = Vec::new();
        let mut state = None;
        for time in 0..3 {
            let (output, next) =
                model.step(input.clone().slice([0..2, time..time + 1, 0..8]), state);
            outputs.push(output);
            state = Some(next);
        }
        near(Tensor::cat(outputs, 1), expected, 2e-5);
    }
}

#[test]
fn checkpoint_bytes_roundtrip_preserves_outputs() {
    let device = Device::flex();
    let model = config().init(&device).unwrap();
    let input = Tensor::random([2, 5, 8], Distribution::Normal(0.0, 1.0), &device);
    let expected = model.forward(input.clone());
    let bytes = model.into_record().into_bytes().unwrap();
    let loaded = config()
        .init(&device)
        .unwrap()
        .load_record(ModuleRecord::from_bytes(bytes).unwrap());
    near(loaded.forward(input), expected, 1e-6);
}

#[test]
fn detach_breaks_gradients_across_chunk_boundary() {
    let device = Device::flex().autodiff();
    let model = config().init(&device).unwrap();
    let first = Tensor::random([1, 3, 8], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let second = Tensor::random([1, 2, 8], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let (_, state) = model.forward_with_state(first.clone(), None);
    let (connected, _) = model.forward_with_state(second.clone(), Some(state.clone()));
    let grads = connected.square().sum().backward();
    assert!(
        first.grad(&grads).is_some(),
        "state should preserve the graph by default"
    );
    let (detached, _) = model.forward_with_state(second.clone(), Some(state.detach()));
    let grads = detached.square().sum().backward();
    assert!(
        first.grad(&grads).is_none(),
        "detach must cut convolution and memory graphs"
    );
    finite(second.grad(&grads).unwrap());
}

#[test]
fn invalid_configuration_returns_useful_errors() {
    let device = Device::flex();
    assert_eq!(
        SlstmConfig::new(7).init(&device).unwrap_err().field(),
        "num_heads"
    );
    assert_eq!(
        MlstmConfig::new(8)
            .with_inner_dim(Some(10))
            .init(&device)
            .unwrap_err()
            .field(),
        "num_heads"
    );
    assert_eq!(
        MlstmConfig::new(8)
            .with_qkv_block_size(0)
            .init(&device)
            .unwrap_err()
            .field(),
        "qkv_block_size"
    );
    assert_eq!(
        MlstmConfig::new(8)
            .with_epsilon(f64::NAN)
            .init(&device)
            .unwrap_err()
            .field(),
        "epsilon"
    );
    assert_eq!(
        config()
            .with_dropout(1.0)
            .init(&device)
            .unwrap_err()
            .field(),
        "dropout"
    );
    assert_eq!(
        XlstmConfig::new(8, vec![])
            .init(&device)
            .unwrap_err()
            .field(),
        "num_blocks"
    );
    assert_eq!(
        SlstmConfig::new(8)
            .with_block_index(2)
            .init(&device)
            .unwrap_err()
            .field(),
        "num_blocks"
    );
}

#[test]
fn one_coordinate_scalar_heads_have_finite_initialization() {
    let device = Device::flex();
    let layer = SlstmConfig::new(4).init(&device).unwrap();
    finite(layer.cell.bias.val());
    finite(layer.forward(Tensor::ones([1, 2, 4], &device)));
}

#[test]
#[should_panic(expected = "stack state depth mismatch")]
fn truncated_state_is_rejected() {
    let device = Device::flex();
    let model = config().init(&device).unwrap();
    let input = Tensor::ones([1, 1, 8], &device);
    let (_, mut state) = model.step(input.clone(), None);
    state.blocks.pop();
    let _ = model.step(input, Some(state));
}

#[test]
fn language_model_streaming_matches_parallel_logits() {
    let device = Device::flex();
    let model = XlstmLmConfig::new(7, config()).init(&device).unwrap();
    let tokens = Tensor::<2, Int>::from_data([[1, 3, 4, 2], [2, 6, 0, 1]], &device);
    let full = model.forward(tokens.clone());
    let mut state = None;
    let mut outputs = Vec::new();
    for t in 0..4 {
        let (output, next) = model.step(tokens.clone().slice([0..2, t..t + 1]), state);
        outputs.push(output);
        state = Some(next);
    }
    near(full, Tensor::cat(outputs, 1), 2e-5);
}

#[test]
fn adam_learns_delayed_token_copy_through_both_memory_types() {
    let device = Device::flex().autodiff();
    device.seed(42);
    let mut model = XlstmLmConfig::new(4, config()).init(&device).unwrap();
    let tokens = Tensor::<2, Int>::from_data(
        [
            [0, 1, 3, 2, 1, 0],
            [2, 1, 0, 3, 2, 3],
            [3, 0, 2, 1, 3, 1],
            [1, 2, 3, 0, 0, 2],
        ],
        &device,
    );
    let targets = Tensor::<2, Int>::from_data(
        [
            [0, 0, 1, 3, 2, 1],
            [0, 2, 1, 0, 3, 2],
            [0, 3, 0, 2, 1, 3],
            [0, 1, 2, 3, 0, 0],
        ],
        &device,
    )
    .reshape([24]);
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let initial = loss_fn
        .forward(
            model.forward(tokens.clone()).reshape([24, 4]),
            targets.clone(),
        )
        .into_scalar::<f32>();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..80 {
        let loss = loss_fn.forward(
            model.forward(tokens.clone()).reshape([24, 4]),
            targets.clone(),
        );
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, grads);
    }
    let final_loss = loss_fn
        .forward(model.forward(tokens).reshape([24, 4]), targets)
        .into_scalar::<f32>();
    assert!(
        final_loss < initial * 0.4,
        "loss did not fall enough: {initial} -> {final_loss}"
    );
    let MemoryLayer::Slstm(layer) = &model.stack.blocks[1].memory else {
        panic!("expected sLSTM")
    };
    assert!(
        layer.cell.recurrent[0]
            .weight
            .val()
            .abs()
            .sum()
            .into_scalar::<f32>()
            > 0.01,
        "zero-initialized recurrent weights must train"
    );
}
