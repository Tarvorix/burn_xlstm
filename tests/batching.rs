//! Padding and packed execution must equal independent sequence processing.
use burn::{
    module::Param,
    tensor::{Bool, Device, Distribution, Tensor, TensorData},
};
use burn_xlstm::{
    batch::{LmBatch, SequenceMask, masked_cross_entropy},
    lm::XlstmLmConfig,
    stack::{BlockKind, MemoryLayer, Xlstm, XlstmConfig},
};

fn config() -> XlstmConfig {
    XlstmConfig::new(8, vec![BlockKind::Mlstm, BlockKind::Slstm])
        .with_num_heads(2)
        .with_mlstm_inner_dim(Some(16))
        .with_feedforward_dim(Some(12))
        .with_conv_kernel(3)
}

// Nonzero recurrent weights ensure hidden-state masking affects later tokens.
fn mix_memory(model: &mut Xlstm, device: &Device) {
    for block in &mut model.blocks {
        if let MemoryLayer::Slstm(layer) = &mut block.memory {
            for recurrent in &mut layer.cell.recurrent {
                recurrent.weight = Param::from_tensor(Tensor::random(
                    recurrent.weight.val().dims(),
                    Distribution::Normal(0.0, 0.2),
                    device,
                ));
            }
        }
    }
}

fn close<const D: usize>(a: Tensor<D>, b: Tensor<D>) {
    assert_eq!(a.dims(), b.dims());
    let difference = (a - b).abs().max().into_scalar::<f32>();
    assert!(
        difference.is_finite() && difference < 3e-5,
        "difference {difference}"
    );
}

fn input(device: &Device) -> Tensor<3> {
    let data: Vec<f32> = (0..120)
        .map(|i| (f32::from(i16::try_from(i).unwrap()) * 0.13).sin())
        .collect();
    Tensor::from_data(TensorData::new(data, [3, 5, 8]), device).require_grad()
}

#[test]
fn padding_preserves_outputs_gradients_and_continuation_for_each_lane() {
    check_padding(&Device::flex().autodiff());
}

fn check_padding(device: &Device) {
    let mut model = config().init(device).unwrap();
    mix_memory(&mut model, device);
    let input = input(device);
    let mask = SequenceMask::from_lengths(&[5, 2, 0], 5, device).unwrap();
    let padded = input
        .clone()
        .mask_fill(mask.valid.clone().bool_not().reshape([3, 5, 1]), f64::NAN);
    let (output, state) = model.forward_masked(padded, &mask, None);
    let continuation = Tensor::<3>::full([3, 1, 8], 0.13, device);
    let (continued, _) = model.step(continuation.clone(), Some(state));
    let mut expected_loss = Tensor::<1>::zeros([1], device);
    for (batch, length) in [5, 2, 0].into_iter().enumerate() {
        let (expected, state) = if length > 0 {
            let (y, state) =
                model.forward_with_state(input.clone().slice([batch..batch + 1, 0..length]), None);
            expected_loss = expected_loss + y.clone().square().sum();
            (Some(y), Some(state))
        } else {
            (None, None)
        };
        if let Some(expected) = expected {
            close(
                output.clone().slice([batch..batch + 1, 0..length]),
                expected,
            );
        }
        if length < 5 {
            close(
                output.clone().slice([batch..batch + 1, length..5]),
                Tensor::zeros([1, 5 - length, 8], device),
            );
        }
        let (expected, _) = model.step(
            continuation.clone().slice([batch..batch + 1, 0..1, 0..8]),
            state,
        );
        close(
            continued.clone().slice([batch..batch + 1, 0..1, 0..8]),
            expected,
        );
    }
    let actual_gradients = output.square().sum().backward();
    let expected_gradients = expected_loss.backward();
    close(
        input.grad(&actual_gradients).unwrap(),
        input.grad(&expected_gradients).unwrap(),
    );
}

#[test]
fn mask_holes_and_resets_do_not_advance_convolution_or_leak_sequence_history() {
    let device = Device::flex().autodiff();
    let mut model = config().init(&device).unwrap();
    mix_memory(&mut model, &device);
    let input = input(&device)
        .slice([0..1, 0..5, 0..8])
        .detach()
        .require_grad();
    let mask = SequenceMask {
        valid: Tensor::<2, Bool>::from_bool([[true, false, true, true, true]], &device),
        reset: Tensor::<2, Bool>::from_bool([[false, true, false, true, false]], &device),
    };
    let (output, _) = model.forward_masked(input.clone(), &mask, None);
    let first = Tensor::cat(
        vec![
            input.clone().slice([0..1, 0..1]),
            input.clone().slice([0..1, 2..3]),
        ],
        1,
    );
    let expected = model.forward_with_state(first, None).0;
    close(
        output.clone().slice([0..1, 0..1]),
        expected.clone().slice([0..1, 0..1]),
    );
    close(
        output.clone().slice([0..1, 2..3]),
        expected.slice([0..1, 1..2]),
    );
    let second = input.clone().slice([0..1, 3..5]);
    close(
        output.clone().slice([0..1, 3..5]),
        model.forward_with_state(second, None).0,
    );
    let gradients = output.slice([0..1, 3..5]).square().sum().backward();
    let before_reset = input.grad(&gradients).unwrap().slice([0..1, 0..3]);
    close(before_reset, Tensor::zeros([1, 3, 8], &Device::flex()));
}

#[test]
fn packed_and_padded_language_batches_match_loss_and_parameter_gradients() {
    check_packed(&Device::flex().autodiff());
}

fn check_packed(device: &Device) {
    let mut model = XlstmLmConfig::new(6, config()).init(device).unwrap();
    mix_memory(&mut model.stack, device);
    let sequences = vec![
        vec![1, 2, 3, 4],
        vec![5, 4],
        vec![3, 1, 5, 2],
        vec![2],
        vec![],
    ];
    let padded = LmBatch::padded(&sequences, device).unwrap();
    let packed = LmBatch::packed(&sequences, 5, device).unwrap();
    assert_eq!(padded.inputs.dims(), [5, 3]);
    assert_eq!(packed.inputs.dims(), [2, 5]);
    let logits = model
        .forward_masked(
            padded
                .inputs
                .clone()
                .mask_fill(padded.mask.valid.clone().bool_not(), -999),
            &padded.mask,
            None,
        )
        .0;
    let loss = masked_cross_entropy(
        logits,
        padded
            .targets
            .clone()
            .mask_fill(padded.mask.valid.clone().bool_not(), -100),
        padded.mask.valid.clone(),
    );
    let packed_loss = packed.loss(
        model
            .forward_masked(packed.inputs.clone(), &packed.mask, None)
            .0,
    );
    close(loss.clone(), packed_loss.clone());
    let grads = loss.backward();
    let packed_grads = packed_loss.backward();
    close(
        model.embedding.weight.val().grad(&grads).unwrap(),
        model.embedding.weight.val().grad(&packed_grads).unwrap(),
    );
    close(
        model.head.weight.val().grad(&grads).unwrap(),
        model.head.weight.val().grad(&packed_grads).unwrap(),
    );
}

#[test]
fn entirely_padded_loss_is_finite_zero_with_zero_gradients() {
    let device = Device::flex().autodiff();
    let logits = Tensor::<3>::full([2, 3, 4], f64::NAN, &device).require_grad();
    let targets = Tensor::full([2, 3], -999, &device);
    let valid = Tensor::<2, Bool>::zeros([2, 3], &device);
    let loss = masked_cross_entropy(logits.clone(), targets, valid);
    close(loss.clone(), Tensor::zeros([1], &device));
    close(
        logits.grad(&loss.backward()).unwrap(),
        Tensor::zeros([2, 3, 4], &Device::flex()),
    );
}

#[test]
fn invalid_batch_inputs_return_errors() {
    let device = Device::flex();
    assert_eq!(
        SequenceMask::from_lengths(&[3], 2, &device)
            .unwrap_err()
            .field(),
        "lengths"
    );
    assert_eq!(
        LmBatch::padded(&[vec![1]], &device).unwrap_err().field(),
        "sequences"
    );
    assert_eq!(
        LmBatch::packed(&[vec![1, 2, 3, 4]], 2, &device)
            .unwrap_err()
            .field(),
        "row_width"
    );
    assert_eq!(
        LmBatch::padded(&[vec![1, -1]], &device)
            .unwrap_err()
            .field(),
        "tokens"
    );
    assert_eq!(
        config()
            .with_mlstm_chunk_size(Some(0))
            .init(&device)
            .unwrap_err()
            .field(),
        "chunk_size"
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a working GPU"]
fn wgpu_padded_and_packed_outputs_and_gradients() {
    let mut device = Device::wgpu(burn::tensor::DeviceKind::default());
    device.configure(burn::tensor::FloatDType::F32).unwrap();
    let device = device.autodiff();
    check_padding(&device);
    check_packed(&device);
}
