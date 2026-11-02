//! Learn unseen delayed-copy sequences with convolution disabled.
use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, Int, Tensor, TensorData},
};
use burn_xlstm::{
    lm::{XlstmLm, XlstmLmConfig},
    stack::{BlockKind, XlstmConfig},
};

// Dataset randomness is independent of Burn's global model initializer RNG.
struct Tokens(u64);

impl Tokens {
    fn batch(
        &mut self,
        batch: usize,
        time: usize,
        device: &Device,
    ) -> (Tensor<2, Int>, Tensor<1, Int>) {
        let mut inputs = Vec::with_capacity(batch * time);
        let mut targets = Vec::with_capacity(batch * time);
        for _ in 0..batch {
            let mut previous = 0;
            for _ in 0..time {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                let token = i32::try_from(self.0 % 4).unwrap();
                inputs.push(token);
                targets.push(previous);
                previous = token;
            }
        }
        (
            Tensor::from_data(TensorData::new(inputs, [batch, time]), device),
            Tensor::from_data(TensorData::new(targets, [batch * time]), device),
        )
    }
}

fn evaluate(model: &XlstmLm, device: &Device, time: usize, seed: u64) -> (f32, f32) {
    let (inputs, targets) = Tokens(seed).batch(64, time, device);
    let logits = model.valid().forward(inputs).reshape([64 * time, 4]);
    let correct = logits
        .clone()
        .argmax(1)
        .reshape([64 * time])
        .equal(targets.clone())
        .int()
        .sum()
        .into_scalar::<i64>();
    let loss = CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, targets)
        .into_scalar::<f32>();
    let accuracy =
        f32::from(u16::try_from(correct).unwrap()) / f32::from(u16::try_from(64 * time).unwrap());
    (loss, accuracy)
}

#[test]
fn all_architectures_learn_held_out_copy_without_convolution() {
    let device = Device::flex().autodiff();
    for (name, blocks) in [
        ("scalar", vec![BlockKind::Slstm]),
        ("matrix", vec![BlockKind::Mlstm, BlockKind::Mlstm]),
        ("mixed", vec![BlockKind::Mlstm, BlockKind::Slstm]),
    ] {
        device.seed(1729);
        let config = XlstmConfig::new(16, blocks)
            .with_num_heads(2)
            .with_mlstm_inner_dim(Some(32))
            .with_mlstm_chunk_size(Some(4))
            .with_feedforward_dim(Some(24))
            .with_conv_kernel(0);
        let mut model = XlstmLmConfig::new(4, config).init(&device).unwrap();
        let (initial, _) = evaluate(&model, &device, 12, 0xa66d_8c51_3390_ef27);
        let mut training = Tokens(0x1234_5678_9abc_def0);
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let mut optimizer = AdamConfig::new().init();
        for _ in 0..400 {
            let (inputs, targets) = training.batch(16, 12, &device);
            let loss = loss_fn.forward(model.forward(inputs).reshape([16 * 12, 4]), targets);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.003, model, grads);
        }
        // These two evaluation streams were not used to select the training settings.
        for (time, seed) in [
            (12, 0xa66d_8c51_3390_ef27),
            (24, 0xa66d_8c51_3390_ef27),
            (12, 0x509b_f17c_d3a8_2261),
            (24, 0x509b_f17c_d3a8_2261),
        ] {
            let (loss, accuracy) = evaluate(&model, &device, time, seed);
            eprintln!(
                "{name}, seed {seed:x}, unseen length {time}: loss {initial:.6} -> {loss:.6}, accuracy {accuracy:.4}"
            );
            assert!(
                loss.is_finite() && loss < initial * 0.15,
                "{name}: held-out loss {loss}"
            );
            assert!(accuracy > 0.95, "{name}: held-out accuracy {accuracy}");
        }
    }
}
