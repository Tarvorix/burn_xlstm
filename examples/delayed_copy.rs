//! Train a small mixed xLSTM to reproduce the previous input token.
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    tensor::{Device, Int, Tensor},
};
use burn_xlstm::{
    lm::XlstmLmConfig,
    stack::{BlockKind, XlstmConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::flex().autodiff();
    device.seed(42);
    let stack = XlstmConfig::new(8, vec![BlockKind::Mlstm, BlockKind::Slstm])
        .with_num_heads(2)
        .with_mlstm_inner_dim(Some(16))
        .with_feedforward_dim(Some(12))
        .with_conv_kernel(3);
    let mut model = XlstmLmConfig::new(4, stack).init(&device)?;
    let input = Tensor::<2, Int>::from_data(
        [
            [0, 1, 3, 2, 1, 0],
            [2, 1, 0, 3, 2, 3],
            [3, 0, 2, 1, 3, 1],
            [1, 2, 3, 0, 0, 2],
        ],
        &device,
    );
    // Predict the previous token. Position zero uses a fixed zero target.
    let target = Tensor::<2, Int>::from_data(
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
    let mut optimizer = AdamConfig::new().init();
    for step in 0..=80 {
        let logits = model.forward(input.clone()).reshape([24, 4]);
        let loss = loss_fn.forward(logits, target.clone());
        if step % 20 == 0 {
            println!(
                "step {step:3}: loss {:.6}",
                loss.clone().into_scalar::<f32>()
            );
        }
        if step < 80 {
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.02, model, grads);
        }
    }
    Ok(())
}
