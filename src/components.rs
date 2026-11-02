//! Projection, normalization, and convolution components used by xLSTM blocks.

use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Device, Distribution, Tensor},
};

pub(crate) fn size_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("model dimension exceeds u32"))
}

pub(crate) fn small_init(dim: usize) -> Initializer {
    // NX-AI's Nguyen/Salazar initialization.
    Initializer::Normal {
        mean: 0.0,
        std: (2.0 / (5.0 * size_f64(dim))).sqrt(),
    }
}

pub(crate) fn output_init(dim: usize, blocks: usize) -> Initializer {
    // NX-AI's depth-dependent Wang initialization.
    Initializer::Normal {
        mean: 0.0,
        std: 2.0 / size_f64(blocks) / size_f64(dim).sqrt(),
    }
}

pub(crate) fn linear(
    input: usize,
    output: usize,
    initializer: Initializer,
    device: &Device,
) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(false)
        .with_initializer(initializer)
        .init(device)
}

/// Independent linear projections within feature groups.
#[derive(Module, Debug)]
pub struct HeadLinear {
    /// Weights `[heads, output_per_head, input_per_head]` in reference layout.
    pub weight: Param<Tensor<3>>,
    /// Optional flattened output bias.
    pub bias: Option<Param<Tensor<1>>>,
}

impl HeadLinear {
    #[expect(
        clippy::needless_pass_by_value,
        reason = "Initializer is an inexpensive config value, consistent with Burn config APIs"
    )]
    pub(crate) fn init(
        width: usize,
        heads: usize,
        initializer: Initializer,
        device: &Device,
    ) -> Self {
        Self {
            weight: initializer.init([heads, width / heads, width / heads], device),
            bias: None,
        }
    }

    /// Projects `[batch, time, features]` independently within each head.
    ///
    /// # Panics
    /// Panics if the input feature width does not match the weights.
    #[must_use]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, time, width] = input.dims();
        let [heads, out, dim] = self.weight.val().dims();
        assert_eq!(width, heads * dim, "headwise projection feature mismatch");
        let x = input.reshape([batch * time, heads, 1, dim]);
        let w = self.weight.val().swap_dims(1, 2).unsqueeze_dim::<4>(0);
        let y = x.matmul(w).reshape([batch, time, heads * out]);
        match &self.bias {
            Some(bias) => y + bias.val().reshape([1, 1, heads * out]),
            None => y,
        }
    }
}

/// Per-token normalization with a learned residual scale and no bias.
#[derive(Module, Debug)]
pub struct Norm {
    /// Residual scale; effective weight is `1 + weight`.
    pub weight: Param<Tensor<1>>,
    /// Number of independent feature groups.
    pub heads: usize,
    /// Variance epsilon (reference default `1e-5`).
    pub epsilon: f64,
}

impl Norm {
    pub(crate) fn init(width: usize, heads: usize, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([width], device)),
            heads,
            epsilon: 1e-5,
        }
    }

    /// Normalizes each token and head without mixing time or head dimensions.
    ///
    /// # Panics
    /// Panics if the feature width does not match the learned scale.
    #[must_use]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, time, width] = input.dims();
        assert_eq!(
            self.weight.val().dims(),
            [width],
            "normalization feature mismatch"
        );
        let grouped = input.reshape([batch, time, self.heads, width / self.heads]);
        let centered = grouped.clone() - grouped.mean_dim(3);
        let variance = centered.clone().square().mean_dim(3);
        let normalized =
            (centered / (variance + self.epsilon).sqrt()).reshape([batch, time, width]);
        normalized * (self.weight.val() + 1.0).reshape([1, 1, width])
    }
}

/// Causal depthwise convolution with explicit streaming history.
#[derive(Module, Debug)]
pub struct CausalConv {
    /// Kernel `[kernel_size, features]`, ordered oldest to newest.
    pub weight: Param<Tensor<2>>,
    /// Per-feature bias.
    pub bias: Param<Tensor<1>>,
}

impl CausalConv {
    pub(crate) fn init(width: usize, kernel: usize, device: &Device) -> Self {
        let bound = 1.0 / size_f64(kernel).sqrt();
        Self {
            weight: Param::from_tensor(Tensor::random(
                [kernel, width],
                Distribution::Uniform(-bound, bound),
                device,
            )),
            bias: Param::from_tensor(Tensor::random(
                [width],
                Distribution::Uniform(-bound, bound),
                device,
            )),
        }
    }

    /// Convolves a sequence and returns its last `kernel_size - 1` inputs.
    ///
    /// A `None` history starts a new sequence. Kernel size one needs no history
    /// and returns `None`. History remains differentiable until detached.
    ///
    /// # Panics
    /// Panics for empty sequences, mismatched features, or incorrect history shape.
    #[must_use]
    pub fn forward(
        &self,
        input: Tensor<3>,
        history: Option<Tensor<3>>,
    ) -> (Tensor<3>, Option<Tensor<3>>) {
        let [batch, time, width] = input.dims();
        let [kernel, features] = self.weight.val().dims();
        assert!(
            batch > 0 && time > 0,
            "convolution requires nonempty batch and time"
        );
        assert_eq!(width, features, "convolution feature mismatch");
        let padded = if kernel > 1 {
            let history = history
                .unwrap_or_else(|| Tensor::zeros([batch, kernel - 1, width], &input.device()));
            assert_eq!(
                history.dims(),
                [batch, kernel - 1, width],
                "convolution history mismatch"
            );
            Tensor::cat(vec![history, input], 1)
        } else {
            assert!(history.is_none(), "kernel size one takes no history");
            input
        };
        let mut output = Tensor::zeros([batch, time, width], &padded.device());
        for tap in 0..kernel {
            output = output
                + padded.clone().slice([0..batch, tap..tap + time, 0..width])
                    * self
                        .weight
                        .val()
                        .slice([tap..tap + 1, 0..width])
                        .reshape([1, 1, width]);
        }
        let history =
            (kernel > 1).then(|| padded.slice([0..batch, time..time + kernel - 1, 0..width]));
        (output + self.bias.val().reshape([1, 1, width]), history)
    }
}
