//! Pre-normalized residual blocks and configurable mixed xLSTM stacks.

use crate::{
    components::{Norm, linear, output_init, small_init},
    config::{self, ConfigError},
    layers::{MlstmConfig, MlstmLayer, MlstmLayerState, SlstmConfig, SlstmLayer, SlstmLayerState},
};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear},
    tensor::{Device, Tensor},
};

/// The memory type used at a position in the residual stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, burn::serde::Serialize, burn::serde::Deserialize)]
#[serde(crate = "burn::serde")]
pub enum BlockKind {
    /// Scalar memory with recurrent mixing and a separate feedforward residual.
    Slstm,
    /// Expanded matrix memory with an internal output gate.
    Mlstm,
}

/// GELU-gated feedforward sublayer of an sLSTM residual block.
#[derive(Module, Debug)]
pub struct FeedForward {
    /// Gate and value up-projection, concatenated in that order.
    pub up: Linear,
    /// Projection back to model width.
    pub down: Linear,
    /// Output dropout.
    pub dropout: Dropout,
}

impl FeedForward {
    fn init(width: usize, hidden: usize, blocks: usize, dropout: f64, device: &Device) -> Self {
        Self {
            up: linear(width, hidden * 2, small_init(width), device),
            down: linear(hidden, width, output_init(width, blocks), device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    /// Applies `down(gelu(gate) * value)` to each token.
    ///
    /// # Panics
    /// Panics for an incompatible input feature width.
    #[must_use]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let expanded = self.up.forward(input);
        let [batch, time, doubled] = expanded.dims();
        let hidden = doubled / 2;
        let gate = expanded.clone().slice([0..batch, 0..time, 0..hidden]);
        let value = expanded.slice([0..batch, 0..time, hidden..doubled]);
        // Compose exact GELU so autodiff differentiates erf itself. Burn 0.22's
        // generic gelu_backward uses a tanh approximation on some GPU backends.
        let activated = gate.clone() * ((gate / std::f64::consts::SQRT_2).erf() + 1.0) * 0.5;
        self.dropout.forward(self.down.forward(activated * value))
    }
}

/// Learned memory sublayer of a residual block.
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Layers own cheap tensor handles; keeping them inline avoids another allocation per block"
)]
pub enum MemoryLayer {
    /// Scalar-memory layer.
    Slstm(SlstmLayer),
    /// Matrix-memory layer.
    Mlstm(MlstmLayer),
}

/// Recurrent and convolution state for one residual block.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "Streaming states own tensor handles and are passed directly between calls"
)]
pub enum BlockState {
    /// Scalar-memory state.
    Slstm(SlstmLayerState),
    /// Matrix-memory state.
    Mlstm(MlstmLayerState),
}

impl BlockState {
    /// Detaches recurrent and convolution state from the backward graph.
    #[must_use]
    pub fn detach(self) -> Self {
        match self {
            Self::Slstm(state) => Self::Slstm(state.detach()),
            Self::Mlstm(state) => Self::Mlstm(state.detach()),
        }
    }
}

/// An xLSTM residual block with pre-normalization.
#[derive(Module, Debug)]
pub struct XlstmBlock {
    /// Normalization before the memory sublayer.
    pub norm: Norm,
    /// Scalar or matrix memory.
    pub memory: MemoryLayer,
    /// Pre-normalization and gated feedforward, present for scalar-memory blocks.
    pub feedforward: Option<(Norm, FeedForward)>,
}

impl XlstmBlock {
    /// Processes a sequence from empty memory with residual additions.
    ///
    /// # Panics
    /// Panics for empty dimensions or incompatible features.
    #[must_use]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let normalized = self.norm.forward(input.clone());
        let hidden = match &self.memory {
            MemoryLayer::Slstm(layer) => layer.forward(normalized),
            MemoryLayer::Mlstm(layer) => layer.forward(normalized),
        };
        self.feedforward(input + hidden)
    }

    fn feedforward(&self, input: Tensor<3>) -> Tensor<3> {
        match &self.feedforward {
            Some((norm, feedforward)) => input.clone() + feedforward.forward(norm.forward(input)),
            None => input,
        }
    }

    /// Processes a sequence chunk while retaining memory across calls.
    ///
    /// # Panics
    /// Panics for invalid input/state shapes or the wrong state variant.
    #[must_use]
    pub fn forward_with_state(
        &self,
        input: Tensor<3>,
        state: Option<BlockState>,
    ) -> (Tensor<3>, BlockState) {
        let normalized = self.norm.forward(input.clone());
        let (hidden, state) = match (&self.memory, state) {
            (MemoryLayer::Slstm(layer), None) => {
                let (x, state) = layer.forward_with_state(normalized, None);
                (x, BlockState::Slstm(state))
            }
            (MemoryLayer::Slstm(layer), Some(BlockState::Slstm(state))) => {
                let (x, state) = layer.forward_with_state(normalized, Some(state));
                (x, BlockState::Slstm(state))
            }
            (MemoryLayer::Mlstm(layer), None) => {
                let (x, state) = layer.forward_with_state(normalized, None);
                (x, BlockState::Mlstm(state))
            }
            (MemoryLayer::Mlstm(layer), Some(BlockState::Mlstm(state))) => {
                let (x, state) = layer.forward_with_state(normalized, Some(state));
                (x, BlockState::Mlstm(state))
            }
            _ => panic!("block state variant does not match its memory layer"),
        };
        (self.feedforward(input + hidden), state)
    }

    /// Processes one token while retaining memory.
    ///
    /// # Panics
    /// Panics if time is not one or the supplied state is incompatible.
    #[must_use]
    pub fn step(&self, input: Tensor<3>, state: Option<BlockState>) -> (Tensor<3>, BlockState) {
        assert_eq!(input.dims()[1], 1, "block step requires one token");
        self.forward_with_state(input, state)
    }
}

/// Configuration for a stack with an explicit order of memory block types.
#[derive(Config, Debug)]
pub struct XlstmConfig {
    /// Input and output feature width.
    pub embedding_dim: usize,
    /// Memory type at each layer, in execution order; must not be empty.
    pub blocks: Vec<BlockKind>,
    /// Memory heads in both block types.
    #[config(default = 4)]
    pub num_heads: usize,
    /// Expanded matrix-memory width; defaults to `ceil(2 * width / 64) * 64`.
    pub mlstm_inner_dim: Option<usize>,
    /// sLSTM feedforward width; defaults to `ceil(1.3 * width / 64) * 64`.
    pub feedforward_dim: Option<usize>,
    /// Width of independent mLSTM q/k/v projection groups.
    #[config(default = 4)]
    pub qkv_block_size: usize,
    /// Causal convolution kernel, zero to disable.
    #[config(default = 4)]
    pub conv_kernel: usize,
    /// Residual branch dropout.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Whether to apply a final layer normalization.
    #[config(default = true)]
    pub post_norm: bool,
}

impl XlstmConfig {
    /// Validates all settings and initializes the residual stack.
    ///
    /// # Errors
    /// Returns an error for empty stacks or invalid dimensions/numerical settings.
    pub fn init(&self, device: &Device) -> Result<Xlstm, ConfigError> {
        config::dimensions(self.embedding_dim, 1)?;
        config::depth(0, self.blocks.len())?;
        config::dropout(self.dropout)?;
        let hidden = self
            .feedforward_dim
            .unwrap_or_else(|| config::rounded(self.embedding_dim, 13, 10));
        config::dimensions(hidden, 1)?;
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (index, kind) in self.blocks.iter().enumerate() {
            let (memory, feedforward) = match kind {
                BlockKind::Slstm => {
                    let layer = SlstmConfig::new(self.embedding_dim)
                        .with_num_heads(self.num_heads)
                        .with_conv_kernel(self.conv_kernel)
                        .with_dropout(self.dropout)
                        .with_block_index(index)
                        .with_num_blocks(self.blocks.len())
                        .init(device)?;
                    (
                        MemoryLayer::Slstm(layer),
                        Some((
                            Norm::init(self.embedding_dim, 1, device),
                            FeedForward::init(
                                self.embedding_dim,
                                hidden,
                                self.blocks.len(),
                                self.dropout,
                                device,
                            ),
                        )),
                    )
                }
                BlockKind::Mlstm => {
                    let layer = MlstmConfig::new(self.embedding_dim)
                        .with_inner_dim(self.mlstm_inner_dim)
                        .with_num_heads(self.num_heads)
                        .with_qkv_block_size(self.qkv_block_size)
                        .with_conv_kernel(self.conv_kernel)
                        .with_dropout(self.dropout)
                        .with_num_blocks(self.blocks.len())
                        .init(device)?;
                    (MemoryLayer::Mlstm(layer), None)
                }
            };
            blocks.push(XlstmBlock {
                norm: Norm::init(self.embedding_dim, 1, device),
                memory,
                feedforward,
            });
        }
        Ok(Xlstm {
            blocks,
            post_norm: self
                .post_norm
                .then(|| Norm::init(self.embedding_dim, 1, device)),
        })
    }
}

/// Residually stacked sLSTM and mLSTM blocks.
#[derive(Module, Debug)]
pub struct Xlstm {
    /// Blocks in execution order.
    pub blocks: Vec<XlstmBlock>,
    /// Optional final layer normalization.
    pub post_norm: Option<Norm>,
}

/// Streaming state for every block of an xLSTM stack.
#[derive(Debug, Clone)]
pub struct XlstmState {
    /// State in the same order as the model's blocks.
    pub blocks: Vec<BlockState>,
}

impl XlstmState {
    /// Cuts the backward graph of every block's recurrent state.
    #[must_use]
    pub fn detach(self) -> Self {
        Self {
            blocks: self.blocks.into_iter().map(BlockState::detach).collect(),
        }
    }
}

impl Xlstm {
    /// Processes `[batch, time, embedding_dim]` from empty state.
    ///
    /// mLSTM blocks use parallel kernels; sLSTM blocks use recurrent memory mixing.
    ///
    /// # Panics
    /// Panics for empty dimensions or an incompatible feature width.
    #[must_use]
    pub fn forward(&self, mut input: Tensor<3>) -> Tensor<3> {
        for block in &self.blocks {
            input = block.forward(input);
        }
        self.normalize(input)
    }

    fn normalize(&self, input: Tensor<3>) -> Tensor<3> {
        match &self.post_norm {
            Some(norm) => norm.forward(input),
            None => input,
        }
    }

    /// Processes a chunk and returns state for exact recurrent continuation.
    ///
    /// The final stack normalization is applied only to outputs, never to the
    /// stored recurrent state. For sequence/step comparisons disable dropout.
    ///
    /// # Panics
    /// Panics if state count, state variants, or tensor shapes are incompatible.
    #[must_use]
    pub fn forward_with_state(
        &self,
        mut input: Tensor<3>,
        state: Option<XlstmState>,
    ) -> (Tensor<3>, XlstmState) {
        let states: Vec<_> = match state {
            Some(state) => {
                assert_eq!(
                    state.blocks.len(),
                    self.blocks.len(),
                    "stack state depth mismatch"
                );
                state.blocks.into_iter().map(Some).collect()
            }
            None => (0..self.blocks.len()).map(|_| None).collect(),
        };
        let mut next = Vec::with_capacity(self.blocks.len());
        for (block, state) in self.blocks.iter().zip(states) {
            let (output, state) = block.forward_with_state(input, state);
            input = output;
            next.push(state);
        }
        (self.normalize(input), XlstmState { blocks: next })
    }

    /// Processes one token with optional state from a previous call.
    ///
    /// # Panics
    /// Panics if time is not one or the state is incompatible.
    #[must_use]
    pub fn step(&self, input: Tensor<3>, state: Option<XlstmState>) -> (Tensor<3>, XlstmState) {
        assert_eq!(input.dims()[1], 1, "stack step requires one token");
        self.forward_with_state(input, state)
    }
}
