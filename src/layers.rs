//! sLSTM and mLSTM layers with convolution and streaming state.

use crate::{
    cells::{MlstmCell, SlstmCell},
    components::{CausalConv, HeadLinear, Norm, linear, output_init, small_init},
    config::{self, ConfigError},
    functional::{MlstmState, SlstmGates, SlstmState},
};
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Linear},
    tensor::{Device, Tensor, activation},
};

/// Configuration for a scalar-memory layer.
#[derive(Config, Debug)]
pub struct SlstmConfig {
    /// Input and output feature width.
    pub embedding_dim: usize,
    /// Number of recurrent heads.
    #[config(default = 4)]
    pub num_heads: usize,
    /// Causal convolution kernel; zero disables convolution and its activation.
    #[config(default = 4)]
    pub conv_kernel: usize,
    /// Dropout before head normalization.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Position in the residual stack, used for forget-bias initialization.
    #[config(default = 0)]
    pub block_index: usize,
    /// Residual stack depth.
    #[config(default = 1)]
    pub num_blocks: usize,
}

impl SlstmConfig {
    /// Validates dimensions and initializes a scalar-memory layer.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, head division, depth, or dropout.
    pub fn init(&self, device: &Device) -> Result<SlstmLayer, ConfigError> {
        config::dimensions(self.embedding_dim, self.num_heads)?;
        config::dropout(self.dropout)?;
        config::depth(self.block_index, self.num_blocks)?;
        config::kernel(self.conv_kernel)?;
        let width = self.embedding_dim;
        Ok(SlstmLayer {
            conv: (self.conv_kernel > 0).then(|| CausalConv::init(width, self.conv_kernel, device)),
            input: HeadLinear::init(width, self.num_heads, small_init(width), device),
            forget: HeadLinear::init(width, self.num_heads, small_init(width), device),
            candidate: HeadLinear::init(width, self.num_heads, small_init(width), device),
            output: HeadLinear::init(width, self.num_heads, small_init(width), device),
            cell: SlstmCell::init(
                width,
                self.num_heads,
                self.block_index,
                self.num_blocks,
                device,
            ),
            norm: Norm::init(width, self.num_heads, device),
            dropout: DropoutConfig::new(self.dropout).init(),
        })
    }
}

/// Scalar-memory layer with headwise input and recurrent projections.
#[derive(Module, Debug)]
pub struct SlstmLayer {
    /// Optional causal convolution for input/forget gates.
    pub conv: Option<CausalConv>,
    /// Input-gate projection (called `fgate` in the reference layer).
    pub input: HeadLinear,
    /// Forget-gate projection (called `igate` in the reference layer).
    pub forget: HeadLinear,
    /// Candidate projection from the unconvolved input.
    pub candidate: HeadLinear,
    /// Output-gate projection from the unconvolved input.
    pub output: HeadLinear,
    /// Recurrent scalar-memory cell.
    pub cell: SlstmCell,
    /// Output normalization within each head.
    pub norm: Norm,
    /// Dropout applied before output normalization.
    pub dropout: Dropout,
}

/// Scalar-memory layer state, including causal convolution history.
#[derive(Debug, Clone)]
pub struct SlstmLayerState {
    /// Scalar memory and hidden state.
    pub cell: SlstmState,
    /// Last convolution inputs, or `None` when no history is needed.
    pub conv: Option<Tensor<3>>,
}

impl SlstmLayerState {
    /// Detaches both recurrent memory and convolution history.
    #[must_use]
    pub fn detach(self) -> Self {
        Self {
            cell: self.cell.detach(),
            conv: self.conv.map(Tensor::detach),
        }
    }
}

impl SlstmLayer {
    /// Processes `[batch, time, features]` from empty memory.
    ///
    /// # Panics
    /// Panics for empty dimensions or incompatible input width.
    #[must_use]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        self.forward_with_state(input, None).0
    }

    /// Processes a chunk and returns memory for continuation.
    ///
    /// # Panics
    /// Panics for empty dimensions, incompatible features, or invalid state shapes.
    #[must_use]
    pub fn forward_with_state(
        &self,
        input: Tensor<3>,
        state: Option<SlstmLayerState>,
    ) -> (Tensor<3>, SlstmLayerState) {
        let [batch, time, width] = input.dims();
        assert!(
            batch > 0 && time > 0,
            "sLSTM requires nonempty batch and time"
        );
        let heads = self.norm.heads;
        let dim = width / heads;
        let (mut cell, history) = match state {
            Some(state) => (state.cell, state.conv),
            None => (
                SlstmState::zeros([batch, heads, dim], &input.device()),
                None,
            ),
        };
        let (gate_input, conv) = if let Some(conv) = &self.conv {
            let (x, history) = conv.forward(input.clone(), history);
            (activation::silu(x), history)
        } else {
            assert!(history.is_none(), "disabled convolution takes no history");
            (input.clone(), None)
        };
        let input_gate = self.input.forward(gate_input.clone());
        let forget_gate = self.forget.forward(gate_input);
        let candidate = self.candidate.forward(input.clone());
        let output_gate = self.output.forward(input);
        let mut outputs = Vec::with_capacity(time);
        for token in 0..time {
            let take = |x: &Tensor<3>| {
                x.clone()
                    .slice([0..batch, token..token + 1, 0..width])
                    .reshape([batch, heads, dim])
            };
            cell = self.cell.step(
                SlstmGates {
                    input: take(&input_gate),
                    forget: take(&forget_gate),
                    candidate: take(&candidate),
                    output: take(&output_gate),
                },
                cell,
            );
            outputs.push(cell.hidden.clone().reshape([batch, 1, width]));
        }
        let output = self
            .norm
            .forward(self.dropout.forward(Tensor::cat(outputs, 1)));
        (output, SlstmLayerState { cell, conv })
    }

    /// Processes exactly one token with optional existing state.
    ///
    /// # Panics
    /// Panics when time is not one or tensor/state shapes are incompatible.
    #[must_use]
    pub fn step(
        &self,
        input: Tensor<3>,
        state: Option<SlstmLayerState>,
    ) -> (Tensor<3>, SlstmLayerState) {
        assert_eq!(input.dims()[1], 1, "sLSTM step requires one token");
        self.forward_with_state(input, state)
    }
}

/// Configuration for an expanded matrix-memory layer.
#[derive(Config, Debug)]
pub struct MlstmConfig {
    /// Input and output feature width.
    pub embedding_dim: usize,
    /// Expanded width; defaults to twice the input width rounded up to 64.
    pub inner_dim: Option<usize>,
    /// Number of matrix memory heads.
    #[config(default = 4)]
    pub num_heads: usize,
    /// Width of each independent q/k/v projection group.
    #[config(default = 4)]
    pub qkv_block_size: usize,
    /// Causal convolution kernel; zero disables convolution (`SiLU` remains).
    #[config(default = 4)]
    pub conv_kernel: usize,
    /// Output dropout.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Positive readout epsilon.
    #[config(default = 1e-6)]
    pub epsilon: f64,
    /// Stack depth for output projection initialization.
    #[config(default = 1)]
    pub num_blocks: usize,
}

impl MlstmConfig {
    /// Validates dimensions and initializes a matrix-memory layer.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, group division, dropout, or epsilon.
    pub fn init(&self, device: &Device) -> Result<MlstmLayer, ConfigError> {
        config::dimensions(self.embedding_dim, 1)?;
        let width = self
            .inner_dim
            .unwrap_or_else(|| config::rounded(self.embedding_dim, 2, 1));
        config::dimensions(width, self.num_heads)?;
        config::dropout(self.dropout)?;
        config::depth(0, self.num_blocks)?;
        config::kernel(self.conv_kernel)?;
        if self.qkv_block_size == 0 || !width.is_multiple_of(self.qkv_block_size) {
            return Err(ConfigError::new(
                "qkv_block_size",
                format!(
                    "{} must be positive and divide {width}",
                    self.qkv_block_size
                ),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(ConfigError::new("epsilon", "must be finite and positive"));
        }
        let projection_heads = width / self.qkv_block_size;
        Ok(MlstmLayer {
            up: linear(
                self.embedding_dim,
                2 * width,
                small_init(self.embedding_dim),
                device,
            ),
            down: linear(
                width,
                self.embedding_dim,
                output_init(self.embedding_dim, self.num_blocks),
                device,
            ),
            query: HeadLinear::init(
                width,
                projection_heads,
                small_init(self.embedding_dim),
                device,
            ),
            key: HeadLinear::init(
                width,
                projection_heads,
                small_init(self.embedding_dim),
                device,
            ),
            value: HeadLinear::init(
                width,
                projection_heads,
                small_init(self.embedding_dim),
                device,
            ),
            conv: (self.conv_kernel > 0).then(|| CausalConv::init(width, self.conv_kernel, device)),
            cell: MlstmCell::init(width, self.num_heads, self.epsilon, device),
            skip: Param::from_tensor(Tensor::ones([width], device)),
            dropout: DropoutConfig::new(self.dropout).init(),
        })
    }
}

/// Expanded mLSTM layer with an output gate and learned local skip.
#[derive(Module, Debug)]
pub struct MlstmLayer {
    /// Joint memory/output-gate expansion.
    pub up: Linear,
    /// Projection back to model width.
    pub down: Linear,
    /// Headwise query projection from activated convolution output.
    pub query: HeadLinear,
    /// Headwise key projection from activated convolution output.
    pub key: HeadLinear,
    /// Headwise value projection from unconvolved expanded input.
    pub value: HeadLinear,
    /// Optional causal convolution on the memory branch.
    pub conv: Option<CausalConv>,
    /// Matrix memory with gate projections and output normalization.
    pub cell: MlstmCell,
    /// Learned elementwise skip, initialized to one.
    pub skip: Param<Tensor<1>>,
    /// Dropout after output projection.
    pub dropout: Dropout,
}

/// Matrix-memory layer state, including convolution history.
#[derive(Debug, Clone)]
pub struct MlstmLayerState {
    /// Matrix memory, normalizer, and log scale.
    pub cell: MlstmState,
    /// Last convolution inputs, or `None` when no history is needed.
    pub conv: Option<Tensor<3>>,
}

impl MlstmLayerState {
    /// Detaches both matrix memory and convolution history.
    #[must_use]
    pub fn detach(self) -> Self {
        Self {
            cell: self.cell.detach(),
            conv: self.conv.map(Tensor::detach),
        }
    }
}

impl MlstmLayer {
    fn prepare(
        &self,
        input: Tensor<3>,
        history: Option<Tensor<3>>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>, Option<Tensor<3>>) {
        let [batch, time, _] = input.dims();
        let [width] = self.skip.val().dims();
        let expanded = self.up.forward(input);
        let memory = expanded.clone().slice([0..batch, 0..time, 0..width]);
        let gate = activation::silu(expanded.slice([0..batch, 0..time, width..2 * width]));
        let (convolved, history) = if let Some(conv) = &self.conv {
            conv.forward(memory.clone(), history)
        } else {
            assert!(history.is_none(), "disabled convolution takes no history");
            (memory.clone(), None)
        };
        (memory, activation::silu(convolved), gate, history)
    }

    fn finish(&self, hidden: Tensor<3>, convolved: Tensor<3>, gate: Tensor<3>) -> Tensor<3> {
        let [width] = self.skip.val().dims();
        let hidden = hidden + convolved * self.skip.val().reshape([1, 1, width]);
        self.dropout.forward(self.down.forward(hidden * gate))
    }

    /// Processes a full sequence with parallel matrix memory from empty state.
    ///
    /// # Panics
    /// Panics for empty dimensions or incompatible feature widths.
    #[must_use]
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let (memory, convolved, gate, _) = self.prepare(input, None);
        let hidden = self.cell.forward(
            self.query.forward(convolved.clone()),
            self.key.forward(convolved.clone()),
            self.value.forward(memory),
        );
        self.finish(hidden, convolved, gate)
    }

    /// Processes a chunk recurrently, returning state for the next chunk.
    ///
    /// Uses memory linear in sequence length during training and constant
    /// recurrent state during inference; it does not build a time-by-time matrix.
    ///
    /// # Panics
    /// Panics for empty dimensions or incompatible input/state shapes.
    #[must_use]
    pub fn forward_with_state(
        &self,
        input: Tensor<3>,
        state: Option<MlstmLayerState>,
    ) -> (Tensor<3>, MlstmLayerState) {
        let [batch, time, _] = input.dims();
        assert!(
            batch > 0 && time > 0,
            "mLSTM requires nonempty batch and time"
        );
        let [width] = self.skip.val().dims();
        let (mut cell, history) = match state {
            Some(state) => (state.cell, state.conv),
            None => (
                MlstmState::zeros(
                    [batch, self.cell.heads, width / self.cell.heads],
                    &input.device(),
                ),
                None,
            ),
        };
        let (memory, convolved, gate, conv) = self.prepare(input, history);
        let query = self.query.forward(convolved.clone());
        let key = self.key.forward(convolved.clone());
        let value = self.value.forward(memory);
        let mut outputs = Vec::with_capacity(time);
        for token in 0..time {
            let take = |x: &Tensor<3>| x.clone().slice([0..batch, token..token + 1, 0..width]);
            let (hidden, next) = self
                .cell
                .step(take(&query), take(&key), take(&value), Some(cell));
            outputs.push(hidden);
            cell = next;
        }
        (
            self.finish(Tensor::cat(outputs, 1), convolved, gate),
            MlstmLayerState { cell, conv },
        )
    }

    /// Processes exactly one token with optional existing state.
    ///
    /// # Panics
    /// Panics when time is not one or tensor/state shapes are incompatible.
    #[must_use]
    pub fn step(
        &self,
        input: Tensor<3>,
        state: Option<MlstmLayerState>,
    ) -> (Tensor<3>, MlstmLayerState) {
        assert_eq!(input.dims()[1], 1, "mLSTM step requires one token");
        self.forward_with_state(input, state)
    }
}
