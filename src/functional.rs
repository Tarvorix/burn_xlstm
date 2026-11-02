//! Differentiable, stabilized xLSTM recurrences without learned projections.
//!
//! sLSTM tensors use `[batch, heads, head_dim]`. mLSTM inputs use
//! `[batch, heads, time, head_dim]`, with scalar gates on the last axis.
//! All operations stay on the tensor device, including stabilization.

use burn::tensor::{Device, Int, Tensor, activation};

/// Scalar memory, normalization, log scale, and recurrent hidden state.
#[derive(Debug, Clone)]
pub struct SlstmState {
    /// Hidden output `[batch, heads, head_dim]`.
    pub hidden: Tensor<3>,
    /// Scaled scalar memory, with the same shape as `hidden`.
    pub memory: Tensor<3>,
    /// Scaled normalizer, with the same shape as `hidden`.
    pub normalizer: Tensor<3>,
    /// Log stabilization scale, with the same shape as `hidden`.
    pub log_scale: Tensor<3>,
}

impl SlstmState {
    /// Creates an empty recurrent state.
    #[must_use]
    pub fn zeros(shape: [usize; 3], device: &Device) -> Self {
        Self {
            hidden: Tensor::zeros(shape, device),
            memory: Tensor::zeros(shape, device),
            normalizer: Tensor::zeros(shape, device),
            log_scale: Tensor::zeros(shape, device),
        }
    }

    /// Cuts the backward graph for truncated backpropagation through time.
    #[must_use]
    pub fn detach(self) -> Self {
        Self {
            hidden: self.hidden.detach(),
            memory: self.memory.detach(),
            normalizer: self.normalizer.detach(),
            log_scale: self.log_scale.detach(),
        }
    }
}

/// sLSTM gate preactivations after input and recurrent projections.
#[derive(Debug, Clone)]
pub struct SlstmGates {
    /// Exponential input gate logits.
    pub input: Tensor<3>,
    /// Sigmoid forget gate logits.
    pub forget: Tensor<3>,
    /// Memory candidate before tanh.
    pub candidate: Tensor<3>,
    /// Output gate before sigmoid.
    pub output: Tensor<3>,
}

impl SlstmGates {
    /// Updates scalar memory and returns the new hidden state within the state.
    ///
    /// Empty coordinates use the input logit as their first scale, avoiding
    /// underflow for very negative first input logits. This generalizes the
    /// reference's all-zero initialization branch to independently reset lanes.
    ///
    /// # Panics
    /// Panics if gate or state shapes differ.
    #[must_use]
    pub fn step(self, state: SlstmState) -> SlstmState {
        let shape = self.input.dims();
        for actual in [
            self.forget.dims(),
            self.candidate.dims(),
            self.output.dims(),
            state.hidden.dims(),
            state.memory.dims(),
            state.normalizer.dims(),
            state.log_scale.dims(),
        ] {
            assert_eq!(actual, shape, "sLSTM gate/state shape mismatch");
        }
        let log_forget = activation::log_sigmoid(self.forget) + state.log_scale;
        let empty = state.normalizer.clone().equal_elem(0.0);
        let log_scale = self
            .input
            .clone()
            .max_pair(log_forget.clone())
            .mask_where(empty, self.input.clone());
        // Clamp before exp: on the first step an unused forget coefficient can
        // otherwise overflow even though the previous memory is exactly zero.
        let input = (self.input - log_scale.clone()).clamp_max(0.0).exp();
        let forget = (log_forget - log_scale.clone()).clamp_max(0.0).exp();
        let memory = forget.clone() * state.memory + input.clone() * self.candidate.tanh();
        let normalizer = forget * state.normalizer + input;
        let hidden = activation::sigmoid(self.output) * memory.clone() / normalizer.clone();
        SlstmState {
            hidden,
            memory,
            normalizer,
            log_scale,
        }
    }
}

/// Matrix memory and its stabilizing normalizer.
#[derive(Debug, Clone)]
pub struct MlstmState {
    /// Scaled key-by-value matrix `[batch, heads, head_dim, head_dim]`.
    pub memory: Tensor<4>,
    /// Scaled key sum `[batch, heads, head_dim, 1]`.
    pub normalizer: Tensor<4>,
    /// Log scale `[batch, heads, 1, 1]`.
    pub log_scale: Tensor<4>,
}

impl MlstmState {
    /// Creates an empty matrix memory for `[batch, heads, head_dim]`.
    #[must_use]
    pub fn zeros([batch, heads, dim]: [usize; 3], device: &Device) -> Self {
        Self {
            memory: Tensor::zeros([batch, heads, dim, dim], device),
            normalizer: Tensor::zeros([batch, heads, dim, 1], device),
            log_scale: Tensor::zeros([batch, heads, 1, 1], device),
        }
    }

    /// Cuts the backward graph while retaining numerical memory.
    #[must_use]
    pub fn detach(self) -> Self {
        Self {
            memory: self.memory.detach(),
            normalizer: self.normalizer.detach(),
            log_scale: self.log_scale.detach(),
        }
    }
}

/// mLSTM queries, keys, values, and gate logits.
#[derive(Debug, Clone)]
pub struct MlstmInput {
    /// Queries `[batch, heads, time, head_dim]`.
    pub query: Tensor<4>,
    /// Keys of the same shape; scaled internally by `1 / sqrt(head_dim)`.
    pub key: Tensor<4>,
    /// Values of the same shape.
    pub value: Tensor<4>,
    /// Exponential input gate logits `[batch, heads, time, 1]`.
    pub input: Tensor<4>,
    /// Sigmoid forget gate logits `[batch, heads, time, 1]`.
    pub forget: Tensor<4>,
}

impl MlstmInput {
    fn check(&self, epsilon: f64) -> [usize; 4] {
        let [batch, heads, time, dim] = self.query.dims();
        assert!(
            batch > 0 && heads > 0 && time > 0 && dim > 0,
            "mLSTM dimensions must be nonzero, got {:?}",
            self.query.dims()
        );
        assert!(
            epsilon.is_finite() && epsilon > 0.0,
            "invalid epsilon {epsilon}"
        );
        assert_eq!(
            self.key.dims(),
            self.query.dims(),
            "mLSTM key shape mismatch"
        );
        assert_eq!(
            self.value.dims(),
            self.query.dims(),
            "mLSTM value shape mismatch"
        );
        assert_eq!(
            self.input.dims(),
            [batch, heads, time, 1],
            "mLSTM input gate shape mismatch"
        );
        assert_eq!(
            self.forget.dims(),
            [batch, heads, time, 1],
            "mLSTM forget gate shape mismatch"
        );
        [batch, heads, time, dim]
    }

    /// Computes one recurrent step, retaining memory for the next token.
    ///
    /// The stored matrix follows NX-AI's key-by-value orientation; retrieval
    /// is `q^T C`. This is the transpose of the paper's value-by-key notation.
    ///
    /// # Panics
    /// Panics for mismatched shapes, time other than one, or invalid epsilon.
    #[must_use]
    pub fn step(self, state: MlstmState, epsilon: f64) -> (Tensor<4>, MlstmState) {
        let [batch, heads, time, dim] = self.check(epsilon);
        assert_eq!(time, 1, "mLSTM step requires exactly one token");
        assert_eq!(
            state.memory.dims(),
            [batch, heads, dim, dim],
            "mLSTM memory shape mismatch"
        );
        assert_eq!(
            state.normalizer.dims(),
            [batch, heads, dim, 1],
            "mLSTM normalizer shape mismatch"
        );
        assert_eq!(
            state.log_scale.dims(),
            [batch, heads, 1, 1],
            "mLSTM scale shape mismatch"
        );
        let log_forget = activation::log_sigmoid(self.forget) + state.log_scale;
        let log_scale = log_forget.clone().max_pair(self.input.clone());
        let forget = (log_forget - log_scale.clone()).exp();
        let input = (self.input - log_scale.clone()).exp();
        let key = self.key.swap_dims(2, 3) / crate::components::size_f64(dim).sqrt();
        let memory = forget.clone() * state.memory + input.clone() * key.clone().matmul(self.value);
        let normalizer = forget * state.normalizer + input * key;
        let numerator = self.query.clone().matmul(memory.clone());
        let dot = self.query.matmul(normalizer.clone());
        let hidden = readout(numerator, dot, log_scale.clone(), epsilon);
        (
            hidden,
            MlstmState {
                memory,
                normalizer,
                log_scale,
            },
        )
    }

    /// Computes a causal sequence in parallel from empty memory.
    ///
    /// Uses quadratic time-axis storage. For bounded memory or continuation,
    /// use recurrent steps instead. The row-wise stabilization and epsilon
    /// follow the reference parallel implementation.
    ///
    /// # Panics
    /// Panics for mismatched shapes, empty dimensions, or invalid epsilon.
    #[must_use]
    pub fn parallel(self, epsilon: f64) -> Tensor<4> {
        let [_, _, time, dim] = self.check(epsilon);
        let device = self.query.device();
        let indices = Tensor::<1, Int>::arange(
            0..i64::try_from(time).expect("sequence length exceeds i64"),
            &device,
        );
        let future = indices
            .clone()
            .reshape([1, time])
            .greater(indices.reshape([time, 1]))
            .reshape([1, 1, time, time]);
        let accumulated = activation::log_sigmoid(self.forget).cumsum(2);
        let decay = accumulated.clone() - accumulated.swap_dims(2, 3) + self.input.swap_dims(2, 3);
        let decay = decay.mask_fill(future, f64::NEG_INFINITY);
        let log_scale = decay.clone().max_dim(3);
        let weights = (decay - log_scale.clone()).exp();
        let similarity =
            self.query.matmul(self.key.swap_dims(2, 3)) / crate::components::size_f64(dim).sqrt();
        let combination = similarity * weights;
        // Normalize before multiplying values, as in the reference backend.
        let normalized = readout(
            combination.clone(),
            combination.sum_dim(3),
            log_scale,
            epsilon,
        );
        normalized.matmul(self.value)
    }
}

// Algebraically equivalent to num / (max(abs(dot), exp(-m)) + eps),
// but never evaluates exp of a positive log scale. Very negative input gates
// therefore produce finite zeros instead of overflowing the denominator.
fn readout(numerator: Tensor<4>, dot: Tensor<4>, scale: Tensor<4>, epsilon: f64) -> Tensor<4> {
    let factor = scale.clone().clamp_max(0.0).exp();
    let floor = scale.clamp_min(0.0).neg().exp();
    numerator * factor.clone() / ((dot.abs() * factor.clone()).max_pair(floor) + factor * epsilon)
}
