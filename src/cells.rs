//! Learned scalar and matrix memory cells.

use crate::{
    components::{HeadLinear, Norm, linear, size_f64},
    functional::{MlstmInput, MlstmState, SlstmGates, SlstmState},
};
use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear},
    tensor::{Device, Tensor, TensorData},
};

/// Multi-head sLSTM cell with recurrent memory mixing inside each head.
#[derive(Module, Debug)]
pub struct SlstmCell {
    /// Recurrent projections in input, forget, candidate, output order.
    pub recurrent: Vec<HeadLinear>,
    /// Biases `[4, heads, head_dim]` in the same gate order.
    pub bias: Param<Tensor<3>>,
}

impl SlstmCell {
    pub(crate) fn init(
        width: usize,
        heads: usize,
        index: usize,
        blocks: usize,
        device: &Device,
    ) -> Self {
        let dim = width / heads;
        let ratio = if blocks > 1 {
            size_f64(index) / size_f64(blocks - 1)
        } else {
            0.0
        };
        let mut bias = vec![0.0_f64; 4 * width];
        for head in 0..heads {
            for coordinate in 0..dim {
                // Block-dependent power-law forget biases from the reference.
                // A one-coordinate head uses the left endpoint instead of 0/0.
                let position = if dim > 1 {
                    size_f64(coordinate) / size_f64(dim - 1)
                } else {
                    0.0
                };
                bias[width + head * dim + coordinate] =
                    5.0 - 12.0 * position.powf(0.3 + 1.3 * ratio);
            }
        }
        Self {
            recurrent: (0..4)
                .map(|_| HeadLinear::init(width, heads, Initializer::Zeros, device))
                .collect(),
            bias: Param::from_tensor(Tensor::from_data(
                TensorData::new(bias, [4, heads, dim]),
                device,
            )),
        }
    }

    /// Adds recurrent projections and updates scalar memory for one token.
    ///
    /// `gates` contains input projections only; cell biases and recurrence are
    /// added here. Head boundaries are preserved by every recurrent matrix.
    ///
    /// # Panics
    /// Panics for incompatible gate, state, or parameter shapes.
    #[must_use]
    pub fn step(&self, gates: SlstmGates, state: SlstmState) -> SlstmState {
        let [batch, heads, dim] = gates.input.dims();
        let previous = state.hidden.clone().reshape([batch, 1, heads * dim]);
        let add = |input: Tensor<3>, gate: usize| {
            input
                + self.recurrent[gate]
                    .forward(previous.clone())
                    .reshape([batch, heads, dim])
                + self.bias.val().slice([gate..gate + 1, 0..heads, 0..dim])
        };
        SlstmGates {
            input: add(gates.input, 0),
            forget: add(gates.forget, 1),
            candidate: add(gates.candidate, 2),
            output: add(gates.output, 3),
        }
        .step(state)
    }
}

/// mLSTM gate projections and per-head output normalization.
#[derive(Module, Debug)]
pub struct MlstmCell {
    /// Input gate from concatenated q, k, v; one logit per memory head.
    pub input_gate: Linear,
    /// Forget gate from concatenated q, k, v; one logit per memory head.
    pub forget_gate: Linear,
    /// Per-token, per-head normalization after memory retrieval.
    pub norm: Norm,
    /// Memory head count (independent of q/k/v projection groups).
    pub heads: usize,
    /// Readout epsilon.
    pub epsilon: f64,
}

impl MlstmCell {
    pub(crate) fn init(width: usize, heads: usize, epsilon: f64, device: &Device) -> Self {
        let mut input_gate = linear(3 * width, heads, Initializer::Zeros, device);
        input_gate.bias = Some(
            Initializer::Normal {
                mean: 0.0,
                std: 0.1,
            }
            .init([heads], device),
        );
        let mut forget_gate = linear(3 * width, heads, Initializer::Zeros, device);
        let biases: Vec<f64> = (0..heads)
            .map(|head| {
                if heads == 1 {
                    3.0
                } else {
                    3.0 + 3.0 * size_f64(head) / size_f64(heads - 1)
                }
            })
            .collect();
        forget_gate.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(biases, [heads]),
            device,
        )));
        Self {
            input_gate,
            forget_gate,
            norm: Norm::init(width, heads, device),
            heads,
            epsilon,
        }
    }

    fn project(&self, query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> MlstmInput {
        let [batch, time, width] = query.dims();
        let gate_input = Tensor::cat(vec![query.clone(), key.clone(), value.clone()], 2);
        let split = |x: Tensor<3>| {
            x.reshape([batch, time, self.heads, width / self.heads])
                .swap_dims(1, 2)
        };
        MlstmInput {
            query: split(query),
            key: split(key),
            value: split(value),
            input: self
                .input_gate
                .forward(gate_input.clone())
                .swap_dims(1, 2)
                .unsqueeze_dim(3),
            forget: self
                .forget_gate
                .forward(gate_input)
                .swap_dims(1, 2)
                .unsqueeze_dim(3),
        }
    }

    fn normalize(&self, hidden: Tensor<4>) -> Tensor<3> {
        let [batch, heads, time, dim] = hidden.dims();
        self.norm
            .forward(hidden.swap_dims(1, 2).reshape([batch, time, heads * dim]))
    }

    /// Retrieves normalized memory for a full sequence from empty state.
    ///
    /// Inputs and output use `[batch, time, features]`. Uses the parallel kernel.
    ///
    /// # Panics
    /// Panics for incompatible input or parameter shapes, or empty dimensions.
    #[must_use]
    pub fn forward(&self, query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> Tensor<3> {
        self.normalize(self.project(query, key, value).parallel(self.epsilon))
    }

    /// Retrieves normalized memory for one token and returns the new state.
    ///
    /// # Panics
    /// Panics if time is not one, or input/state shapes are incompatible.
    #[must_use]
    pub fn step(
        &self,
        query: Tensor<3>,
        key: Tensor<3>,
        value: Tensor<3>,
        state: Option<MlstmState>,
    ) -> (Tensor<3>, MlstmState) {
        let [batch, _, width] = query.dims();
        let state = state.unwrap_or_else(|| {
            MlstmState::zeros([batch, self.heads, width / self.heads], &query.device())
        });
        let (hidden, state) = self.project(query, key, value).step(state, self.epsilon);
        (self.normalize(hidden), state)
    }
}
