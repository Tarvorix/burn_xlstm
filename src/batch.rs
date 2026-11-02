//! Padding, packed sequence boundaries, and token-weighted language-model loss.
use crate::{
    config::ConfigError,
    lm::XlstmLm,
    stack::{BlockState, Xlstm, XlstmState},
};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData, activation};

/// Valid-token positions and independent sequence boundaries, shaped `[batch, time]`.
#[derive(Debug, Clone)]
pub struct SequenceMask {
    /// True positions consume a token; false positions leave all state unchanged.
    pub valid: Tensor<2, Bool>,
    /// Reset all memory and convolution history before this valid token.
    /// Reset flags at invalid positions have no effect.
    pub reset: Tensor<2, Bool>,
}

impl SequenceMask {
    /// Creates a padding mask without internal sequence resets.
    #[must_use]
    pub fn new(valid: Tensor<2, Bool>) -> Self {
        Self {
            reset: valid.zeros_like(),
            valid,
        }
    }

    /// Creates a right-padding mask; zero-length batch items are supported.
    ///
    /// # Errors
    /// Returns an error for empty dimensions or a length greater than `time`.
    pub fn from_lengths(
        lengths: &[usize],
        time: usize,
        device: &Device,
    ) -> Result<Self, ConfigError> {
        if lengths.is_empty()
            || time == 0
            || lengths.iter().any(|&n| n > time)
            || lengths.len().checked_mul(time).is_none()
        {
            return Err(ConfigError::new(
                "lengths",
                "require a nonempty batch, positive time, and lengths <= time",
            ));
        }
        let valid: Vec<bool> = lengths
            .iter()
            .flat_map(|&n| (0..time).map(move |t| t < n))
            .collect();
        Ok(Self::new(Tensor::from_data(
            TensorData::new(valid, [lengths.len(), time]),
            device,
        )))
    }

    pub(crate) fn check(&self, shape: [usize; 2]) {
        assert_eq!(self.valid.dims(), shape, "valid-token mask shape mismatch");
        assert_eq!(
            self.reset.dims(),
            shape,
            "sequence-reset mask shape mismatch"
        );
    }
}

fn choose<const D: usize>(
    next: Tensor<D>,
    previous: Option<Tensor<D>>,
    active: Tensor<1, Bool>,
) -> Tensor<D> {
    let [batch] = active.dims();
    assert_eq!(next.dims()[0], batch, "state/mask batch mismatch");
    let mut shape = [1; D];
    shape[0] = batch;
    previous
        .unwrap_or_else(|| next.zeros_like())
        .mask_where(active.reshape(shape), next)
}

fn history(
    next: Option<Tensor<3>>,
    previous: Option<Tensor<3>>,
    active: Tensor<1, Bool>,
) -> Option<Tensor<3>> {
    if let Some(next) = next {
        Some(choose(next, previous, active))
    } else {
        assert!(previous.is_none(), "unexpected convolution history");
        None
    }
}

impl BlockState {
    fn select(self, previous: Option<Self>, active: Tensor<1, Bool>) -> Self {
        match self {
            Self::Slstm(mut next) => {
                let previous = match previous {
                    Some(Self::Slstm(state)) => Some(state),
                    None => None,
                    _ => panic!("state variant mismatch"),
                };
                next.cell.hidden = choose(
                    next.cell.hidden,
                    previous.as_ref().map(|s| s.cell.hidden.clone()),
                    active.clone(),
                );
                next.cell.memory = choose(
                    next.cell.memory,
                    previous.as_ref().map(|s| s.cell.memory.clone()),
                    active.clone(),
                );
                next.cell.normalizer = choose(
                    next.cell.normalizer,
                    previous.as_ref().map(|s| s.cell.normalizer.clone()),
                    active.clone(),
                );
                next.cell.log_scale = choose(
                    next.cell.log_scale,
                    previous.as_ref().map(|s| s.cell.log_scale.clone()),
                    active.clone(),
                );
                next.conv = history(next.conv, previous.and_then(|s| s.conv), active);
                Self::Slstm(next)
            }
            Self::Mlstm(mut next) => {
                let previous = match previous {
                    Some(Self::Mlstm(state)) => Some(state),
                    None => None,
                    _ => panic!("state variant mismatch"),
                };
                next.cell.memory = choose(
                    next.cell.memory,
                    previous.as_ref().map(|s| s.cell.memory.clone()),
                    active.clone(),
                );
                next.cell.normalizer = choose(
                    next.cell.normalizer,
                    previous.as_ref().map(|s| s.cell.normalizer.clone()),
                    active.clone(),
                );
                next.cell.log_scale = choose(
                    next.cell.log_scale,
                    previous.as_ref().map(|s| s.cell.log_scale.clone()),
                    active.clone(),
                );
                next.conv = history(next.conv, previous.and_then(|s| s.conv), active);
                Self::Mlstm(next)
            }
        }
    }
}

impl XlstmState {
    fn select(self, previous: Option<Self>, active: &Tensor<1, Bool>) -> Self {
        let previous: Vec<_> = match previous {
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
        Self {
            blocks: self
                .blocks
                .into_iter()
                .zip(previous)
                .map(|(next, old)| next.select(old, active.clone()))
                .collect(),
        }
    }

    /// Clears every memory and convolution tensor in the selected batch lanes.
    ///
    /// Reset lanes lose their backward connection to the previous sequence.
    /// # Panics
    /// Panics when the mask's batch size does not match the state.
    #[must_use]
    pub fn reset(self, reset: Tensor<1, Bool>) -> Self {
        self.select(None, &reset.bool_not())
    }
}

impl Xlstm {
    /// Processes padding and packed sequences with exact state preservation.
    ///
    /// Invalid positions produce zeros and do not advance memory or convolution.
    /// Reset flags clear state before valid tokens. This path executes recurrently
    /// to support arbitrary mask holes and sequence boundaries without host readback.
    /// Disable dropout when comparing packed, padded, and individual execution.
    ///
    /// # Panics
    /// Panics for empty dimensions, mismatched masks, or incompatible input/state.
    #[must_use]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "tensor ownership matches the other forward APIs"
    )]
    pub fn forward_masked(
        &self,
        input: Tensor<3>,
        mask: &SequenceMask,
        mut state: Option<XlstmState>,
    ) -> (Tensor<3>, XlstmState) {
        let [batch, time, width] = input.dims();
        assert!(
            batch > 0 && time > 0,
            "masked execution needs nonempty batch and time"
        );
        mask.check([batch, time]);
        let mut outputs = Vec::with_capacity(time);
        for t in 0..time {
            let active = mask
                .valid
                .clone()
                .slice([0..batch, t..t + 1])
                .reshape([batch]);
            let reset = mask
                .reset
                .clone()
                .slice([0..batch, t..t + 1])
                .reshape([batch])
                .bool_and(active.clone());
            let previous = state.map(|s| s.reset(reset));
            let inactive = active.clone().bool_not().reshape([batch, 1, 1]);
            // Sanitize before projections: padded NaNs must never enter a graph.
            let token = input
                .clone()
                .slice([0..batch, t..t + 1, 0..width])
                .mask_fill(inactive.clone(), 0.0);
            let (output, next) = self.step(token, previous.clone());
            outputs.push(output.mask_fill(inactive, 0.0));
            state = Some(next.select(previous, &active));
        }
        (
            Tensor::cat(outputs, 1),
            state.expect("nonempty sequence creates state"),
        )
    }
}

impl XlstmLm {
    /// Processes masked tokens, returning zero logits for invalid positions.
    ///
    /// Invalid token IDs may be used as padding; they are replaced before embedding.
    /// # Panics
    /// Panics for invalid unmasked IDs, empty dimensions, or incompatible masks/state.
    #[must_use]
    pub fn forward_masked(
        &self,
        tokens: Tensor<2, Int>,
        mask: &SequenceMask,
        state: Option<XlstmState>,
    ) -> (Tensor<3>, XlstmState) {
        mask.check(tokens.dims());
        let tokens = tokens.mask_fill(mask.valid.clone().bool_not(), 0);
        let (hidden, state) =
            self.stack
                .forward_masked(self.embedding.forward(tokens), mask, state);
        (self.head.forward(hidden), state)
    }
}

/// Mean cross entropy over valid tokens, with zero loss for an empty mask.
///
/// Invalid logits and targets are sanitized before log-softmax/gather. Targets
/// must already be aligned with logits (no implicit next-token shift).
/// # Panics
/// Panics for shape mismatches or unmasked targets outside the vocabulary.
#[must_use]
pub fn masked_cross_entropy(
    logits: Tensor<3>,
    targets: Tensor<2, Int>,
    valid: Tensor<2, Bool>,
) -> Tensor<1> {
    let [batch, time, _] = logits.dims();
    assert_eq!(targets.dims(), [batch, time], "target shape mismatch");
    assert_eq!(valid.dims(), [batch, time], "loss mask shape mismatch");
    let invalid = valid.clone().bool_not();
    let logits = logits.mask_fill(invalid.clone().reshape([batch, time, 1]), 0.0);
    let targets = targets
        .mask_fill(invalid.clone(), 0)
        .reshape([batch, time, 1]);
    let losses = -activation::log_softmax(logits, 2)
        .gather(2, targets)
        .reshape([batch, time]);
    losses.mask_fill(invalid, 0.0).sum() / valid.float().sum().clamp_min(1.0)
}

/// Next-token training tensors with padding and sequence-boundary metadata.
#[derive(Debug, Clone)]
pub struct LmBatch {
    /// Input token IDs `[batch, time]`.
    pub inputs: Tensor<2, Int>,
    /// Next-token targets, aligned with `inputs`; never cross sequence boundaries.
    pub targets: Tensor<2, Int>,
    /// Valid positions and resets at the start of each original sequence.
    pub mask: SequenceMask,
}

impl LmBatch {
    /// Pads each independent sequence to the longest next-token input length.
    ///
    /// Sequences shorter than two tokens contribute no valid training positions.
    /// # Errors
    /// Returns an error for negative token IDs or no next-token pairs.
    pub fn padded(sequences: &[Vec<i32>], device: &Device) -> Result<Self, ConfigError> {
        check_tokens(sequences)?;
        let rows: Vec<_> = sequences
            .iter()
            .map(|sequence| Self::pairs(sequence))
            .collect();
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        Self::assemble(rows, width, device)
    }

    /// Packs whole sequences into fixed-width rows without cross-sequence targets.
    ///
    /// Memory and convolution reset at every original sequence boundary. A
    /// sequence is never split across rows. Short sequences may share a row.
    /// # Errors
    /// Returns an error for zero row width, an oversized sequence, negative IDs,
    /// or no next-token pairs.
    pub fn packed(
        sequences: &[Vec<i32>],
        row_width: usize,
        device: &Device,
    ) -> Result<Self, ConfigError> {
        if row_width == 0 {
            return Err(ConfigError::new("row_width", "must be positive"));
        }
        check_tokens(sequences)?;
        let mut rows = Vec::new();
        let mut row = Vec::new();
        for sequence in sequences {
            let pairs = Self::pairs(sequence);
            if pairs.len() > row_width {
                return Err(ConfigError::new(
                    "row_width",
                    "must fit every sequence's next-token pairs",
                ));
            }
            if row.len() > row_width - pairs.len() {
                rows.push(std::mem::take(&mut row));
            }
            row.extend(pairs);
        }
        if !row.is_empty() {
            rows.push(row);
        }
        Self::assemble(rows, row_width, device)
    }

    fn pairs(sequence: &[i32]) -> Vec<(i32, i32, bool)> {
        sequence
            .windows(2)
            .enumerate()
            .map(|(i, pair)| (pair[0], pair[1], i == 0))
            .collect()
    }

    fn assemble(
        rows: Vec<Vec<(i32, i32, bool)>>,
        width: usize,
        device: &Device,
    ) -> Result<Self, ConfigError> {
        let batch = rows.len();
        let size = batch
            .checked_mul(width)
            .ok_or_else(|| ConfigError::new("batch", "dimensions overflow"))?;
        if size == 0 || !rows.iter().any(|row| !row.is_empty()) {
            return Err(ConfigError::new(
                "sequences",
                "need at least one sequence with two tokens",
            ));
        }
        let mut inputs = vec![0; size];
        let mut targets = vec![0; size];
        let mut valid = vec![false; size];
        let mut reset = vec![false; size];
        for (b, row) in rows.into_iter().enumerate() {
            for (t, (input, target, boundary)) in row.into_iter().enumerate() {
                if input < 0 || target < 0 {
                    return Err(ConfigError::new("tokens", "token IDs must be nonnegative"));
                }
                let index = b * width + t;
                inputs[index] = input;
                targets[index] = target;
                valid[index] = true;
                reset[index] = boundary;
            }
        }
        let shape = [batch, width];
        Ok(Self {
            inputs: Tensor::from_data(TensorData::new(inputs, shape), device),
            targets: Tensor::from_data(TensorData::new(targets, shape), device),
            mask: SequenceMask {
                valid: Tensor::from_data(TensorData::new(valid, shape), device),
                reset: Tensor::from_data(TensorData::new(reset, shape), device),
            },
        })
    }

    /// Computes mean loss over valid next-token pairs in this batch.
    /// # Panics
    /// Panics for incompatible logits or targets outside the vocabulary.
    #[must_use]
    pub fn loss(&self, logits: Tensor<3>) -> Tensor<1> {
        masked_cross_entropy(logits, self.targets.clone(), self.mask.valid.clone())
    }
}

fn check_tokens(sequences: &[Vec<i32>]) -> Result<(), ConfigError> {
    if sequences.iter().flatten().any(|&token| token < 0) {
        return Err(ConfigError::new("tokens", "token IDs must be nonnegative"));
    }
    Ok(())
}
