//! Token embedding and language-model output head around an xLSTM stack.
use crate::{
    components::{linear, small_init},
    config::{self, ConfigError},
    stack::{Xlstm, XlstmConfig, XlstmState},
};
use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear},
    tensor::{Device, Int, Tensor},
};

/// Configuration for an untied embedding/output-head language model.
#[derive(Config, Debug)]
pub struct XlstmLmConfig {
    /// Number of vocabulary entries.
    pub vocab_size: usize,
    /// Residual stack configuration.
    pub stack: XlstmConfig,
}

impl XlstmLmConfig {
    /// Initializes token embeddings, the residual stack, and vocabulary projection.
    ///
    /// # Errors
    /// Returns an error for invalid vocabulary size or stack settings.
    pub fn init(&self, device: &Device) -> Result<XlstmLm, ConfigError> {
        config::dimensions(self.vocab_size, 1)?;
        let stack = self.stack.init(device)?;
        let width = self.stack.embedding_dim;
        Ok(XlstmLm {
            embedding: EmbeddingConfig::new(self.vocab_size, width)
                .with_initializer(small_init(width))
                .init(device),
            stack,
            head: linear(width, self.vocab_size, small_init(width), device),
        })
    }
}

/// xLSTM language model returning unnormalized vocabulary logits.
#[derive(Module, Debug)]
pub struct XlstmLm {
    /// Learned token embeddings.
    pub embedding: Embedding,
    /// Residual memory stack.
    pub stack: Xlstm,
    /// Untied, bias-free vocabulary projection.
    pub head: Linear,
}

impl XlstmLm {
    /// Maps token IDs `[batch, time]` to logits `[batch, time, vocab_size]`.
    ///
    /// # Panics
    /// Panics for empty dimensions or token IDs outside the vocabulary.
    #[must_use]
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.head
            .forward(self.stack.forward(self.embedding.forward(tokens)))
    }

    /// Processes a chunk and returns state for continued autoregressive decoding.
    ///
    /// # Panics
    /// Panics for invalid token IDs, empty dimensions, or incompatible state.
    #[must_use]
    pub fn forward_with_state(
        &self,
        tokens: Tensor<2, Int>,
        state: Option<XlstmState>,
    ) -> (Tensor<3>, XlstmState) {
        let (hidden, state) = self
            .stack
            .forward_with_state(self.embedding.forward(tokens), state);
        (self.head.forward(hidden), state)
    }

    /// Decodes one token per batch item with optional existing state.
    ///
    /// # Panics
    /// Panics if time is not one, token IDs are invalid, or state is incompatible.
    #[must_use]
    pub fn step(
        &self,
        tokens: Tensor<2, Int>,
        state: Option<XlstmState>,
    ) -> (Tensor<3>, XlstmState) {
        assert_eq!(
            tokens.dims()[1],
            1,
            "language model step requires one token"
        );
        self.forward_with_state(tokens, state)
    }
}
