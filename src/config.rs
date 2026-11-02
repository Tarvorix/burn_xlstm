//! Validation errors for model configuration.
use std::{backtrace::Backtrace, error::Error, fmt};

/// An invalid model dimension or numerical setting.
#[derive(Debug)]
pub struct ConfigError {
    field: &'static str,
    message: String,
    backtrace: Backtrace,
}

impl ConfigError {
    pub(crate) fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns the offending configuration field.
    #[must_use]
    pub fn field(&self) -> &'static str {
        self.field
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {}: {}\n{}",
            self.field, self.message, self.backtrace
        )
    }
}
impl Error for ConfigError {}

pub(crate) fn dimensions(width: usize, heads: usize) -> Result<(), ConfigError> {
    if width == 0 || width > u32::MAX as usize / 8 {
        return Err(ConfigError::new(
            "embedding_dim",
            format!("must be in 1..={}, got {width}", u32::MAX / 8),
        ));
    }
    if heads == 0 || !width.is_multiple_of(heads) {
        return Err(ConfigError::new(
            "num_heads",
            format!("{heads} must be positive and divide {width}"),
        ));
    }
    Ok(())
}

pub(crate) fn dropout(probability: f64) -> Result<(), ConfigError> {
    if !probability.is_finite() || !(0.0..1.0).contains(&probability) {
        return Err(ConfigError::new(
            "dropout",
            format!("must be finite and in [0, 1), got {probability}"),
        ));
    }
    Ok(())
}

pub(crate) fn depth(index: usize, blocks: usize) -> Result<(), ConfigError> {
    if blocks == 0 || index >= blocks || blocks > u32::MAX as usize {
        return Err(ConfigError::new(
            "num_blocks",
            format!("invalid block index {index} for depth {blocks}"),
        ));
    }
    Ok(())
}

pub(crate) fn kernel(size: usize) -> Result<(), ConfigError> {
    if size > u32::MAX as usize / 8 {
        return Err(ConfigError::new(
            "conv_kernel",
            "kernel size exceeds supported dimensions",
        ));
    }
    Ok(())
}

pub(crate) fn rounded(width: usize, numerator: usize, denominator: usize) -> usize {
    // The original xLSTM rounds expanded widths up to a multiple of 64.
    const MULTIPLE: usize = 64;
    (width * numerator).div_ceil(denominator * MULTIPLE) * MULTIPLE
}
