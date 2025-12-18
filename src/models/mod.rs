//! Model loading and architecture implementations.
//!
//! This module provides utilities for loading ML models from various formats
//! (primarily safetensors) and implementing transformer architectures.
//!
//! # Model Loading
//!
//! Load models using [`ModelLoader`]:
//!
//! ```no_run
//! use metal_candle::models::ModelLoader;
//! use metal_candle::Device;
//!
//! let device = Device::new_with_fallback(0);
//! let loader = ModelLoader::new(device);
//! let model = loader.load("model.safetensors")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Architecture Components
//!
//! Build transformer models using reusable components (coming in Phase 2 part 2).

use crate::error::Result;
use candle_core::{Device, Tensor};

pub mod config;
pub mod loader;
pub mod qwen;
mod qwen_adapter_tests;
pub mod transformer;

// Re-export commonly used types
pub use config::ModelConfig;
pub use loader::ModelLoader;
pub use qwen::Qwen;

/// Trait for language models that can generate text.
///
/// This trait provides a common interface for different model architectures
/// to be used with the text generation pipeline.
pub trait LanguageModel {
    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs, shape: `(batch, seq_len)`
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    ///
    /// Logits tensor of shape `(batch, seq_len, vocab_size)`
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;

    /// Returns the device the model is on.
    fn device(&self) -> &Device;

    /// Returns the vocabulary size of the model.
    fn vocab_size(&self) -> usize;
}

/// Implement `LanguageModel` for `Qwen`.
impl LanguageModel for Qwen {
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward(input_ids, attention_mask)
    }

    fn device(&self) -> &Device {
        self.lm_head.weight().device()
    }

    fn vocab_size(&self) -> usize {
        self.lm_head.weight().dims()[0]
    }
}
