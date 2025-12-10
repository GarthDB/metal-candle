//! metal-candle: Production-quality Rust ML for Apple Silicon
//!
//! `metal-candle` is a machine learning library built on [Candle](https://github.com/huggingface/candle)
//! with Metal backend, providing `LoRA` training, model loading, and text generation
//! for transformer models on Apple Silicon.
//!
//! # Features
//!
//! - **`LoRA` Training**: Fine-tune transformer models efficiently using Low-Rank Adaptation
//! - **Model Loading**: Support for safetensors format with extensibility for others
//! - **Text Generation**: High-level [`Generator`](inference::Generator) API with streaming, repetition penalty, and stop conditions
//! - **Sampling Strategies**: Greedy, Top-k, Top-p (nucleus), and Temperature sampling
//! - **Metal Acceleration**: Native Metal backend for optimal Apple Silicon performance
//!
//! # Examples
//!
//! ## Text Generation
//!
//! ```
//! use metal_candle::inference::{GeneratorConfig, SamplingStrategy};
//!
//! // Configure generation
//! let gen_config = GeneratorConfig {
//!     max_tokens: 128,
//!     sampling: SamplingStrategy::TopP { p: 0.95 },
//!     repetition_penalty: 1.1,
//!     ..Default::default()
//! };
//!
//! // With a loaded model, you would use:
//! // let mut generator = Generator::new(Box::new(model), gen_config)?;
//! // let output = generator.generate(&input_ids)?;
//! ```
//!
//! # Project Status
//!
//! v1.1.0: Production-ready text generation API with comprehensive testing and documentation.

// Deny unsafe code by default, but allow it where explicitly justified
#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod backend;
pub mod error;
#[cfg(feature = "graph")]
pub mod graph;
pub mod inference;
pub mod models;
pub mod training;

#[cfg(feature = "embeddings")]
pub mod embeddings;

// Re-export key types for convenience
pub use backend::{Device, DeviceInfo, DeviceType, TensorExt};
pub use error::{Error, Result};
pub use inference::{
    apply_repetition_penalty, sample_token, Generator, GeneratorConfig, KVCache, KVCacheConfig,
    SamplingStrategy,
};
pub use training::{
    cross_entropy_loss, cross_entropy_loss_with_smoothing, AdamW, AdamWConfig, LRScheduler,
    LoRAAdapter, LoRAAdapterConfig, LoRAConfig, LoRALayer, StepMetrics, TargetModule, Trainer,
    TrainingConfig, TrainingStep,
};

/// Current version of the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_exists() {
        // VERSION is a compile-time constant from CARGO_PKG_VERSION
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }
}
