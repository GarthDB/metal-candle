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
//! - **Text Generation**: Fast inference with multiple sampling strategies and KV-cache
//! - **Metal Acceleration**: Native Metal backend for optimal Apple Silicon performance
//!
//! # Examples
//!
//! ```no_run
//! # use metal_candle::Result;
//! # fn main() -> Result<()> {
//! // Example will be added as APIs are implemented
//! # Ok(())
//! # }
//! ```
//!
//! # Project Status
//!
//! This crate is under active development. APIs are subject to change before v1.0.

// Deny unsafe code by default, but allow it where explicitly justified
#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod backend;
pub mod error;
pub mod models;

// Re-export key types for convenience
pub use backend::{Device, DeviceInfo, DeviceType, TensorExt};
pub use error::{Error, Result};

/// Current version of the crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_exists() {
        assert!(!VERSION.is_empty());
    }
}
