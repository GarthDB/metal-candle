//! Inference and text generation for transformer models.
//!
//! This module provides efficient text generation with KV-cache,
//! multiple sampling strategies, and streaming support.
//!
//! # Features
//!
//! - **KV-cache**: Efficient caching of key/value pairs for autoregressive generation
//! - **Sampling**: Greedy, top-k, top-p (nucleus), and temperature sampling
//! - **Generation**: Complete text generation pipeline with configurable parameters
//! - **Streaming**: Token-by-token generation for real-time applications
//!
//! # Examples
//!
//! ```no_run
//! use metal_candle::inference::{Generator, GeneratorConfig, SamplingStrategy};
//! use candle_core::Device;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::Cpu;
//! let config = GeneratorConfig {
//!     max_tokens: 100,
//!     temperature: 0.7,
//!     sampling: SamplingStrategy::TopP { p: 0.9 },
//!     ..Default::default()
//! };
//!
//! // Generator would wrap your model
//! // let generator = Generator::new(model, tokenizer, config, &device)?;
//! // let text = generator.generate("Hello, world!")?;
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod generator;
pub mod sampling;

// Re-export main types
pub use cache::{KVCache, KVCacheConfig};
pub use generator::{Generator, GeneratorConfig};
pub use sampling::{SamplingStrategy, sample_token};

