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
pub use sampling::{
    apply_repetition_penalty, sample_token, sample_token_with_metadata, SamplingStrategy,
};

/// Token with generation metadata for streaming.
///
/// Provides rich information about each generated token including
/// the token ID, decoded text (if tokenizer available), probability,
/// and end-of-sequence status.
///
/// # Examples
///
/// ```
/// use metal_candle::inference::StreamToken;
///
/// let token = StreamToken {
///     token_id: 42,
///     text: Some("hello".to_string()),
///     logit: 3.5,
///     probability: 0.85,
///     is_eos: false,
/// };
///
/// println!("Token {}: {} (prob: {:.2}%)",
///          token.token_id,
///          token.text.as_deref().unwrap_or("?"),
///          token.probability * 100.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StreamToken {
    /// The generated token ID.
    pub token_id: u32,

    /// Decoded text representation (if tokenizer available).
    ///
    /// Will be `None` if the generator was created without a tokenizer,
    /// or if decoding fails for this particular token.
    pub text: Option<String>,

    /// Raw logit score before softmax.
    ///
    /// Higher values indicate the model's stronger preference for this token.
    pub logit: f32,

    /// Probability after softmax (0.0-1.0).
    ///
    /// Represents the model's confidence in this token choice.
    /// Values closer to 1.0 indicate higher confidence.
    pub probability: f32,

    /// Whether this token is an end-of-sequence (EOS) token.
    ///
    /// When `true`, generation should typically stop.
    pub is_eos: bool,
}
