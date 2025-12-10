//! Sentence-transformer embeddings for semantic search and RAG.
//!
//! This module provides functionality to generate dense vector embeddings from text
//! using sentence-transformer models like E5 and `MiniLM`. These embeddings capture
//! semantic meaning and enable:
//!
//! - Semantic search (finding similar documents)
//! - Clustering and classification
//! - Retrieval-augmented generation (RAG)
//!
//! # Quick Start
//!
//! ```no_run
//! use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
//! use candle_core::Device;
//!
//! # fn example() -> Result<(), metal_candle::error::Error> {
//! // Load model (auto-downloads from HuggingFace on first run)
//! let device = Device::Cpu;
//! let model = EmbeddingModel::from_pretrained(
//!     EmbeddingModelType::E5SmallV2,
//!     device,
//! )?;
//!
//! // Generate embeddings
//! let texts = vec!["Hello world", "Semantic search with Rust"];
//! let embeddings = model.encode(&texts)?;
//!
//! // embeddings is a tensor of shape [2, 384]
//! assert_eq!(embeddings.dims(), &[2, 384]);
//! # Ok(())
//! # }
//! ```
//!
//! # Supported Models
//!
//! - **E5-small-v2** (default): 384 dimensions, 130MB, state-of-the-art retrieval
//! - **MiniLM-L6-v2**: 384 dimensions, 90MB, lightweight and fast
//! - **MPNet-base-v2**: 768 dimensions, 420MB, highest quality
//!
//! # Model Caching
//!
//! Models are automatically cached in `~/.cache/ferris/models/` after first download.
//! Subsequent loads are fast (< 1 second).

mod bert;
mod config;
mod loader;
mod metal_bert;
pub mod pooling;
mod vendored_bert;

pub use config::{EmbeddingConfig, EmbeddingModelType};

use candle_core::{Device, Result as CandleResult, Tensor};
use tokenizers::Tokenizer;

use crate::error::{EmbeddingError, Result};
use bert::BertEncoder;
use loader::{download_model, load_config, load_weights};

/// Main embedding model for generating sentence embeddings.
///
/// This model loads a sentence-transformer from `HuggingFace` Hub and provides
/// a simple API for encoding text into dense vector embeddings.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```no_run
/// use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
/// use candle_core::Device;
///
/// # fn example() -> Result<(), metal_candle::error::Error> {
/// let device = Device::Cpu;
/// let model = EmbeddingModel::from_pretrained(
///     EmbeddingModelType::E5SmallV2,
///     device,
/// )?;
///
/// let texts = vec!["Rust is awesome", "Metal acceleration"];
/// let embeddings = model.encode(&texts)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Computing Similarity
///
/// ```no_run
/// use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
/// use candle_core::Device;
///
/// # fn example() -> Result<(), metal_candle::error::Error> {
/// let device = Device::Cpu;
/// let model = EmbeddingModel::from_pretrained(
///     EmbeddingModelType::E5SmallV2,
///     device,
/// )?;
///
/// let texts = vec![
///     "The cat sits on the mat",
///     "A cat is on a mat",
///     "Python programming language",
/// ];
/// let embeddings = model.encode(&texts)?;
///
/// // Compute cosine similarity (embeddings are already normalized)
/// // similarity = dot product for normalized vectors
/// let vecs = embeddings.to_vec2::<f32>()?;
/// let similarity_01 = dot_product(&vecs[0], &vecs[1]);
/// let similarity_02 = dot_product(&vecs[0], &vecs[2]);
///
/// // Similar sentences should have higher similarity
/// assert!(similarity_01 > similarity_02);
/// # Ok(())
/// # }
///
/// fn dot_product(a: &[f32], b: &[f32]) -> f32 {
///     a.iter().zip(b).map(|(x, y)| x * y).sum()
/// }
/// ```
///
/// ## Custom Configuration
///
/// ```no_run
/// use metal_candle::embeddings::{EmbeddingConfig, EmbeddingModel, EmbeddingModelType};
/// use candle_core::Device;
///
/// # fn example() -> Result<(), metal_candle::error::Error> {
/// let config = EmbeddingConfig {
///     model_type: EmbeddingModelType::AllMpnetBaseV2,
///     normalize: true,  // L2 normalization (recommended)
///     max_seq_length: 256,  // Shorter sequences
/// };
///
/// let device = Device::Cpu;
/// let model = EmbeddingModel::from_config(config, device)?;
/// # Ok(())
/// # }
/// ```
pub struct EmbeddingModel {
    encoder: BertEncoder,
    tokenizer: Tokenizer,
    config: EmbeddingConfig,
    device: Device,
}

impl EmbeddingModel {
    /// Load a sentence-transformer model from `HuggingFace` Hub.
    ///
    /// On first call, downloads the model from `HuggingFace`. Subsequent calls
    /// use the cached model from `~/.cache/ferris/models/`.
    ///
    /// # Arguments
    ///
    /// * `model_type` - The embedding model to load (e.g., `E5SmallV2`)
    /// * `device` - Device to run on (CPU or Metal)
    ///
    /// # Returns
    ///
    /// A new `EmbeddingModel` ready to encode text.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
    /// use candle_core::Device;
    ///
    /// # fn example() -> Result<(), metal_candle::error::Error> {
    /// // Use CPU
    /// let model = EmbeddingModel::from_pretrained(
    ///     EmbeddingModelType::E5SmallV2,
    ///     Device::Cpu,
    /// )?;
    ///
    /// // Use Metal GPU on Apple Silicon (if available, else CPU)
    /// #[cfg(feature = "metal")]
    /// {
    ///     let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    ///     let model = EmbeddingModel::from_pretrained(
    ///         EmbeddingModelType::E5SmallV2,
    ///         device,
    ///     )?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::DownloadFailed`] if the model cannot be downloaded.
    /// Returns [`EmbeddingError::ModelNotFound`] if the model does not exist on `HuggingFace`.
    /// Returns [`EmbeddingError::TokenizerFailed`] if the tokenizer cannot be loaded.
    pub fn from_pretrained(model_type: EmbeddingModelType, device: Device) -> Result<Self> {
        let config = EmbeddingConfig {
            model_type,
            ..Default::default()
        };

        Self::from_config(config, device)
    }

    /// Load model from a custom configuration.
    ///
    /// Provides more control over model loading, including normalization settings
    /// and maximum sequence length.
    ///
    /// # Arguments
    ///
    /// * `config` - Embedding configuration
    /// * `device` - Device to run on (CPU or Metal)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::embeddings::{EmbeddingConfig, EmbeddingModel, EmbeddingModelType};
    /// use candle_core::Device;
    ///
    /// # fn example() -> Result<(), metal_candle::error::Error> {
    /// let config = EmbeddingConfig {
    ///     model_type: EmbeddingModelType::AllMiniLmL6V2,
    ///     normalize: false,  // Skip normalization
    ///     max_seq_length: 128,
    /// };
    ///
    /// let model = EmbeddingModel::from_config(config, Device::Cpu)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::DownloadFailed`] if model download fails.
    /// Returns [`EmbeddingError::InvalidConfig`] if the configuration is invalid.
    /// Returns [`EmbeddingError::TokenizerFailed`] if tokenizer loading fails.
    pub fn from_config(config: EmbeddingConfig, device: Device) -> Result<Self> {
        // Download model if not cached
        let model_dir = download_model(config.model_type)?;

        // Load BERT config
        let bert_config = load_config(&model_dir)?;

        // Load weights
        let vb = load_weights(&model_dir, &device)?;

        // Create BERT encoder
        let encoder = BertEncoder::new(&bert_config, vb, config.normalize)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| EmbeddingError::TokenizerFailed {
                reason: format!("Failed to load tokenizer: {e}"),
            })?;

        // Configure tokenizer truncation and padding
        if let Some(pp) = tokenizer.get_truncation_mut() {
            pp.max_length = config.max_seq_length;
        } else {
            let truncation = tokenizers::TruncationParams {
                max_length: config.max_seq_length,
                ..Default::default()
            };
            tokenizer.with_truncation(Some(truncation)).map_err(|e| {
                EmbeddingError::TokenizerFailed {
                    reason: format!("Failed to configure truncation: {e}"),
                }
            })?;
        }

        // Configure padding to ensure all sequences in a batch have the same length
        let padding = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));

        Ok(Self {
            encoder,
            tokenizer,
            config,
            device,
        })
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// Tokenizes the input texts and encodes them into dense vector embeddings.
    /// If normalization is enabled (default), embeddings will have unit length
    /// and cosine similarity becomes a simple dot product.
    ///
    /// # Arguments
    ///
    /// * `texts` - Array of text strings to embed
    ///
    /// # Returns
    ///
    /// A tensor of shape `[batch_size, embedding_dim]` containing the embeddings.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
    /// use candle_core::Device;
    ///
    /// # fn example() -> Result<(), metal_candle::error::Error> {
    /// let model = EmbeddingModel::from_pretrained(
    ///     EmbeddingModelType::E5SmallV2,
    ///     Device::Cpu,
    /// )?;
    ///
    /// let texts = vec!["Hello world", "Rust embeddings"];
    /// let embeddings = model.encode(&texts)?;
    ///
    /// assert_eq!(embeddings.dims(), &[2, 384]);  // [batch, dimension]
    ///
    /// // Convert to Vec for further processing
    /// let vecs = embeddings.to_vec2::<f32>()?;
    /// println!("First embedding: {:?}", &vecs[0][..5]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::EmptyInput`] if `texts` is empty.
    /// Returns [`EmbeddingError::TokenizationFailed`] if tokenization fails.
    /// Returns an error if encoding or tensor operations fail.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn encode(&self, texts: &[&str]) -> CandleResult<Tensor> {
        if texts.is_empty() {
            return Err(candle_core::Error::Msg(
                "Cannot encode empty text array".to_string(),
            ));
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization failed: {e}")))?;

        // Convert to tensors
        let input_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();

        let attention_mask: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| e.get_attention_mask().to_vec())
            .collect();

        let input_ids = Tensor::new(input_ids, &self.device)?;
        let attention_mask = Tensor::new(attention_mask, &self.device)?;

        // Encode
        self.encoder.encode(&input_ids, &attention_mask)
    }

    /// Get the embedding dimension for this model.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
    /// use candle_core::Device;
    ///
    /// # fn example() -> Result<(), metal_candle::error::Error> {
    /// let model = EmbeddingModel::from_pretrained(
    ///     EmbeddingModelType::E5SmallV2,
    ///     Device::Cpu,
    /// )?;
    ///
    /// assert_eq!(model.dimension(), 384);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.config.model_type.dimension()
    }

    /// Get the model type being used.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
    /// use candle_core::Device;
    ///
    /// # fn example() -> Result<(), metal_candle::error::Error> {
    /// let model = EmbeddingModel::from_pretrained(
    ///     EmbeddingModelType::E5SmallV2,
    ///     Device::Cpu,
    /// )?;
    ///
    /// assert_eq!(model.model_type(), EmbeddingModelType::E5SmallV2);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn model_type(&self) -> EmbeddingModelType {
        self.config.model_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_type, EmbeddingModelType::E5SmallV2);
        assert!(config.normalize);
        assert_eq!(config.max_seq_length, 512);
    }

    #[test]
    fn test_embedding_model_type_properties() {
        assert_eq!(EmbeddingModelType::E5SmallV2.dimension(), 384);
        assert_eq!(EmbeddingModelType::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(EmbeddingModelType::AllMpnetBaseV2.dimension(), 768);
    }

    // Integration tests that download models are in tests/embeddings/
}
