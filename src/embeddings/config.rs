//! Configuration for embedding models.
//!
//! This module defines the supported embedding models and their configurations.

use serde::{Deserialize, Serialize};

/// Supported embedding model types.
///
/// These are sentence-transformer models that generate dense vector representations
/// of text for semantic search and retrieval-augmented generation (RAG).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbeddingModelType {
    /// intfloat/e5-small-v2 (384 dimensions, 130MB)
    ///
    /// Default model with excellent retrieval performance.
    /// Competitive with much larger models on MTEB benchmarks.
    #[default]
    E5SmallV2,

    /// sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, 90MB)
    ///
    /// Lightweight model with good general-purpose performance.
    AllMiniLmL6V2,

    /// sentence-transformers/all-mpnet-base-v2 (768 dimensions, 420MB)
    ///
    /// Larger model with higher quality embeddings.
    AllMpnetBaseV2,
}

impl EmbeddingModelType {
    /// Get the `HuggingFace` model ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::embeddings::EmbeddingModelType;
    ///
    /// assert_eq!(
    ///     EmbeddingModelType::E5SmallV2.model_id(),
    ///     "intfloat/e5-small-v2"
    /// );
    /// ```
    #[must_use]
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::E5SmallV2 => "intfloat/e5-small-v2",
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMpnetBaseV2 => "sentence-transformers/all-mpnet-base-v2",
        }
    }

    /// Get the embedding dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::embeddings::EmbeddingModelType;
    ///
    /// assert_eq!(EmbeddingModelType::E5SmallV2.dimension(), 384);
    /// assert_eq!(EmbeddingModelType::AllMpnetBaseV2.dimension(), 768);
    /// ```
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self {
            Self::E5SmallV2 | Self::AllMiniLmL6V2 => 384,
            Self::AllMpnetBaseV2 => 768,
        }
    }
}

/// Configuration for embedding models.
///
/// Controls how text is encoded into embeddings.
///
/// # Examples
///
/// ```
/// use metal_candle::embeddings::{EmbeddingConfig, EmbeddingModelType};
///
/// // Use default configuration
/// let config = EmbeddingConfig::default();
/// assert_eq!(config.model_type, EmbeddingModelType::E5SmallV2);
/// assert!(config.normalize);
///
/// // Custom configuration
/// let config = EmbeddingConfig {
///     model_type: EmbeddingModelType::AllMpnetBaseV2,
///     normalize: true,
///     max_seq_length: 512,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// The embedding model to use.
    pub model_type: EmbeddingModelType,

    /// Whether to apply L2 normalization to embeddings.
    ///
    /// Recommended for cosine similarity computations.
    /// When `true`, all embeddings will have unit length.
    pub normalize: bool,

    /// Maximum sequence length in tokens.
    ///
    /// Longer sequences will be truncated.
    pub max_seq_length: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_type: EmbeddingModelType::default(),
            normalize: true,
            max_seq_length: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_ids() {
        assert_eq!(
            EmbeddingModelType::E5SmallV2.model_id(),
            "intfloat/e5-small-v2"
        );
        assert_eq!(
            EmbeddingModelType::AllMiniLmL6V2.model_id(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!(
            EmbeddingModelType::AllMpnetBaseV2.model_id(),
            "sentence-transformers/all-mpnet-base-v2"
        );
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(EmbeddingModelType::E5SmallV2.dimension(), 384);
        assert_eq!(EmbeddingModelType::AllMiniLmL6V2.dimension(), 384);
        assert_eq!(EmbeddingModelType::AllMpnetBaseV2.dimension(), 768);
    }

    #[test]
    fn test_default_model_type() {
        assert_eq!(EmbeddingModelType::default(), EmbeddingModelType::E5SmallV2);
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_type, EmbeddingModelType::E5SmallV2);
        assert!(config.normalize);
        assert_eq!(config.max_seq_length, 512);
    }

    #[test]
    fn test_custom_config() {
        let config = EmbeddingConfig {
            model_type: EmbeddingModelType::AllMpnetBaseV2,
            normalize: false,
            max_seq_length: 256,
        };
        assert_eq!(config.model_type, EmbeddingModelType::AllMpnetBaseV2);
        assert!(!config.normalize);
        assert_eq!(config.max_seq_length, 256);
    }
}
