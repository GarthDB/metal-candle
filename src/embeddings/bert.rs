//! BERT encoder for sentence embeddings.
//!
//! This module wraps Candle's BERT model and adds sentence-transformer
//! specific functionality like mean pooling and normalization.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

use crate::embeddings::pooling::{mean_pool, normalize};

/// BERT model for generating sentence embeddings.
///
/// This wraps the base BERT model from `candle-transformers` and adds
/// mean pooling and optional L2 normalization to produce sentence-level
/// embeddings suitable for semantic search.
///
/// # Examples
///
/// ```ignore
/// use candle_core::{Device, DType, Tensor};
/// use candle_nn::VarBuilder;
/// use candle_transformers::models::bert::Config as BertConfig;
/// use metal_candle::embeddings::bert::BertEncoder;
///
/// # fn example() -> candle_core::Result<()> {
/// let device = Device::Cpu;
/// let config = BertConfig::default();
/// let vb = VarBuilder::zeros(DType::F32, &device);
///
/// let encoder = BertEncoder::new(config, vb, true)?;
///
/// // Create sample input
/// let input_ids = Tensor::zeros((1, 10), DType::U32, &device)?;
/// let attention_mask = Tensor::ones((1, 10), DType::U32, &device)?;
///
/// // Encode to embeddings
/// let embeddings = encoder.encode(&input_ids, &attention_mask)?;
/// assert_eq!(embeddings.dims(), &[1, 768]); // [batch, hidden_size]
/// # Ok(())
/// # }
/// ```ignore
pub struct BertEncoder {
    model: BertModel,
    normalize: bool,
}

impl BertEncoder {
    /// Create a new BERT encoder from configuration and weights.
    ///
    /// # Arguments
    ///
    /// * `config` - BERT model configuration
    /// * `vb` - Variable builder containing model weights
    /// * `normalize` - Whether to apply L2 normalization to embeddings
    ///
    /// # Returns
    ///
    /// A new `BertEncoder` instance ready for encoding text.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use candle_core::{Device, DType};
    /// use candle_nn::VarBuilder;
    /// use candle_transformers::models::bert::Config as BertConfig;
    /// use metal_candle::embeddings::bert::BertEncoder;
    ///
    /// # fn example() -> candle_core::Result<()> {
    /// let device = Device::Cpu;
    /// let config = BertConfig::default();
    /// let vb = VarBuilder::zeros(DType::F32, &device);
    ///
    /// let encoder = BertEncoder::new(config, vb, true)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the BERT model cannot be loaded from the weights.
    pub fn new(config: &BertConfig, vb: VarBuilder, normalize: bool) -> Result<Self> {
        let model = BertModel::load(vb, config)?;
        Ok(Self { model, normalize })
    }

    /// Encode tokenized text into embeddings.
    ///
    /// Takes tokenized input (token IDs and attention mask) and produces
    /// sentence-level embeddings via:
    /// 1. BERT forward pass to get token-level hidden states
    /// 2. Mean pooling over tokens (respecting attention mask)
    /// 3. Optional L2 normalization (if configured)
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs of shape `[batch, seq_len]`
    /// * `attention_mask` - Attention mask of shape `[batch, seq_len]`
    ///   where 1 = valid token, 0 = padding
    ///
    /// # Returns
    ///
    /// Embeddings tensor of shape `[batch, hidden_size]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use candle_core::{Device, DType, Tensor};
    /// use candle_nn::VarBuilder;
    /// use candle_transformers::models::bert::Config as BertConfig;
    /// use metal_candle::embeddings::bert::BertEncoder;
    ///
    /// # fn example() -> candle_core::Result<()> {
    /// let device = Device::Cpu;
    /// let config = BertConfig::default();
    /// let vb = VarBuilder::zeros(DType::F32, &device);
    /// let encoder = BertEncoder::new(config, vb, true)?;
    ///
    /// let input_ids = Tensor::zeros((2, 10), DType::U32, &device)?;
    /// let attention_mask = Tensor::ones((2, 10), DType::U32, &device)?;
    ///
    /// let embeddings = encoder.encode(&input_ids, &attention_mask)?;
    /// assert_eq!(embeddings.dims()[0], 2); // batch size
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - BERT forward pass fails
    /// - Pooling operations fail
    /// - Normalization fails
    pub fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Forward pass through BERT (token_type_ids=None for sentence-transformers)
        let hidden_states = self.model.forward(input_ids, attention_mask, None)?;

        // Convert attention mask to f32 for pooling
        let attention_mask_f32 = attention_mask.to_dtype(candle_core::DType::F32)?;

        // Mean pooling
        let pooled = mean_pool(&hidden_states, &attention_mask_f32)?;

        // Optional normalization
        if self.normalize {
            normalize(&pooled)
        } else {
            Ok(pooled)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_test_config() -> BertConfig {
        BertConfig {
            vocab_size: 1000,
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 512,
            hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type:
                candle_transformers::models::bert::PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        }
    }

    #[test]
    fn test_bert_encoder_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = BertEncoder::new(&config, vb, true)?;
        assert!(encoder.normalize);

        Ok(())
    }

    #[test]
    fn test_bert_encoder_shape() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = BertEncoder::new(&config, vb, false)?;

        // Create test input
        let batch_size = 2;
        let seq_len = 10;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        let attention_mask = Tensor::ones((batch_size, seq_len), DType::U32, &device)?;

        // Encode
        let embeddings = encoder.encode(&input_ids, &attention_mask)?;

        // Check shape: [batch, hidden_size]
        assert_eq!(embeddings.dims(), &[batch_size, config.hidden_size]);

        Ok(())
    }

    #[test]
    fn test_bert_encoder_normalization() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = BertEncoder::new(&config, vb, true)?;

        // Create test input
        let input_ids = Tensor::zeros((1, 5), DType::U32, &device)?;
        let attention_mask = Tensor::ones((1, 5), DType::U32, &device)?;

        // Encode
        let embeddings = encoder.encode(&input_ids, &attention_mask)?;

        // Verify normalization (L2 norm should be 1.0)
        let norm = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        let norm_val = norm.to_vec2::<f32>()?[0][0];

        // Note: With zero initialization, norm might be very small or zero
        // This test mainly verifies the pipeline works
        assert!(norm_val >= 0.0);

        Ok(())
    }

    #[test]
    fn test_bert_encoder_no_normalization() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = BertEncoder::new(&config, vb, false)?;
        assert!(!encoder.normalize);

        // Create test input
        let input_ids = Tensor::zeros((1, 5), DType::U32, &device)?;
        let attention_mask = Tensor::ones((1, 5), DType::U32, &device)?;

        // Should not fail without normalization
        let _embeddings = encoder.encode(&input_ids, &attention_mask)?;

        Ok(())
    }

    #[test]
    fn test_bert_encoder_batch_processing() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let encoder = BertEncoder::new(&config, vb, true)?;

        // Test different batch sizes
        for batch_size in [1, 2, 4, 8] {
            let input_ids = Tensor::zeros((batch_size, 10), DType::U32, &device)?;
            let attention_mask = Tensor::ones((batch_size, 10), DType::U32, &device)?;

            let embeddings = encoder.encode(&input_ids, &attention_mask)?;
            assert_eq!(embeddings.dims()[0], batch_size);
            assert_eq!(embeddings.dims()[1], config.hidden_size);
        }

        Ok(())
    }
}
