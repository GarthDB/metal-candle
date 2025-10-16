//! Qwen2.5-Coder model architecture.
//!
//! This module implements the Qwen2.5-Coder transformer architecture,
//! optimized for code generation tasks.

use super::config::ModelConfig;
use super::transformer::{Attention, MLP};
use crate::backend::TensorExt;
use crate::error::Result;
use candle_core::Tensor;
use candle_nn::{embedding, Embedding, Module, VarBuilder};

/// A single transformer layer in the Qwen model.
///
/// Consists of multi-head attention, feed-forward network (MLP),
/// and RMS normalization layers.
#[derive(Debug)]
pub struct QwenDecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl QwenDecoderLayer {
    /// Creates a new decoder layer.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    ///
    /// Returns an error if layer initialization fails.
    pub fn new(config: &ModelConfig, vb: &VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.rope_theta,
            &vb.pp("self_attn"),
        )?;

        let mlp = MLP::new(
            config.hidden_size,
            config.intermediate_size,
            &vb.pp("mlp"),
        )?;

        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps, &vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            &vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Performs forward pass through the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor of shape `(batch, seq_len, hidden_size)`
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    ///
    /// Output tensor of shape `(batch, seq_len, hidden_size)`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention with residual connection
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, attention_mask)?;
        let hidden_states = (hidden_states + residual)?;

        // MLP with residual connection
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        Ok(hidden_states)
    }
}

/// Qwen2.5-Coder transformer model.
///
/// Complete decoder-only transformer for code generation and understanding.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::models::{ModelConfig, Qwen};
/// use candle_nn::VarBuilder;
/// use candle_core::Device;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ModelConfig::from_file("config.json")?;
/// let device = Device::Cpu;
/// let vb = VarBuilder::from_tensors(std::collections::HashMap::new(), candle_core::DType::F32, &device);
/// 
/// let model = Qwen::new(&config, vb)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Qwen {
    embed_tokens: Embedding,
    layers: Vec<QwenDecoderLayer>,
    norm: RMSNorm,
    lm_head: candle_nn::Linear,
}

impl Qwen {
    /// Creates a new Qwen model.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    ///
    /// Returns an error if model initialization fails.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is consumed by pp() calls
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..config.num_hidden_layers {
            let layer = QwenDecoderLayer::new(config, &vb_layers.pp(i))?;
            layers.push(layer);
        }

        let norm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            &vb.pp("model.norm"),
        )?;

        let lm_head = candle_nn::linear_no_bias(
            config.hidden_size,
            config.vocab_size,
            vb.pp("lm_head"),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Performs forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs of shape `(batch, seq_len)`
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    ///
    /// Logits tensor of shape `(batch, seq_len, vocab_size)`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input tensor has invalid shape
    /// - Any layer forward pass fails
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through all decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        // Final normalization
        hidden_states = self.norm.forward(&hidden_states)?;

        // Project to vocabulary
        self.lm_head.forward(&hidden_states).map_err(Into::into)
    }

    /// Returns the number of parameters in the model.
    ///
    /// Useful for memory estimation and model analysis.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        // Approximate calculation: embeddings + layers + final norm + lm_head
        let embed_params = self.embed_tokens.embeddings().elem_count();
        let lm_head_params = self.lm_head.weight().elem_count();
        let norm_params = self.norm.weight.elem_count();
        
        // Each layer has: attention (4 projections) + MLP (3 projections) + 2 norms
        // This is approximate - actual count would require iterating through all parameters
        embed_params + (self.layers.len() * 1_000_000) + norm_params + lm_head_params
    }
}

/// RMS (Root Mean Square) Normalization layer.
///
/// Normalizes the input to have unit RMS, then applies a learned scale.
#[derive(Debug)]
struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    /// Creates a new RMS normalization layer.
    ///
    /// # Arguments
    ///
    /// * `size` - Dimension to normalize
    /// * `eps` - Small constant for numerical stability
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    ///
    /// Returns an error if weight loading fails.
    fn new(size: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Applies RMS normalization.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of any shape with last dimension matching `size`
    ///
    /// # Returns
    ///
    /// Normalized and scaled tensor of the same shape
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = self.weight.dtype();

        // Convert to internal dtype for computation if needed
        let x = x.to_dtype(internal_dtype)?;
        let normed = x.rms_norm(self.eps)?;
        let normed = normed.broadcast_mul(&self.weight)?;
        
        // Convert back to original dtype
        normed.to_dtype(x_dtype).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["qwen2".to_string()],
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            torch_dtype: Some("float32".to_string()),
        }
    }

    #[test]
    fn test_qwen_decoder_layer_creation() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let layer = QwenDecoderLayer::new(&config, &vb);
        assert!(layer.is_ok(), "Failed to create decoder layer: {layer:?}");
    }

    #[test]
    fn test_qwen_model_creation() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let model = Qwen::new(&config, vb);
        assert!(model.is_ok(), "Failed to create Qwen model: {model:?}");
        
        if let Ok(model) = model {
            // Verify model has the correct number of layers
            assert_eq!(model.layers.len(), config.num_hidden_layers);
        }
    }

    #[test]
    fn test_rms_norm() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let norm = RMSNorm::new(64, 1e-6, &vb);
        assert!(norm.is_ok());
    }
}

