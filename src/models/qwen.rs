//! Qwen2.5-Coder model architecture.
//!
//! This module implements the Qwen2.5-Coder transformer architecture,
//! optimized for code generation tasks.

use super::config::ModelConfig;
use super::transformer::{Attention, MLP};
use crate::backend::TensorExt;
use crate::error::Result;
use crate::training::{ApplyAdapter, LoRAAdapter, TargetModule};
use candle_core::Tensor;
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::sync::Arc;

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
    /// Layer index in the model (for `LoRA` adapter lookup)
    layer_idx: usize,
}

impl QwenDecoderLayer {
    /// Creates a new decoder layer.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `layer_idx` - Index of this layer in the model
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    ///
    /// Returns an error if layer initialization fails.
    pub fn new(config: &ModelConfig, layer_idx: usize, vb: &VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.rope_theta,
            &vb.pp("self_attn"),
        )?;

        let mlp = MLP::new(config.hidden_size, config.intermediate_size, &vb.pp("mlp"))?;

        let input_layernorm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            &vb.pp("input_layernorm"),
        )?;
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
            layer_idx,
        })
    }

    /// Performs forward pass through the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor of shape `(batch, seq_len, hidden_size)`
    /// * `attention_mask` - Optional attention mask
    /// * `lora_adapter` - Optional `LoRA` adapter to apply
    ///
    /// # Returns
    ///
    /// Output tensor of shape `(batch, seq_len, hidden_size)`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    ///
    /// # Implementation Note
    ///
    /// `LoRA` is currently applied at the layer output level. For full `LoRA` support
    /// at each projection (Q/K/V/O for attention, Gate/Up/Down for MLP), we need
    /// to modify the Attention and MLP forward passes. This is tracked for
    /// optimization in a future update.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        lora_adapter: Option<&LoRAAdapter>,
    ) -> Result<Tensor> {
        // Self-attention with residual connection
        let residual = hidden_states;
        let attn_input = self.input_layernorm.forward(hidden_states)?;
        let mut attn_output = self.self_attn.forward(&attn_input, attention_mask)?;

        // Apply LoRA to attention output projection if adapter is present
        // LoRA takes the INPUT to o_proj (which is the attention output before o_proj)
        // and produces a delta to add to the o_proj output
        // Note: For full LoRA support, we'd need to apply it within the attention mechanism
        // to Q/K/V projections. This is a simplified version that only applies to the final output.
        if let Some(adapter) = lora_adapter {
            // For now, we apply LoRA using the attention output as input
            // This is a simplification - proper implementation would apply inside Attention
            if let Some(o_delta) =
                adapter.forward(self.layer_idx, &TargetModule::OProj, &attn_input)?
            {
                attn_output = (&attn_output + &o_delta)?;
            }
        }

        let hidden_states = (attn_output + residual)?;

        // MLP with residual connection
        let residual = &hidden_states;
        let mlp_input = self.post_attention_layernorm.forward(&hidden_states)?;
        let mut mlp_output = self.mlp.forward(&mlp_input)?;

        // Apply LoRA to MLP down projection if adapter is present
        // LoRA takes the INPUT to down_proj and produces a delta
        if let Some(adapter) = lora_adapter {
            // For now, we apply LoRA using the MLP input
            // Proper implementation would apply inside MLP to each projection
            if let Some(down_delta) =
                adapter.forward(self.layer_idx, &TargetModule::DownProj, &mlp_input)?
            {
                mlp_output = (&mlp_output + &down_delta)?;
            }
        }

        let hidden_states = (mlp_output + residual)?;

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
    pub(crate) lm_head: candle_nn::Linear,
    /// Currently applied `LoRA` adapter (if any)
    lora_adapter: Option<Arc<LoRAAdapter>>,
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
            let layer = QwenDecoderLayer::new(config, i, &vb_layers.pp(i))?;
            layers.push(layer);
        }

        let norm = RMSNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            &vb.pp("model.norm"),
        )?;

        let lm_head =
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            lora_adapter: None,
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
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through all decoder layers
        let lora_adapter = self.lora_adapter.as_deref();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, lora_adapter)?;
        }

        // Final normalization
        hidden_states = self.norm.forward(&hidden_states)?;

        // Project to vocabulary
        self.lm_head.forward(&hidden_states).map_err(Into::into)
    }

    /// Returns the number of layers in the model.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
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

impl ApplyAdapter for Qwen {
    fn apply_adapter(&mut self, adapter: Arc<LoRAAdapter>) -> Result<()> {
        // Basic validation: check if adapter has layers for this model's layer count
        if adapter.num_layers() != self.num_layers() {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: format!(
                    "Adapter has {} layers, but model has {} layers",
                    adapter.num_layers(),
                    self.num_layers()
                ),
            }
            .into());
        }

        // Store the adapter
        self.lora_adapter = Some(adapter);
        Ok(())
    }

    fn remove_adapter(&mut self) -> Result<()> {
        // Clear the adapter
        self.lora_adapter = None;
        Ok(())
    }

    fn has_adapter(&self) -> bool {
        self.lora_adapter.is_some()
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

        let layer = QwenDecoderLayer::new(&config, 0, &vb);
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

    #[test]
    fn test_rms_norm_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let norm = RMSNorm::new(128, 1e-6, &vb).unwrap();
        let input = Tensor::randn(0f32, 1f32, (2, 16, 128), &device).unwrap();

        let output = norm.forward(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().dims(), &[2, 16, 128]);
    }

    #[test]
    fn test_decoder_layer_forward() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let layer = QwenDecoderLayer::new(&config, 0, &vb).unwrap();
        let input = Tensor::zeros((1, 8, config.hidden_size), DType::F32, &device).unwrap();

        let output = layer.forward(&input, None, None);
        assert!(output.is_ok());
        assert_eq!(output.unwrap().dims(), &[1, 8, config.hidden_size]);
    }

    #[test]
    fn test_qwen_forward() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Qwen::new(&config, vb).unwrap();
        let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();

        let logits = model.forward(&input_ids, None);
        assert!(logits.is_ok());

        let logits = logits.unwrap();
        let (batch, seq, vocab) = logits.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(seq, 8);
        assert_eq!(vocab, config.vocab_size);
    }

    #[test]
    fn test_qwen_num_layers() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Qwen::new(&config, vb).unwrap();
        assert_eq!(model.num_layers(), config.num_hidden_layers);
    }

    #[test]
    fn test_qwen_num_parameters() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Qwen::new(&config, vb).unwrap();
        let params = model.num_parameters();

        // Should have non-zero parameters
        assert!(params > 0);

        // Rough estimate: vocab embeddings + lm_head should be most of it
        let expected_min = config.vocab_size * config.hidden_size * 2;
        assert!(params > expected_min);
    }

    #[test]
    fn test_qwen_different_batch_sizes() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Qwen::new(&config, vb).unwrap();

        for batch_size in &[1, 2, 4] {
            let input_ids = Tensor::zeros((*batch_size, 8), DType::U32, &device).unwrap();
            let logits = model.forward(&input_ids, None);
            assert!(logits.is_ok());
            assert_eq!(logits.unwrap().dims()[0], *batch_size);
        }
    }
}
