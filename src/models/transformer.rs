//! Generic transformer components for building neural networks.
//!
//! This module provides reusable building blocks for transformer architectures,
//! including attention mechanisms, feed-forward networks, and position embeddings.

use crate::backend::TensorExt;
use crate::error::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{linear, linear_no_bias, ops, Linear, Module, VarBuilder};
use std::sync::Arc;

/// Rotary Position Embeddings (`RoPE`) for attention mechanisms.
///
/// `RoPE` encodes position information by rotating query and key vectors,
/// allowing the model to learn relative positions effectively.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::models::transformer::RotaryEmbedding;
/// use candle_core::Device;
///
/// let rope = RotaryEmbedding::new(64, 2048, 10_000.0, &Device::Cpu)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// Creates a new rotary embedding.
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    /// * `max_seq_len` - Maximum sequence length
    /// * `theta` - Base for the geometric progression (typically 10,000.0)
    /// * `device` - Device to place the embeddings on
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    #[allow(clippy::cast_precision_loss)] // head_dim and max_seq_len are small, precision loss acceptable
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let head_dim_f = head_dim as f64;
        
        // Create frequency tensor: theta^(-2i/d) for i in 0..d/2
        let inv_freq: Vec<f64> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim_f))
            .collect();
        
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?;

        // Create position indices: [0, 1, 2, ..., max_seq_len-1]
        let positions: Vec<f64> = (0..max_seq_len).map(|i| i as f64).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len,), device)?;

        // Outer product: positions ⊗ inv_freq
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        
        // Concatenate [freqs, freqs] to match head_dim
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

        Ok(Self {
            sin: emb.sin()?,
            cos: emb.cos()?,
        })
    }

    /// Applies rotary embeddings to query or key tensors.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `(batch, seq_len, num_heads, head_dim)`
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    ///
    /// Rotated tensor of the same shape
    ///
    /// # Errors
    ///
    /// Returns an error if tensor shapes are incompatible.
    pub fn apply_rotary_emb(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let (_batch, _seq, _num_heads, head_dim) = x.dims4()?;
        let dtype = x.dtype();
        
        // Get sin/cos for this sequence length and convert to input dtype
        let sin = self.sin.i(..seq_len)?.unsqueeze(0)?.unsqueeze(2)?.to_dtype(dtype)?; // (1, seq, 1, head_dim)
        let cos = self.cos.i(..seq_len)?.unsqueeze(0)?.unsqueeze(2)?.to_dtype(dtype)?; // (1, seq, 1, head_dim)

        // Split x into two halves
        let x1 = x.i((.., .., .., ..head_dim / 2))?;
        let x2 = x.i((.., .., .., head_dim / 2..))?;

        // Apply rotation: [x1, x2] -> [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let rotated1 = (x1.broadcast_mul(&cos.i((.., .., .., ..head_dim / 2))?)? -
                       x2.broadcast_mul(&sin.i((.., .., .., ..head_dim / 2))?))?;
        let rotated2 = (x1.broadcast_mul(&sin.i((.., .., .., head_dim / 2..))?)? +
                       x2.broadcast_mul(&cos.i((.., .., .., head_dim / 2..))?))?;

        Tensor::cat(&[&rotated1, &rotated2], 3).map_err(Into::into)
    }
}

/// Multi-head attention with grouped-query attention support.
///
/// Implements scaled dot-product attention with optional causal masking
/// and support for key-value caching.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::models::transformer::Attention;
/// use candle_nn::VarBuilder;
/// use candle_core::Device;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
/// 
/// let attention = Attention::new(
///     768,      // hidden_size
///     12,       // num_heads
///     Some(4),  // num_kv_heads (grouped-query attention)
///     2048,     // max_seq_len
///     10_000.0, // rope_theta
///     vb.pp("attn")?
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Arc<RotaryEmbedding>,
}

impl Attention {
    /// Creates a new attention layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Model hidden dimension
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key-value heads (for grouped-query attention)
    /// * `max_seq_len` - Maximum sequence length
    /// * `rope_theta` - Rotary embedding base
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    ///
    /// Returns an error if layer initialization fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        max_seq_len: usize,
        rope_theta: f64,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let num_kv_heads = num_kv_heads.unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;

        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let rope = Arc::new(RotaryEmbedding::new(
            head_dim,
            max_seq_len,
            rope_theta,
            vb.device(),
        )?);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }

    /// Returns the dimension of each attention head.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Performs forward pass through the attention layer.
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
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to (batch, seq_len, num_heads, head_dim)
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply rotary embeddings
        let q = self.rope.apply_rotary_emb(&q, seq_len)?;
        let k = self.rope.apply_rotary_emb(&k, seq_len)?;

        // Transpose to (batch, num_heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Repeat K, V for grouped-query attention
        let k = Self::repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = Self::repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Compute attention: softmax(QK^T / sqrt(d_k))V
        let att_weights = q.matmul(&k.transpose(2, 3)?)?;
        #[allow(clippy::cast_precision_loss)] // head_dim is small, precision loss acceptable
        let att_weights = (att_weights / (self.head_dim as f64).sqrt())?;

        // Apply attention mask if provided
        let att_weights = if let Some(mask) = attention_mask {
            att_weights.broadcast_add(mask)?
        } else {
            att_weights
        };

        let att_weights = att_weights.softmax_stable()?;

        // Apply attention to values
        let out = att_weights.matmul(&v)?;

        // Reshape back to (batch, seq_len, hidden_size)
        let out = out.transpose(1, 2)?;
        let out = out.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&out).map_err(Into::into)
    }

    /// Repeats key/value tensors for grouped-query attention.
    fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x);
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        
        x.unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
            .map_err(Into::into)
    }
}

/// Feed-forward network (MLP) layer.
///
/// Implements the position-wise feed-forward network used in transformers:
/// `FFN(x) = activation(xW1 + b1)W2 + b2`
#[allow(clippy::struct_field_names)] // gate_proj, up_proj, down_proj are standard transformer naming
///
/// # Examples
///
/// ```no_run
/// use metal_candle::models::transformer::MLP;
/// use candle_nn::VarBuilder;
/// use candle_core::Device;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
/// 
/// let mlp = MLP::new(768, 3072, vb.pp("mlp"))?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    /// Creates a new MLP layer.
    ///
    /// Uses `SwiGLU` activation: `SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)`
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden layer dimension
    /// * `vb` - Variable builder for loading weights
    ///
    /// # Errors
    ///
    /// Returns an error if layer initialization fails.
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: &VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Performs forward pass through the MLP.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `(batch, seq_len, hidden_size)`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `(batch, seq_len, hidden_size)`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU activation: silu(gate) * up
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 128, 10_000.0, &device);
        assert!(rope.is_ok());
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 128, 10_000.0, &device).unwrap();
        
        // Create dummy tensor: (batch=2, seq=10, heads=8, head_dim=64)
        let x = Tensor::randn(0f32, 1f32, (2, 10, 8, 64), &device).unwrap();
        let rotated = rope.apply_rotary_emb(&x, 10);
        
        if let Err(e) = &rotated {
            eprintln!("Error in apply_rotary_emb: {e}");
        }
        assert!(rotated.is_ok(), "apply_rotary_emb failed: {rotated:?}");
        let rotated = rotated.unwrap();
        assert_eq!(rotated.dims(), &[2, 10, 8, 64]);
    }

    // More comprehensive tests would require actual weight loading
    // which we'll add in integration tests
}

