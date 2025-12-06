//! Pooling and normalization operations for embeddings.
//!
//! This module provides functions for transforming BERT hidden states into
//! sentence embeddings via mean pooling and L2 normalization.

use candle_core::{Result, Tensor};

/// Mean pooling: average token embeddings, weighted by attention mask.
///
/// This function computes the mean of token embeddings while properly handling
/// padding tokens by using the attention mask. Only non-padding tokens contribute
/// to the mean.
///
/// # Arguments
///
/// * `hidden_states` - BERT output tensor of shape `[batch, seq_len, hidden_size]`
/// * `attention_mask` - Attention mask of shape `[batch, seq_len]` where 1 indicates
///   a valid token and 0 indicates padding
///
/// # Returns
///
/// A tensor of shape `[batch, hidden_size]` containing the pooled embeddings.
///
/// # Examples
///
/// ```no_run
/// use candle_core::{Tensor, Device, DType};
/// use metal_candle::embeddings::pooling::mean_pool;
///
/// let device = Device::Cpu;
/// let hidden_states = Tensor::randn(0f32, 1f32, (2, 10, 384), &device)?;
/// let attention_mask = Tensor::ones((2, 10), DType::F32, &device)?;
///
/// let pooled = mean_pool(&hidden_states, &attention_mask)?;
/// assert_eq!(pooled.dims(), &[2, 384]);
/// # Ok::<(), candle_core::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Tensor shapes are incompatible
/// - Tensor operations fail
pub fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // hidden_states: [batch, seq_len, hidden_size]
    // attention_mask: [batch, seq_len]

    // Expand attention_mask to [batch, seq_len, hidden_size]
    let mask_expanded = attention_mask
        .unsqueeze(2)?
        .expand(hidden_states.shape())?
        .to_dtype(hidden_states.dtype())?;

    // Multiply hidden_states by mask (zero out padding tokens)
    let masked = hidden_states.mul(&mask_expanded)?;

    // Sum across sequence dimension
    let sum = masked.sum(1)?;

    // Sum mask to get counts (avoid division by zero)
    let mask_sum = mask_expanded.sum(1)?.clamp(1e-9, f64::from(f32::MAX))?;

    // Divide to get mean
    sum.broadcast_div(&mask_sum)
}

/// L2 normalization for cosine similarity.
///
/// Normalizes embeddings to unit length, making them suitable for cosine similarity
/// computations (which becomes a simple dot product for normalized vectors).
///
/// # Arguments
///
/// * `embeddings` - Embeddings tensor of shape `[batch, hidden_size]`
///
/// # Returns
///
/// A tensor of the same shape with L2-normalized embeddings (unit length).
///
/// # Examples
///
/// ```no_run
/// use candle_core::{Tensor, Device};
/// use metal_candle::embeddings::pooling::normalize;
///
/// let device = Device::Cpu;
/// let embeddings = Tensor::randn(0f32, 1f32, (2, 384), &device)?;
/// let normalized = normalize(&embeddings)?;
///
/// // Verify unit length
/// let norms = normalized.sqr()?.sum_keepdim(1)?.sqrt()?;
/// let norms_vec = norms.to_vec2::<f32>()?;
/// for norm in norms_vec[0].iter() {
///     assert!((norm - 1.0).abs() < 1e-5);
/// }
/// # Ok::<(), candle_core::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if tensor operations fail.
///
/// # Panics
///
/// This function does not panic.
pub fn normalize(embeddings: &Tensor) -> Result<Tensor> {
    // embeddings: [batch, hidden_size]

    // Compute L2 norm: sqrt(sum(x^2))
    let norm = embeddings
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12, f64::MAX)?; // Avoid division by zero

    // Divide by norm
    embeddings.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::{DType, Device};

    #[test]
    fn test_mean_pool_basic() -> Result<()> {
        let device = Device::Cpu;

        // Create simple test tensors
        let hidden_states = Tensor::ones((2, 3, 4), DType::F32, &device)?;
        let attention_mask = Tensor::ones((2, 3), DType::F32, &device)?;

        let pooled = mean_pool(&hidden_states, &attention_mask)?;

        // Should average to ones
        assert_eq!(pooled.dims(), &[2, 4]);
        let values = pooled.to_vec2::<f32>()?;
        for batch in &values {
            for &val in batch {
                assert_relative_eq!(val, 1.0, epsilon = 1e-5);
            }
        }

        Ok(())
    }

    #[test]
    fn test_mean_pool_with_padding() -> Result<()> {
        let device = Device::Cpu;

        // Create test data: [batch=1, seq_len=3, hidden_size=2]
        // Values: [[1, 1], [2, 2], [0, 0]] with last token being padding
        let hidden_states = Tensor::new(&[[[1.0f32, 1.0], [2.0, 2.0], [0.0, 0.0]]], &device)?;
        let attention_mask = Tensor::new(&[[1.0f32, 1.0, 0.0]], &device)?;

        let pooled = mean_pool(&hidden_states, &attention_mask)?;

        // Should average to [1.5, 1.5] (mean of [1,1] and [2,2], ignoring [0,0])
        assert_eq!(pooled.dims(), &[1, 2]);
        let values = pooled.to_vec2::<f32>()?;
        assert_relative_eq!(values[0][0], 1.5, epsilon = 1e-5);
        assert_relative_eq!(values[0][1], 1.5, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_normalize_basic() -> Result<()> {
        let device = Device::Cpu;

        // Create unnormalized embeddings
        let embeddings = Tensor::new(&[[3.0f32, 4.0], [1.0, 0.0]], &device)?;
        let normalized = normalize(&embeddings)?;

        // Check shape preserved
        assert_eq!(normalized.dims(), &[2, 2]);

        // Check unit length
        let values = normalized.to_vec2::<f32>()?;

        // First vector: [3, 4] -> [0.6, 0.8] (length 5)
        assert_relative_eq!(values[0][0], 0.6, epsilon = 1e-5);
        assert_relative_eq!(values[0][1], 0.8, epsilon = 1e-5);

        // Second vector: [1, 0] -> [1, 0] (already normalized)
        assert_relative_eq!(values[1][0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(values[1][1], 0.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_normalize_unit_length() -> Result<()> {
        let device = Device::Cpu;

        // Random embeddings
        let embeddings = Tensor::randn(0f32, 1.0, (5, 384), &device)?;
        let normalized = normalize(&embeddings)?;

        // Compute norms
        let norms = normalized.sqr()?.sum_keepdim(1)?.sqrt()?;
        let norms_vec = norms.to_vec2::<f32>()?;

        // All norms should be 1.0
        for batch in &norms_vec {
            for &norm in batch {
                assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
            }
        }

        Ok(())
    }

    #[test]
    fn test_normalize_zero_vector() -> Result<()> {
        let device = Device::Cpu;

        // Zero vector (edge case)
        let embeddings = Tensor::zeros((1, 4), DType::F32, &device)?;
        let normalized = normalize(&embeddings)?;

        // Should handle gracefully (result will be zero due to clamping)
        let values = normalized.to_vec2::<f32>()?;
        for batch in &values {
            for &val in batch {
                assert_relative_eq!(val, 0.0, epsilon = 1e-5);
            }
        }

        Ok(())
    }
}
