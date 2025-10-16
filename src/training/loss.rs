//! Loss functions for training.
//!
//! Implements various loss functions used in language model training,
//! with focus on numerical stability and efficient computation.

use crate::error::Result;
use candle_core::Tensor;

/// Computes cross-entropy loss for language modeling.
///
/// This is the standard loss function for autoregressive language models.
/// The loss is computed as the negative log-likelihood of the correct tokens.
///
/// # Formula
///
/// ```text
/// loss = -1/N * Σ log(P(target_i | logits_i))
/// ```
///
/// where `P(target_i | logits_i) = softmax(logits_i)[target_i]`
///
/// # Arguments
///
/// * `logits` - Model output logits of shape `(batch, seq_len, vocab_size)`
/// * `targets` - Target token IDs of shape `(batch, seq_len)`
/// * `ignore_index` - Optional token ID to ignore (e.g., padding token)
///
/// # Returns
///
/// Scalar tensor containing the mean loss over all non-ignored tokens.
///
/// # Errors
///
/// Returns an error if tensor operations fail.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::cross_entropy_loss;
/// use candle_core::{Tensor, Device, DType};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
///
/// // Model outputs: (batch=2, seq_len=10, vocab_size=50257)
/// let logits = Tensor::randn(0f32, 1f32, (2, 10, 50257), &device)?;
///
/// // Targets: (batch=2, seq_len=10)
/// let targets = Tensor::zeros((2, 10), DType::U32, &device)?;
///
/// // Compute loss, ignoring padding token (index 0)
/// let loss = cross_entropy_loss(&logits, &targets, Some(0))?;
///
/// println!("Loss: {}", loss.to_scalar::<f32>()?);
/// # Ok(())
/// # }
/// ```
pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: Option<u32>,
) -> Result<Tensor> {
    // logits: (batch, seq_len, vocab_size)
    // targets: (batch, seq_len)

    let logits_shape = logits.dims();
    let targets_shape = targets.dims();

    // Validate shapes
    if logits_shape.len() != 3 {
        return Err(crate::error::TrainingError::InvalidConfig {
            reason: format!(
                "logits must be 3D (batch, seq_len, vocab_size), got shape {logits_shape:?}"
            ),
        }
        .into());
    }

    if targets_shape.len() != 2 {
        return Err(crate::error::TrainingError::InvalidConfig {
            reason: format!("targets must be 2D (batch, seq_len), got shape {targets_shape:?}"),
        }
        .into());
    }

    if logits_shape[0] != targets_shape[0] || logits_shape[1] != targets_shape[1] {
        return Err(crate::error::TrainingError::InvalidConfig {
            reason: format!(
                "logits and targets batch/seq_len mismatch: logits {:?} vs targets {:?}",
                &logits_shape[..2],
                targets_shape
            ),
        }
        .into());
    }

    let batch_size = logits_shape[0];
    let seq_len = logits_shape[1];
    let vocab_size = logits_shape[2];

    // Flatten logits: (batch * seq_len, vocab_size)
    let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;

    // Flatten targets: (batch * seq_len,)
    let targets_flat = targets.reshape((batch_size * seq_len,))?;

    // Compute log softmax for numerical stability
    // log_softmax(x) = x - log(sum(exp(x)))
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;

    // Gather log probabilities for target tokens
    // For each position, get log_probs[i, targets[i]]
    let target_log_probs = gather_last_dim(&log_probs, &targets_flat)?;

    // Apply ignore_index mask if specified
    let masked_log_probs = if let Some(ignore_idx) = ignore_index {
        // Create mask: 1.0 for valid tokens, 0.0 for ignored tokens
        let mask = targets_flat.ne(f64::from(ignore_idx))?;
        let mask_f32 = mask.to_dtype(candle_core::DType::F32)?;

        // Multiply log_probs by mask (zeros out ignored positions)
        let masked = target_log_probs.mul(&mask_f32)?;

        // Compute mean over non-ignored tokens
        let sum = masked.sum_all()?;
        let count = mask_f32.sum_all()?;

        // Avoid division by zero
        let count_val = count.to_scalar::<f32>()?;
        if count_val == 0.0 {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "all tokens are ignored, cannot compute loss".to_string(),
            }
            .into());
        }

        sum.div(&count)?
    } else {
        // No masking, just compute mean
        target_log_probs.mean_all()?
    };

    // Return negative log likelihood
    let loss = masked_log_probs.neg()?;

    Ok(loss)
}

/// Gathers values from the last dimension using indices.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape `(N, D)`
/// * `indices` - Indices tensor of shape `(N,)` with values in `[0, D)`
///
/// # Returns
///
/// Tensor of shape `(N,)` where `output[i] = tensor[i, indices[i]]`
fn gather_last_dim(tensor: &Tensor, indices: &Tensor) -> Result<Tensor> {
    // tensor: (N, D)
    // indices: (N,)
    // output: (N,)

    let shape = tensor.dims();
    let n = shape[0];

    // Convert indices to the data from the tensor
    let indices_u32 = indices.to_dtype(candle_core::DType::U32)?;
    let indices_vec = indices_u32.to_vec1::<u32>()?;

    // For each row, we need to gather tensor[i, indices[i]]
    let mut result_vec = Vec::with_capacity(n);

    for (i, &idx) in indices_vec.iter().enumerate() {
        // Get the row as a 1D tensor
        let row = tensor.get(i)?;

        // Get the specific index from that row
        let value = row.get(idx as usize)?;

        result_vec.push(value);
    }

    // Stack all values into a single tensor
    Tensor::stack(&result_vec, 0).map_err(Into::into)
}

/// Computes cross-entropy loss with label smoothing.
///
/// Label smoothing is a regularization technique that prevents the model
/// from becoming too confident by distributing some probability mass
/// from the correct label to all other labels.
///
/// # Arguments
///
/// * `logits` - Model output logits of shape `(batch, seq_len, vocab_size)`
/// * `targets` - Target token IDs of shape `(batch, seq_len)`
/// * `smoothing` - Smoothing factor in `[0, 1)`. Use 0.0 for no smoothing, typically 0.1
/// * `ignore_index` - Optional token ID to ignore
///
/// # Returns
///
/// Scalar tensor containing the smoothed cross-entropy loss.
///
/// # Errors
///
/// Returns an error if tensor operations fail or smoothing is invalid.
pub fn cross_entropy_loss_with_smoothing(
    logits: &Tensor,
    targets: &Tensor,
    smoothing: f32,
    ignore_index: Option<u32>,
) -> Result<Tensor> {
    if !(0.0..1.0).contains(&smoothing) {
        return Err(crate::error::TrainingError::InvalidConfig {
            reason: format!("smoothing must be in [0, 1), got {smoothing}"),
        }
        .into());
    }

    if smoothing == 0.0 {
        // No smoothing, use standard cross-entropy
        return cross_entropy_loss(logits, targets, ignore_index);
    }

    // With label smoothing:
    // loss = (1 - smoothing) * NLL(target) + smoothing * mean(log_probs)
    //
    // This distributes smoothing probability uniformly across all classes

    // Compute standard loss
    let nll_loss = cross_entropy_loss(logits, targets, ignore_index)?;

    // Compute mean log prob over all classes
    let log_probs = candle_nn::ops::log_softmax(logits, 2)?;
    let smooth_loss = log_probs.mean_all()?.neg()?;

    // Combine: (1 - smoothing) * nll + smoothing * smooth
    let nll_weight = 1.0 - smoothing;
    let weighted_nll = (nll_loss * f64::from(nll_weight))?;
    let weighted_smooth = (smooth_loss * f64::from(smoothing))?;

    weighted_nll.add(&weighted_smooth).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cross_entropy_loss_basic() {
        let device = Device::Cpu;

        // Create simple logits: (batch=1, seq=1, vocab=3)
        // Favor class 0
        let logits = Tensor::new(&[[[10.0f32, 0.0, 0.0]]], &device).unwrap();

        // Target is class 0
        let targets = Tensor::new(&[[0u32]], &device).unwrap();

        let loss = cross_entropy_loss(&logits, &targets, None).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // Loss should be very small since logits strongly predict class 0
        assert!(loss_val < 0.1, "loss = {loss_val}");
    }

    #[test]
    fn test_cross_entropy_loss_wrong_prediction() {
        let device = Device::Cpu;

        // Create logits favoring class 0
        let logits = Tensor::new(&[[[10.0f32, 0.0, 0.0]]], &device).unwrap();

        // But target is class 2
        let targets = Tensor::new(&[[2u32]], &device).unwrap();

        let loss = cross_entropy_loss(&logits, &targets, None).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // Loss should be large since prediction is wrong
        assert!(loss_val > 5.0, "loss = {loss_val}");
    }

    #[test]
    fn test_cross_entropy_loss_with_ignore() {
        let device = Device::Cpu;

        // Batch of 2, seq_len of 2, vocab of 3
        let logits = Tensor::new(
            &[
                [[1.0f32, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            ],
            &device,
        )
        .unwrap();

        // Targets with padding token (0 should be ignored)
        let targets = Tensor::new(&[[2u32, 0], [2, 0]], &device).unwrap();

        // Loss with ignore_index=0 (ignores half the tokens)
        let loss = cross_entropy_loss(&logits, &targets, Some(0)).unwrap();
        assert!(loss.to_scalar::<f32>().is_ok());
    }

    #[test]
    fn test_cross_entropy_loss_invalid_shapes() {
        let device = Device::Cpu;

        // 2D logits (invalid)
        let logits = Tensor::zeros((2, 10), DType::F32, &device).unwrap();
        let targets = Tensor::zeros((2, 10), DType::U32, &device).unwrap();

        let result = cross_entropy_loss(&logits, &targets, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_loss_shape_mismatch() {
        let device = Device::Cpu;

        let logits = Tensor::zeros((2, 10, 100), DType::F32, &device).unwrap();
        let targets = Tensor::zeros((2, 5), DType::U32, &device).unwrap(); // Wrong seq_len

        let result = cross_entropy_loss(&logits, &targets, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_smoothing_zero() {
        let device = Device::Cpu;

        let logits = Tensor::new(&[[[1.0f32, 2.0, 3.0]]], &device).unwrap();
        let targets = Tensor::new(&[[2u32]], &device).unwrap();

        // Smoothing = 0.0 should be identical to standard cross-entropy
        let loss_standard = cross_entropy_loss(&logits, &targets, None).unwrap();
        let loss_smooth = cross_entropy_loss_with_smoothing(&logits, &targets, 0.0, None).unwrap();

        let val1 = loss_standard.to_scalar::<f32>().unwrap();
        let val2 = loss_smooth.to_scalar::<f32>().unwrap();

        assert!((val1 - val2).abs() < 1e-6, "{val1} vs {val2}");
    }

    #[test]
    fn test_label_smoothing_effect() {
        let device = Device::Cpu;

        let logits = Tensor::new(&[[[10.0f32, 0.0, 0.0]]], &device).unwrap();
        let targets = Tensor::new(&[[0u32]], &device).unwrap();

        let loss_no_smooth = cross_entropy_loss(&logits, &targets, None).unwrap();
        let loss_smooth = cross_entropy_loss_with_smoothing(&logits, &targets, 0.1, None).unwrap();

        let val1 = loss_no_smooth.to_scalar::<f32>().unwrap();
        let val2 = loss_smooth.to_scalar::<f32>().unwrap();

        // Smoothed loss should be slightly higher (regularization effect)
        assert!(val2 > val1, "{val2} should be > {val1}");
    }

    #[test]
    fn test_label_smoothing_invalid() {
        let device = Device::Cpu;

        let logits = Tensor::new(&[[[1.0f32, 2.0, 3.0]]], &device).unwrap();
        let targets = Tensor::new(&[[2u32]], &device).unwrap();

        // Smoothing >= 1.0 should be invalid
        let result = cross_entropy_loss_with_smoothing(&logits, &targets, 1.0, None);
        assert!(result.is_err());

        // Negative smoothing should be invalid
        let result = cross_entropy_loss_with_smoothing(&logits, &targets, -0.1, None);
        assert!(result.is_err());
    }
}
