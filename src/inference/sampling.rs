//! Sampling strategies for text generation.

use crate::error::Result;
use candle_core::Tensor;
use rand::Rng;

/// Sampling strategy for token selection.
#[derive(Debug, Clone, Default)]
pub enum SamplingStrategy {
    /// Greedy sampling (argmax)
    #[default]
    Greedy,

    /// Top-k sampling
    TopK {
        /// Number of top tokens to consider
        k: usize,
    },

    /// Top-p (nucleus) sampling
    TopP {
        /// Cumulative probability threshold
        p: f64,
    },

    /// Temperature sampling
    Temperature {
        /// Temperature value (higher = more random)
        temperature: f64,
    },
}

/// Applies repetition penalty to logits.
///
/// Penalizes previously generated tokens by dividing their logits by the penalty factor.
/// This reduces the likelihood of repetitive text generation.
///
/// # Arguments
///
/// * `logits` - Mutable logits tensor to modify, shape: `(vocab_size,)`
/// * `generated_ids` - Previously generated token IDs to penalize
/// * `penalty` - Penalty factor (> 1.0 = penalize, 1.0 = no penalty)
///
/// # Errors
///
/// Returns an error if tensor operations fail.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::inference::sampling::apply_repetition_penalty;
/// use candle_core::{Device, Tensor};
///
/// let device = Device::Cpu;
/// let mut logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
/// let generated = vec![1, 3]; // Penalize tokens 1 and 3
///
/// apply_repetition_penalty(&mut logits, &generated, 1.2)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn apply_repetition_penalty(
    logits: &mut Tensor,
    generated_ids: &[u32],
    penalty: f32,
) -> Result<()> {
    if generated_ids.is_empty() || (penalty - 1.0).abs() < 1e-7 {
        return Ok(()); // No penalty needed
    }

    let mut logits_vec = logits.to_vec1::<f32>()?;

    // Apply penalty to previously generated tokens
    for &token_id in generated_ids {
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            logits_vec[idx] /= penalty;
        }
    }

    // Replace logits with penalized version
    *logits = Tensor::new(&logits_vec[..], logits.device())?;

    Ok(())
}

/// Samples a token from logits using the specified strategy.
///
/// # Arguments
///
/// * `logits` - Logits tensor, shape: `(vocab_size,)`
/// * `strategy` - Sampling strategy to use
/// * `generated_ids` - Previously generated token IDs (for repetition penalty)
/// * `repetition_penalty` - Penalty factor for repeated tokens (1.0 = no penalty)
///
/// # Returns
///
/// Returns the sampled token ID.
///
/// # Errors
///
/// Returns an error if sampling fails or tensor operations fail.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::inference::sampling::{sample_token, SamplingStrategy};
/// use candle_core::{Device, Tensor};
///
/// let device = Device::Cpu;
/// let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
/// let strategy = SamplingStrategy::Greedy;
/// let generated = vec![1, 2];
///
/// let token = sample_token(&logits, &strategy, &generated, 1.2)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn sample_token(
    logits: &Tensor,
    strategy: &SamplingStrategy,
    generated_ids: &[u32],
    repetition_penalty: f32,
) -> Result<u32> {
    // Apply repetition penalty if needed
    let mut logits = logits.clone();
    apply_repetition_penalty(&mut logits, generated_ids, repetition_penalty)?;

    match strategy {
        SamplingStrategy::Greedy => sample_greedy(&logits),
        SamplingStrategy::TopK { k } => sample_top_k(&logits, *k),
        SamplingStrategy::TopP { p } => sample_top_p(&logits, *p),
        SamplingStrategy::Temperature { temperature } => sample_temperature(&logits, *temperature),
    }
}

/// Greedy sampling (argmax).
fn sample_greedy(logits: &Tensor) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;
    let token = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| u32::try_from(idx).unwrap_or(u32::MAX))
        .ok_or_else(|| crate::error::InferenceError::SamplingError {
            reason: "Empty logits".to_string(),
        })?;
    Ok(token)
}

/// Top-k sampling.
fn sample_top_k(logits: &Tensor, k: usize) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;

    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    indexed.truncate(k);

    // Apply softmax to top-k
    let max_logit = indexed[0].1;
    let exp_sum: f32 = indexed.iter().map(|(_, l)| (l - max_logit).exp()).sum();
    let probs: Vec<f64> = indexed
        .iter()
        .map(|(_, l)| f64::from((l - max_logit).exp() / exp_sum))
        .collect();

    // Sample from top-k
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return Ok(u32::try_from(indexed[i].0).unwrap_or(u32::MAX));
        }
    }

    Ok(u32::try_from(indexed[0].0).unwrap_or(u32::MAX))
}

/// Top-p (nucleus) sampling.
fn sample_top_p(logits: &Tensor, p: f64) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;

    // Sort by probability (descending)
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    // Apply softmax
    let max_logit = indexed[0].1;
    let exp_sum: f32 = indexed.iter().map(|(_, l)| (l - max_logit).exp()).sum();
    let probs: Vec<(usize, f64)> = indexed
        .iter()
        .map(|(idx, l)| (*idx, f64::from((l - max_logit).exp() / exp_sum)))
        .collect();

    // Find nucleus (top-p)
    let mut cumsum = 0.0;
    let mut nucleus = Vec::new();
    for (idx, prob) in probs {
        nucleus.push((idx, prob));
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }

    // Sample from nucleus
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let nucleus_sum: f64 = nucleus.iter().map(|(_, p)| p).sum();
    let mut cumsum = 0.0;
    for (idx, prob) in &nucleus {
        cumsum += prob / nucleus_sum;
        if r <= cumsum {
            return Ok(u32::try_from(*idx).unwrap_or(u32::MAX));
        }
    }

    Ok(u32::try_from(nucleus[0].0).unwrap_or(u32::MAX))
}

/// Temperature sampling.
fn sample_temperature(logits: &Tensor, temperature: f64) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;

    // Apply temperature
    #[allow(clippy::cast_possible_truncation)]
    // temperature is user-controlled, truncation acceptable
    let scaled: Vec<f32> = logits_vec.iter().map(|l| l / temperature as f32).collect();

    // Apply softmax
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scaled.iter().map(|l| (l - max_logit).exp()).sum();
    let probs: Vec<f64> = scaled
        .iter()
        .map(|l| f64::from((l - max_logit).exp() / exp_sum))
        .collect();

    // Sample
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return Ok(u32::try_from(idx).unwrap_or(u32::MAX));
        }
    }

    Ok(u32::try_from(probs.len() - 1).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_greedy_sampling() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 3.0, 2.0, 0.5], &device).unwrap();

        let token = sample_greedy(&logits).unwrap();
        assert_eq!(token, 1); // Index of max value (3.0)
    }

    #[test]
    fn test_top_k_sampling() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 3.0, 2.0, 0.5], &device).unwrap();

        // Top-2: should sample from indices 1 (3.0) or 2 (2.0)
        let token = sample_top_k(&logits, 2).unwrap();
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_sampling_strategy_default() {
        let strategy = SamplingStrategy::default();
        assert!(matches!(strategy, SamplingStrategy::Greedy));
    }

    #[test]
    fn test_apply_repetition_penalty() {
        let device = Device::Cpu;
        let mut logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
        let generated = vec![1, 3]; // Penalize tokens 1 and 3

        apply_repetition_penalty(&mut logits, &generated, 2.0).unwrap();

        let result = logits.to_vec1::<f32>().unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6); // Unchanged
        assert!((result[1] - 1.0).abs() < 1e-6); // 2.0 / 2.0 = 1.0
        assert!((result[2] - 3.0).abs() < 1e-6); // Unchanged
        assert!((result[3] - 2.0).abs() < 1e-6); // 4.0 / 2.0 = 2.0
    }

    #[test]
    fn test_apply_repetition_penalty_empty() {
        let device = Device::Cpu;
        let mut logits = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let original = logits.to_vec1::<f32>().unwrap();

        apply_repetition_penalty(&mut logits, &[], 2.0).unwrap();

        let result = logits.to_vec1::<f32>().unwrap();
        assert_eq!(result, original); // Unchanged with empty generated_ids
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let device = Device::Cpu;
        let mut logits = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let original = logits.to_vec1::<f32>().unwrap();

        apply_repetition_penalty(&mut logits, &[0, 1], 1.0).unwrap();

        let result = logits.to_vec1::<f32>().unwrap();
        assert_eq!(result, original); // Unchanged with penalty = 1.0
    }

    #[test]
    fn test_sample_token_with_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 5.0, 2.0, 0.5], &device).unwrap();

        // Without penalty, greedy should pick token 1 (highest logit 5.0)
        let token = sample_token(&logits, &SamplingStrategy::Greedy, &[], 1.0).unwrap();
        assert_eq!(token, 1);

        // With high penalty on token 1, should pick token 2 (next highest)
        let token = sample_token(&logits, &SamplingStrategy::Greedy, &[1], 10.0).unwrap();
        assert_eq!(token, 2);
    }
}
