//! Sampling strategies for text generation.

use crate::error::Result;
use candle_core::Tensor;
use rand::Rng;

/// Sampling strategy for token selection.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Greedy sampling (argmax)
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

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Greedy
    }
}

/// Samples a token from logits using the specified strategy.
///
/// # Arguments
///
/// * `logits` - Logits tensor, shape: `(vocab_size,)`
/// * `strategy` - Sampling strategy to use
///
/// # Returns
///
/// Returns the sampled token ID.
///
/// # Errors
///
/// Returns an error if sampling fails or tensor operations fail.
pub fn sample_token(logits: &Tensor, strategy: &SamplingStrategy) -> Result<u32> {
    match strategy {
        SamplingStrategy::Greedy => sample_greedy(logits),
        SamplingStrategy::TopK { k } => sample_top_k(logits, *k),
        SamplingStrategy::TopP { p } => sample_top_p(logits, *p),
        SamplingStrategy::Temperature { temperature } => sample_temperature(logits, *temperature),
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
}
