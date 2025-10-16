//! KV-cache for efficient autoregressive generation.
//!
//! Caches key and value tensors from previous tokens to avoid
//! recomputing attention for the entire sequence at each step.

use crate::error::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Configuration for KV-cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Maximum sequence length to cache
    pub max_seq_len: usize,

    /// Number of attention layers
    pub num_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension per head
    pub head_dim: usize,

    /// Batch size
    pub batch_size: usize,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            num_layers: 24,
            num_heads: 14,
            head_dim: 64,
            batch_size: 1,
        }
    }
}

/// KV-cache for efficient autoregressive generation.
///
/// Stores key and value tensors from previous forward passes
/// to avoid recomputing attention over the entire sequence.
///
/// # Architecture
///
/// For each layer, stores:
/// - Keys: `(batch, num_heads, seq_len, head_dim)`
/// - Values: `(batch, num_heads, seq_len, head_dim)`
///
/// # Memory Usage
///
/// For a model with:
/// - 24 layers
/// - 14 heads  
/// - 64 dim per head
/// - 2048 max sequence length
/// - F16 dtype (2 bytes)
///
/// Memory: `24 * 2 * (1 * 14 * 2048 * 64) * 2 ≈ 173 MB`
///
/// # Examples
///
/// ```
/// use metal_candle::inference::{KVCache, KVCacheConfig};
/// use candle_core::Device;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let config = KVCacheConfig {
///     max_seq_len: 2048,
///     num_layers: 24,
///     num_heads: 14,
///     head_dim: 64,
///     batch_size: 1,
/// };
///
/// let mut cache = KVCache::new(config, &device)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct KVCache {
    config: KVCacheConfig,
    #[allow(dead_code)] // Will be used for tensor allocation in future
    device: Device,

    /// Cache for each layer: (key, value)
    /// Key/Value shape: (batch, `num_heads`, `seq_len`, `head_dim`)
    cache: HashMap<usize, (Tensor, Tensor)>,

    /// Current sequence position (number of cached tokens)
    position: usize,
}

impl KVCache {
    /// Creates a new empty KV-cache.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor allocation fails.
    pub fn new(config: KVCacheConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            config,
            device: device.clone(),
            cache: HashMap::new(),
            position: 0,
        })
    }

    /// Returns the current sequence position (number of cached tokens).
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Returns the maximum sequence length.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// Clears the cache and resets position to 0.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.position = 0;
    }

    /// Updates the cache for a specific layer with new key/value tensors.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the attention layer (0-indexed)
    /// * `key` - New key tensor to append, shape: `(batch, num_heads, new_tokens, head_dim)`
    /// * `value` - New value tensor to append, shape: `(batch, num_heads, new_tokens, head_dim)`
    ///
    /// # Returns
    ///
    /// Returns the concatenated (key, value) tensors including cached history.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tensor shapes are incompatible
    /// - Concatenation fails
    /// - Cache is full (`position >= max_seq_len`)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::inference::{KVCache, KVCacheConfig};
    /// # use candle_core::{Device, Tensor, DType};
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let device = Device::Cpu;
    /// # let config = KVCacheConfig::default();
    /// # let mut cache = KVCache::new(config, &device)?;
    /// // First token (prompt)
    /// let key = Tensor::zeros((1, 14, 1, 64), DType::F32, &device)?;
    /// let value = Tensor::zeros((1, 14, 1, 64), DType::F32, &device)?;
    /// let (full_key, full_value) = cache.update(0, &key, &value)?;
    /// // full_key shape: (1, 14, 1, 64)
    ///
    /// // Second token (generation)
    /// let key = Tensor::zeros((1, 14, 1, 64), DType::F32, &device)?;
    /// let value = Tensor::zeros((1, 14, 1, 64), DType::F32, &device)?;
    /// let (full_key, full_value) = cache.update(0, &key, &value)?;
    /// // full_key shape: (1, 14, 2, 64) - includes cached token
    /// # Ok(())
    /// # }
    /// ```
    pub fn update(
        &mut self,
        layer_idx: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get current sequence length from new tensors
        let new_seq_len = key.dims()[2];

        // Check if adding these tokens would exceed max length
        if self.position + new_seq_len > self.config.max_seq_len {
            return Err(crate::error::InferenceError::CacheFull {
                position: self.position,
                max_len: self.config.max_seq_len,
            }
            .into());
        }

        // Get or initialize cache for this layer
        let (full_key, full_value) =
            if let Some((cached_key, cached_value)) = self.cache.get(&layer_idx) {
                // Concatenate with existing cache along sequence dimension (dim 2)
                let full_key = Tensor::cat(&[cached_key, key], 2)?;
                let full_value = Tensor::cat(&[cached_value, value], 2)?;
                (full_key, full_value)
            } else {
                // First time seeing this layer, just use the new tensors
                (key.clone(), value.clone())
            };

        // Store updated cache
        self.cache
            .insert(layer_idx, (full_key.clone(), full_value.clone()));

        // Update position based on the new sequence length
        // Only update on first layer to avoid double-counting
        if layer_idx == 0 || self.cache.len() == 1 {
            self.position += new_seq_len;
        }

        Ok((full_key, full_value))
    }

    /// Gets the cached key/value tensors for a specific layer.
    ///
    /// Returns `None` if the layer hasn't been cached yet.
    #[must_use]
    pub fn get(&self, layer_idx: usize) -> Option<&(Tensor, Tensor)> {
        self.cache.get(&layer_idx)
    }

    /// Returns the number of layers currently in the cache.
    #[must_use]
    pub fn num_cached_layers(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_kv_cache_creation() {
        let device = Device::Cpu;
        let config = KVCacheConfig::default();
        let cache = KVCache::new(config, &device).unwrap();

        assert_eq!(cache.position(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.num_cached_layers(), 0);
    }

    #[test]
    fn test_kv_cache_single_update() {
        let device = Device::Cpu;
        let config = KVCacheConfig {
            max_seq_len: 10,
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            batch_size: 1,
        };
        let mut cache = KVCache::new(config, &device).unwrap();

        // Add first token
        let key = Tensor::zeros((1, 4, 1, 8), DType::F32, &device).unwrap();
        let value = Tensor::zeros((1, 4, 1, 8), DType::F32, &device).unwrap();

        let (full_key, full_value) = cache.update(0, &key, &value).unwrap();

        assert_eq!(cache.position(), 1);
        assert_eq!(full_key.dims(), &[1, 4, 1, 8]);
        assert_eq!(full_value.dims(), &[1, 4, 1, 8]);
        assert_eq!(cache.num_cached_layers(), 1);
    }

    #[test]
    fn test_kv_cache_multiple_updates() {
        let device = Device::Cpu;
        let config = KVCacheConfig {
            max_seq_len: 10,
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            batch_size: 1,
        };
        let mut cache = KVCache::new(config, &device).unwrap();

        // Add first token
        let key1 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        let value1 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        cache.update(0, &key1, &value1).unwrap();

        // Add second token
        let key2 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        let value2 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        let (full_key, full_value) = cache.update(0, &key2, &value2).unwrap();

        assert_eq!(cache.position(), 2);
        assert_eq!(full_key.dims(), &[1, 4, 2, 8]);
        assert_eq!(full_value.dims(), &[1, 4, 2, 8]);
    }

    #[test]
    fn test_kv_cache_clear() {
        let device = Device::Cpu;
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config, &device).unwrap();

        let key = Tensor::zeros((1, 4, 1, 8), DType::F32, &device).unwrap();
        let value = Tensor::zeros((1, 4, 1, 8), DType::F32, &device).unwrap();
        cache.update(0, &key, &value).unwrap();

        assert_eq!(cache.position(), 1);

        cache.clear();

        assert_eq!(cache.position(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.num_cached_layers(), 0);
    }

    #[test]
    fn test_kv_cache_full() {
        let device = Device::Cpu;
        let config = KVCacheConfig {
            max_seq_len: 2,
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            batch_size: 1,
        };
        let mut cache = KVCache::new(config, &device).unwrap();

        // Fill cache
        let key = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
        let value = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
        cache.update(0, &key, &value).unwrap();
        cache.update(0, &key, &value).unwrap();

        assert_eq!(cache.position(), 2);

        // Try to add one more (should fail)
        let result = cache.update(0, &key, &value);
        assert!(result.is_err());
    }
}
