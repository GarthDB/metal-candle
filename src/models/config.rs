//! Model configuration parsing and validation.
//!
//! This module handles loading and validating model configurations from JSON files,
//! typically named `config.json` in model directories.

use crate::error::{ModelError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for transformer models.
///
/// This struct represents the architecture configuration for a transformer model,
/// including dimensions, layer counts, and attention parameters.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::models::ModelConfig;
///
/// let config = ModelConfig::from_file("config.json")?;
/// println!("Model has {} layers", config.num_hidden_layers);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Model architecture type (e.g., "qwen2", "llama")
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden layer dimension
    pub hidden_size: usize,

    /// Intermediate (MLP) dimension
    pub intermediate_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for grouped-query attention)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Rope theta (for rotary embeddings)
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Data type for weights
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

fn default_max_position_embeddings() -> usize {
    2048
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10_000.0
}

impl ModelConfig {
    /// Loads a configuration from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the config.json file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::models::ModelConfig;
    ///
    /// let config = ModelConfig::from_file("model/config.json")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] if:
    /// - The file cannot be read
    /// - The JSON is malformed
    /// - Required fields are missing
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| ModelError::InvalidConfig {
            reason: format!("Failed to read config file {}: {e}", path.display()),
        })?;

        Self::from_json(&content)
    }

    /// Parses a configuration from a JSON string.
    ///
    /// # Arguments
    ///
    /// * `json` - JSON string containing the configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::models::ModelConfig;
    ///
    /// let json = r#"{
    ///     "vocab_size": 32000,
    ///     "hidden_size": 768,
    ///     "intermediate_size": 3072,
    ///     "num_hidden_layers": 12,
    ///     "num_attention_heads": 12
    /// }"#;
    ///
    /// let config = ModelConfig::from_json(json)?;
    /// assert_eq!(config.hidden_size, 768);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] if the JSON is malformed or missing required fields.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            ModelError::InvalidConfig {
                reason: format!("Failed to parse config JSON: {e}"),
            }
            .into()
        })
    }

    /// Validates the configuration for consistency.
    ///
    /// Checks that dimensions are compatible and values are reasonable.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] if validation fails.
    pub fn validate(&self) -> Result<()> {
        // Check that dimensions are non-zero
        if self.vocab_size == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "vocab_size must be > 0".to_string(),
            }
            .into());
        }

        if self.hidden_size == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "hidden_size must be > 0".to_string(),
            }
            .into());
        }

        if self.num_hidden_layers == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "num_hidden_layers must be > 0".to_string(),
            }
            .into());
        }

        if self.num_attention_heads == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "num_attention_heads must be > 0".to_string(),
            }
            .into());
        }

        // Check that hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "hidden_size ({}) must be divisible by num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            }
            .into());
        }

        // Validate num_key_value_heads if present
        if let Some(num_kv_heads) = self.num_key_value_heads {
            if num_kv_heads == 0 {
                return Err(ModelError::InvalidConfig {
                    reason: "num_key_value_heads must be > 0 if specified".to_string(),
                }
                .into());
            }

            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(ModelError::InvalidConfig {
                    reason: format!(
                        "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                        self.num_attention_heads, num_kv_heads
                    ),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Returns the head dimension (`hidden_size` / `num_attention_heads`).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Returns the number of key-value heads, defaulting to `num_attention_heads` if not specified.
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["qwen2".to_string()],
            vocab_size: 32000,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_key_value_heads: Some(12),
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            torch_dtype: Some("float16".to_string()),
        }
    }

    #[test]
    fn test_config_from_json() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12
        }"#;

        let config = ModelConfig::from_json(json).unwrap();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
    }

    #[test]
    fn test_config_validation_success() {
        let config = create_valid_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_vocab() {
        let mut config = create_valid_config();
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_incompatible_dimensions() {
        let mut config = create_valid_config();
        config.hidden_size = 769; // Not divisible by 12
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_kv_heads() {
        let mut config = create_valid_config();
        config.num_key_value_heads = Some(5); // 12 not divisible by 5
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim() {
        let config = create_valid_config();
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_num_kv_heads_default() {
        let mut config = create_valid_config();
        config.num_key_value_heads = None;
        assert_eq!(config.num_kv_heads(), config.num_attention_heads);
    }

    #[test]
    fn test_num_kv_heads_specified() {
        let mut config = create_valid_config();
        config.num_key_value_heads = Some(4);
        assert_eq!(config.num_kv_heads(), 4);
    }

    #[test]
    fn test_config_defaults() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12
        }"#;

        let config = ModelConfig::from_json(json).unwrap();
        assert_eq!(config.max_position_embeddings, 2048);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-7);
        assert!((config.rope_theta - 10_000.0).abs() < 1e-7);
    }
}
