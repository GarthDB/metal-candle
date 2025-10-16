//! `LoRA` (Low-Rank Adaptation) implementation.
//!
//! `LoRA` reduces the number of trainable parameters by learning low-rank
//! decompositions of weight update matrices.
//!
//! # Theory
//!
//! For a pre-trained weight matrix W ∈ ℝ^(d×k), `LoRA` represents the update ΔW
//! as the product of two low-rank matrices:
//!
//! ```text
//! W' = W + ΔW = W + BA
//! where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with r << min(d,k)
//! ```
//!
//! During training:
//! - W is frozen (not updated)
//! - Only A and B are trainable
//! - Output: `h = Wx + s·BAx` where `s = α/r` is the scaling factor
//!
//! # References
//!
//! - Paper: "`LoRA`: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
//! - <https://arxiv.org/abs/2106.09685>

use crate::error::Result;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for `LoRA` layers.
///
/// # Examples
///
/// ```
/// use metal_candle::training::LoRAConfig;
///
/// let config = LoRAConfig {
///     rank: 8,
///     alpha: 16.0,
///     dropout: 0.1,
/// };
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition.
    ///
    /// Typical values: 4, 8, 16, 32
    /// Lower rank = fewer parameters but less capacity
    pub rank: usize,

    /// Scaling factor for `LoRA` updates.
    ///
    /// The actual scaling applied is `alpha / rank`.
    /// Typical value: 2 × rank (e.g., alpha=16 for rank=8)
    pub alpha: f32,

    /// Dropout probability for `LoRA` layers.
    ///
    /// Applied to the output of the A matrix before multiplication with B.
    /// Set to 0.0 to disable dropout.
    pub dropout: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
        }
    }
}

impl LoRAConfig {
    /// Returns the scaling factor (`alpha / rank`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scaling(&self) -> f32 {
        // Safe: rank is typically small (4-32), well within f32 precision
        self.alpha / self.rank as f32
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `rank` is 0
    /// - `alpha` is not positive
    /// - `dropout` is not in [0, 1)
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "LoRA rank must be greater than 0".to_string(),
            }
            .into());
        }

        if self.alpha <= 0.0 {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "LoRA alpha must be positive".to_string(),
            }
            .into());
        }

        if !(0.0..1.0).contains(&self.dropout) {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "LoRA dropout must be in [0, 1)".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// A `LoRA` layer implementing low-rank adaptation.
///
/// This layer can be applied to any linear transformation to enable
/// efficient fine-tuning with a small number of trainable parameters.
///
/// # Architecture
///
/// ```text
/// Input (x)
///    |
///    ├─────> Frozen Linear (Wx) ─────┐
///    |                                 |
///    └─────> A^T ──> B^T ──> scale ──> Add ──> Output
///           (r×k)   (d×r)
/// ```
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::{LoRAConfig, LoRALayer};
/// use candle_core::{Device, Tensor, DType};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let config = LoRAConfig::default();
///
/// // Create LoRA layer for 768-dimensional space
/// let lora = LoRALayer::new(768, 768, &config, &device)?;
///
/// // Forward pass
/// let input = Tensor::zeros((2, 16, 768), DType::F32, &device)?;
/// let output = lora.forward(&input)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct LoRALayer {
    /// Low-rank matrix A: (rank, `in_features`)
    ///
    /// Initialized with Gaussian distribution (mean=0, std=1/√rank)
    lora_a: Tensor,

    /// Low-rank matrix B: (`out_features`, rank)
    ///
    /// Initialized with zeros
    lora_b: Tensor,

    /// `LoRA` configuration
    config: LoRAConfig,

    /// Input dimension
    in_features: usize,

    /// Output dimension
    out_features: usize,
}

impl LoRALayer {
    /// Creates a new `LoRA` layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `config` - `LoRA` configuration
    /// * `device` - Device to place tensors on
    ///
    /// # Initialization
    ///
    /// Following the `LoRA` paper:
    /// - Matrix A: Gaussian distribution N(0, σ²) where σ = 1/√rank
    /// - Matrix B: Zeros
    ///
    /// This ensures the `LoRA` layer initially acts as identity (adds zero).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration is invalid
    /// - Tensor creation fails
    /// - Rank is larger than `min(in_features`, `out_features`)
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: &LoRAConfig,
        device: &Device,
    ) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Validate rank
        let max_rank = in_features.min(out_features);
        if config.rank > max_rank {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: format!(
                    "LoRA rank {} exceeds maximum rank {} (min of in_features={}, out_features={})",
                    config.rank, max_rank, in_features, out_features
                ),
            }
            .into());
        }

        // Initialize A with Gaussian distribution
        // Standard deviation: 1/√rank for stable training
        #[allow(clippy::cast_precision_loss)]
        let std = 1.0 / (config.rank as f32).sqrt(); // Safe: rank is small (4-32)
        let lora_a = Tensor::randn(0f32, std, (config.rank, in_features), device)?;

        // Initialize B with zeros
        let lora_b = Tensor::zeros((out_features, config.rank), DType::F32, device)?;

        Ok(Self {
            lora_a,
            lora_b,
            config: *config,
            in_features,
            out_features,
        })
    }

    /// Performs forward pass through the `LoRA` layer.
    ///
    /// Computes: `scale * (input @ A^T @ B^T)`
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `(..., in_features)`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `(..., out_features)`
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input: (..., in_features)
        // lora_a: (rank, in_features)
        // lora_b: (out_features, rank)

        // For matmul to work correctly with batched inputs, we need:
        // input @ A^T where A^T is (in_features, rank)
        // This gives us (..., rank)

        // input @ A^T -> (..., rank)
        // Candle's broadcast_matmul handles batched dimensions automatically
        // (..., in_features) @ (in_features, rank) -> (..., rank)
        let a_t = self.lora_a.t()?; // (rank, in_features) -> (in_features, rank)
        let hidden = input.broadcast_matmul(&a_t)?;

        // TODO: Apply dropout if configured (for training mode)
        // For now, skip dropout - will add when implementing training loop

        // Step 2: hidden @ B^T -> (..., out_features)
        // hidden: (..., rank)
        // B: (out_features, rank)
        // B^T: (rank, out_features)
        let b_t = self.lora_b.t()?; // (out_features, rank) -> (rank, out_features)
        let output = hidden.broadcast_matmul(&b_t)?;

        // Step 3: Scale by alpha/rank
        let scaling = self.config.scaling();
        let scaled_output = (output * f64::from(scaling))?;

        Ok(scaled_output)
    }

    /// Returns the number of trainable parameters in this `LoRA` layer.
    ///
    /// Parameters: `rank * (in_features + out_features)`
    #[must_use]
    pub const fn num_parameters(&self) -> usize {
        self.config.rank * (self.in_features + self.out_features)
    }

    /// Returns a reference to the A matrix.
    #[must_use]
    pub const fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Returns a reference to the B matrix.
    #[must_use]
    pub const fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// Returns the `LoRA` configuration.
    #[must_use]
    pub const fn config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Returns the input dimension.
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output dimension.
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lora_config_default() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.dropout, 0.0);
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
        };
        assert_eq!(config.scaling(), 2.0);

        let config2 = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
        };
        assert_eq!(config2.scaling(), 2.0);
    }

    #[test]
    fn test_lora_config_validation() {
        let valid = LoRAConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_rank = LoRAConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(invalid_rank.validate().is_err());

        let invalid_alpha = LoRAConfig {
            alpha: -1.0,
            ..Default::default()
        };
        assert!(invalid_alpha.validate().is_err());

        let invalid_dropout = LoRAConfig {
            dropout: 1.5,
            ..Default::default()
        };
        assert!(invalid_dropout.validate().is_err());
    }

    #[test]
    fn test_lora_layer_creation() {
        let device = Device::Cpu;
        let config = LoRAConfig::default();

        let lora = LoRALayer::new(768, 768, &config, &device);
        assert!(lora.is_ok());

        let lora = lora.unwrap();
        assert_eq!(lora.in_features(), 768);
        assert_eq!(lora.out_features(), 768);
        assert_eq!(lora.config().rank, 8);
    }

    #[test]
    fn test_lora_layer_initialization() {
        let device = Device::Cpu;
        let config = LoRAConfig::default();

        let lora = LoRALayer::new(128, 128, &config, &device).unwrap();

        // Check A matrix shape: (rank, in_features)
        assert_eq!(lora.lora_a().dims(), &[8, 128]);

        // Check B matrix shape: (out_features, rank)
        assert_eq!(lora.lora_b().dims(), &[128, 8]);

        // B should be initialized to zeros
        let b_sum = lora.lora_b().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(b_sum, 0.0);
    }

    #[test]
    fn test_lora_layer_invalid_rank() {
        let device = Device::Cpu;
        let config = LoRAConfig {
            rank: 1000, // Larger than dimensions
            ..Default::default()
        };

        let lora = LoRALayer::new(128, 128, &config, &device);
        assert!(lora.is_err());
    }

    #[test]
    fn test_lora_layer_forward() {
        let device = Device::Cpu;
        let config = LoRAConfig::default();

        let lora = LoRALayer::new(128, 128, &config, &device).unwrap();

        // Create input: (batch=2, seq=16, features=128)
        let input = Tensor::randn(0f32, 1f32, (2, 16, 128), &device).unwrap();

        let output = lora.forward(&input);
        if let Err(ref e) = output {
            eprintln!("Forward pass failed: {:?}", e);
        }
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dims(), &[2, 16, 128]);
    }

    #[test]
    fn test_lora_layer_num_parameters() {
        let device = Device::Cpu;
        let config = LoRAConfig {
            rank: 8,
            ..Default::default()
        };

        let lora = LoRALayer::new(768, 768, &config, &device).unwrap();

        // Parameters: rank * (in_features + out_features)
        // = 8 * (768 + 768) = 8 * 1536 = 12,288
        assert_eq!(lora.num_parameters(), 12_288);
    }

    #[test]
    fn test_lora_layer_different_dimensions() {
        let device = Device::Cpu;
        let config = LoRAConfig::default();

        // Test various dimension combinations
        for (in_dim, out_dim) in [(512, 2048), (2048, 512), (1024, 1024)].iter() {
            let lora = LoRALayer::new(*in_dim, *out_dim, &config, &device);
            assert!(lora.is_ok());

            let lora = lora.unwrap();
            assert_eq!(lora.in_features(), *in_dim);
            assert_eq!(lora.out_features(), *out_dim);
        }
    }
}
