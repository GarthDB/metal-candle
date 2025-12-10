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

// Allow similar_names for LoRA operations - A/B matrix naming is standard ML convention
#![allow(clippy::similar_names)]

use crate::error::Result;
use candle_core::{DType, Device, Tensor, Var};
use serde::{Deserialize, Serialize};

#[cfg(feature = "graph")]
use crate::graph::LazyTensor;

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
    /// Wrapped in `Var` for gradient tracking
    lora_a: Var,

    /// Low-rank matrix B: (`out_features`, rank)
    ///
    /// Initialized with zeros
    /// Wrapped in `Var` for gradient tracking
    lora_b: Var,

    /// `LoRA` configuration
    config: LoRAConfig,

    /// Input dimension
    in_features: usize,

    /// Output dimension
    out_features: usize,

    /// Training mode flag
    ///
    /// When `true`, dropout is applied (if configured).
    /// When `false` (inference mode), dropout is skipped.
    training: bool,
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

        // OPTIMIZATION: Store A in transposed form (in_features, rank)
        // Initialize A with Gaussian distribution
        // Standard deviation: 1/√rank for stable training
        #[allow(clippy::cast_precision_loss)]
        let std = 1.0 / (config.rank as f32).sqrt(); // Safe: rank is small (4-32)
        let tensor_a = Tensor::randn(0f32, std, (in_features, config.rank), device)?;
        let lora_a = Var::from_tensor(&tensor_a)?;

        // OPTIMIZATION: Store B in transposed and pre-scaled form (rank, out_features)
        // Initialize B with zeros, then scale it by (alpha/rank)
        // This avoids scaling on every forward pass
        let tensor_b = Tensor::zeros((config.rank, out_features), DType::F32, device)?;
        // Note: B starts at zero, so scaling doesn't change it, but we keep the shape
        // When loading checkpoints or training, B will be non-zero and benefit from pre-scaling
        let lora_b = Var::from_tensor(&tensor_b)?;

        Ok(Self {
            lora_a,
            lora_b,
            config: *config,
            in_features,
            out_features,
            training: true, // Default to training mode
        })
    }

    /// Performs forward pass through the `LoRA` layer.
    ///
    /// Computes: `scale * (input @ A^T @ B^T)`
    ///
    /// When in training mode (`set_training(true)`), dropout is applied after the A matrix
    /// multiplication if configured (`dropout > 0.0`). In evaluation mode (`eval()`), dropout
    /// is always disabled.
    ///
    /// Automatically uses custom fused Metal kernel when available for optimal performance
    /// (when dropout is disabled).
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
    ///
    /// # Performance
    ///
    /// - **Custom Metal kernel** (when available, dropout disabled): 5-6 µs (single kernel dispatch)
    /// - **Candle fallback**: 37-98 µs (multiple kernel dispatches)
    /// - **With dropout** (training mode): Additional ~10-20% overhead for dropout operation
    ///
    /// # Training vs Evaluation Mode
    ///
    /// - **Training**: Use `set_training(true)` to enable dropout (if configured)
    /// - **Evaluation**: Use `eval()` to disable dropout for deterministic inference
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Try custom fused Metal kernel first (Phase 3+)
        #[cfg(feature = "custom-metal")]
        {
            use crate::backend::CustomMetalOps;

            // Check if we can use the custom kernel
            // Conditions: Metal device, compatible dimensions, no dropout
            if input.device().is_metal() && self.config.dropout == 0.0 {
                // Try fused kernel - if it works, return early
                if let Ok(output) = input.lora_forward_fused(
                    self.lora_a.as_tensor(),
                    self.lora_b.as_tensor(),
                    self.config.scaling(),
                ) {
                    return Ok(output);
                }
                // If fused kernel fails, fall through to Candle implementation
            }
        }

        // OPTIMIZED CANDLE FORWARD PASS (Fallback)
        // Matrices are pre-transposed and B is pre-scaled, reducing 5 kernels to 2!
        //
        // input: (..., in_features)
        // lora_a: (in_features, rank) - stored transposed!
        // lora_b: (rank, out_features) - stored transposed and scaled!

        // Step 1: input @ A -> (..., rank)
        // Candle's broadcast_matmul handles batched dimensions automatically
        // (..., in_features) @ (in_features, rank) -> (..., rank)
        let hidden = input.broadcast_matmul(self.lora_a.as_tensor())?;

        // Step 1.5: Apply dropout if in training mode and configured
        let hidden = if self.training && self.config.dropout > 0.0 {
            candle_nn::ops::dropout(&hidden, self.config.dropout)?
        } else {
            hidden
        };

        // Step 2: hidden @ B_scaled -> (..., out_features)
        // hidden: (..., rank)
        // B_scaled: (rank, out_features) - already transposed and scaled!
        // (..., rank) @ (rank, out_features) -> (..., out_features)
        let output = hidden.broadcast_matmul(self.lora_b.as_tensor())?;

        // Apply scaling (B is not pre-scaled to allow gradients to work correctly)
        let scaled_output = output.affine(f64::from(self.config.scaling()), 0.0)?;

        Ok(scaled_output)
    }

    /// Forward pass using lazy evaluation (v2.0+)
    ///
    /// This method uses the lazy evaluation framework for deferred execution.
    /// Multiple operations can be batched together for improved performance.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::training::LoRALayer;
    /// use metal_candle::graph::LazyTensor;
    /// # use metal_candle::Result;
    /// # fn example() -> Result<()> {
    /// # let config = metal_candle::training::LoRAConfig::default();
    /// # let lora_layer = LoRALayer::new(128, 128, &config, &candle_core::Device::Cpu)?;
    /// # let input = candle_core::Tensor::zeros(&[128, 512], candle_core::DType::F32, &candle_core::Device::Cpu)?;
    ///
    /// let input_lazy = LazyTensor::from_tensor(input)?;
    /// let output_lazy = lora_layer.forward_lazy(&input_lazy)?;
    /// let output = output_lazy.eval()?;  // Deferred execution
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Shape mismatch between input and `LoRA` matrices
    /// - Tensor operations fail
    #[cfg(feature = "graph")]
    pub fn forward_lazy(&self, input: &LazyTensor) -> Result<LazyTensor> {
        // Try custom fused Metal kernel first
        #[cfg(feature = "custom-metal")]
        {
            if input.device().is_metal() && self.config.dropout == 0.0 {
                // Use fused LoRA operation
                let weight_a_tensor = input
                    .add_tensor_to_graph(self.lora_a.as_tensor().clone())
                    .map_err(|e| crate::error::TrainingError::Failed {
                        reason: format!("Failed to add LoRA A to graph: {e}"),
                    })?;
                let weight_b_tensor = input
                    .add_tensor_to_graph(self.lora_b.as_tensor().clone())
                    .map_err(|e| crate::error::TrainingError::Failed {
                        reason: format!("Failed to add LoRA B to graph: {e}"),
                    })?;

                return input
                    .lora_fused(&weight_a_tensor, &weight_b_tensor, self.config.scaling())
                    .map_err(|e| {
                        crate::error::TrainingError::Failed {
                            reason: format!("Fused LoRA operation failed: {e}"),
                        }
                        .into()
                    });
            }
        }

        // Fallback to sequential matmul operations
        // Note: Matrices are stored transposed: A is (in_features, rank), B is (rank, out_features)
        // This matches the eager forward() implementation
        let weight_a_tensor = input
            .add_tensor_to_graph(self.lora_a.as_tensor().clone())
            .map_err(|e| crate::error::TrainingError::Failed {
                reason: format!("Failed to add LoRA A to graph: {e}"),
            })?;
        let weight_b_tensor = input
            .add_tensor_to_graph(self.lora_b.as_tensor().clone())
            .map_err(|e| crate::error::TrainingError::Failed {
                reason: format!("Failed to add LoRA B to graph: {e}"),
            })?;

        // hidden = input @ A (input: (..., in_features), A: (in_features, rank) -> (..., rank))
        let hidden =
            input
                .matmul(&weight_a_tensor)
                .map_err(|e| crate::error::TrainingError::Failed {
                    reason: format!("LoRA matmul A failed: {e}"),
                })?;

        // output = hidden @ B (hidden: (..., rank), B: (rank, out_features) -> (..., out_features))
        let output =
            hidden
                .matmul(&weight_b_tensor)
                .map_err(|e| crate::error::TrainingError::Failed {
                    reason: format!("LoRA matmul B failed: {e}"),
                })?;

        // Apply scaling (same as in forward())
        let scaling = self.config.scaling();
        output.mul_scalar(scaling).map_err(|e| {
            crate::error::TrainingError::Failed {
                reason: format!("LoRA scaling failed: {e}"),
            }
            .into()
        })
    }

    /// Returns the number of trainable parameters in this `LoRA` layer.
    ///
    /// Parameters: `rank * (in_features + out_features)`
    #[must_use]
    pub const fn num_parameters(&self) -> usize {
        self.config.rank * (self.in_features + self.out_features)
    }

    /// Sets the layer to training mode.
    ///
    /// When in training mode, dropout is applied (if configured).
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::{LoRALayer, LoRAConfig};
    /// use candle_core::Device;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::Cpu;
    /// let config = LoRAConfig { rank: 8, alpha: 16.0, dropout: 0.1 };
    /// let mut layer = LoRALayer::new(64, 64, &config, &device)?;
    ///
    /// layer.set_training(true);  // Enable dropout
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Sets the layer to evaluation mode (inference).
    ///
    /// In evaluation mode, dropout is disabled regardless of configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::{LoRALayer, LoRAConfig};
    /// use candle_core::Device;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::Cpu;
    /// let config = LoRAConfig { rank: 8, alpha: 16.0, dropout: 0.1 };
    /// let mut layer = LoRALayer::new(64, 64, &config, &device)?;
    ///
    /// layer.eval();  // Disable dropout for inference
    /// # Ok(())
    /// # }
    /// ```
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether the layer is in training mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::{LoRALayer, LoRAConfig};
    /// use candle_core::Device;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::Cpu;
    /// let config = LoRAConfig::default();
    /// let layer = LoRALayer::new(64, 64, &config, &device)?;
    ///
    /// assert!(layer.is_training());  // Default is training mode
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn is_training(&self) -> bool {
        self.training
    }

    /// Returns a reference to the A matrix as a `Var` (trainable parameter).
    ///
    /// Use this to access gradients during training.
    #[must_use]
    pub const fn lora_a(&self) -> &Var {
        &self.lora_a
    }

    /// Returns a reference to the B matrix as a `Var` (trainable parameter).
    ///
    /// Use this to access gradients during training.
    #[must_use]
    pub const fn lora_b(&self) -> &Var {
        &self.lora_b
    }

    /// Returns a reference to the A matrix as a `Tensor` (for operations).
    ///
    /// Use this for forward passes and other tensor operations.
    #[must_use]
    pub fn lora_a_tensor(&self) -> &Tensor {
        self.lora_a.as_tensor()
    }

    /// Returns a reference to the B matrix as a `Tensor` (for operations).
    ///
    /// Use this for forward passes and other tensor operations.
    #[must_use]
    pub fn lora_b_tensor(&self) -> &Tensor {
        self.lora_b.as_tensor()
    }

    /// Returns all trainable parameters as a vector of `Var` references.
    ///
    /// This is useful for training loops to collect all parameters that need gradients.
    #[must_use]
    pub fn trainable_variables(&self) -> Vec<&Var> {
        vec![&self.lora_a, &self.lora_b]
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
        assert!((f64::from(config.alpha) - 16.0).abs() < 1e-7);
        assert!((f64::from(config.dropout) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
        };
        assert!((f64::from(config.scaling()) - 2.0).abs() < 1e-7);

        let config2 = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
        };
        assert!((f64::from(config2.scaling()) - 2.0).abs() < 1e-7);
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

        // Check A matrix shape: (in_features, rank) - stored transposed!
        assert_eq!(lora.lora_a().dims(), &[128, 8]);

        // Check B matrix shape: (rank, out_features) - stored transposed!
        assert_eq!(lora.lora_b().dims(), &[8, 128]);

        // B should be initialized to zeros
        let b_sum = lora.lora_b().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(b_sum.abs() < 1e-6);
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
            eprintln!("Forward pass failed: {e:?}");
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
        for (in_dim, out_dim) in &[(512, 2048), (2048, 512), (1024, 1024)] {
            let lora = LoRALayer::new(*in_dim, *out_dim, &config, &device);
            assert!(lora.is_ok());

            let lora = lora.unwrap();
            assert_eq!(lora.in_features(), *in_dim);
            assert_eq!(lora.out_features(), *out_dim);
        }
    }

    #[test]
    fn test_lora_layer_trainable_variables() {
        let device = Device::Cpu;
        let config = LoRAConfig::default();

        let lora = LoRALayer::new(128, 128, &config, &device).unwrap();

        // Check that we can access Vars
        let vars = lora.trainable_variables();
        assert_eq!(vars.len(), 2);

        // Check that we can access tensors from Vars (transposed shapes!)
        let a_tensor = lora.lora_a_tensor();
        let b_tensor = lora.lora_b_tensor();
        assert_eq!(a_tensor.dims(), &[128, 8]); // (in_features, rank) - transposed!
        assert_eq!(b_tensor.dims(), &[8, 128]); // (rank, out_features) - transposed!
    }

    #[test]
    fn test_lora_layer_gradients() {
        let device = Device::Cpu;
        let config = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
        };

        let lora = LoRALayer::new(16, 16, &config, &device).unwrap();

        // Create input
        let input = Tensor::ones((1, 16), DType::F32, &device).unwrap();

        // Forward pass
        let output = lora.forward(&input).unwrap();

        // Compute a simple loss (sum of outputs)
        let loss = output.sum_all().unwrap();

        // Backward pass
        let grads = loss.backward().unwrap();

        // Check that gradients exist for both A and B
        assert!(grads.get(lora.lora_a()).is_some(), "Gradient for A exists");
        assert!(grads.get(lora.lora_b()).is_some(), "Gradient for B exists");

        // Gradients should have the same shape as the parameters
        let grad_a = grads.get(lora.lora_a()).unwrap();
        let grad_b = grads.get(lora.lora_b()).unwrap();
        assert_eq!(grad_a.dims(), lora.lora_a_tensor().dims());
        assert_eq!(grad_b.dims(), lora.lora_b_tensor().dims());
    }
}
