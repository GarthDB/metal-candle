//! `LoRA` adapter for applying low-rank adaptation to model layers.
//!
//! This module provides functionality to inject `LoRA` layers into existing
//! transformer models, enabling efficient fine-tuning with a small number
//! of trainable parameters.

use super::lora::{LoRAConfig, LoRALayer};
use crate::error::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Target modules for `LoRA` adaptation.
///
/// Specifies which layers in the model should have `LoRA` applied.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TargetModule {
    /// Query projection in attention
    QProj,
    /// Key projection in attention
    KProj,
    /// Value projection in attention
    VProj,
    /// Output projection in attention
    OProj,
    /// Gate projection in MLP
    GateProj,
    /// Up projection in MLP
    UpProj,
    /// Down projection in MLP
    DownProj,
}

impl TargetModule {
    /// Returns the canonical name for this module.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
        }
    }

    /// Parses a module name into a `TargetModule`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "q_proj" => Some(Self::QProj),
            "k_proj" => Some(Self::KProj),
            "v_proj" => Some(Self::VProj),
            "o_proj" => Some(Self::OProj),
            "gate_proj" => Some(Self::GateProj),
            "up_proj" => Some(Self::UpProj),
            "down_proj" => Some(Self::DownProj),
            _ => None,
        }
    }
}

/// Configuration for `LoRA` adapter.
///
/// Specifies which layers to apply `LoRA` to and the `LoRA` hyperparameters.
///
/// # Examples
///
/// ```
/// use metal_candle::training::{LoRAAdapterConfig, TargetModule};
///
/// // Apply LoRA to Q and V projections only (common choice)
/// let config = LoRAAdapterConfig {
///     rank: 8,
///     alpha: 16.0,
///     dropout: 0.0,
///     target_modules: vec![TargetModule::QProj, TargetModule::VProj],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LoRAAdapterConfig {
    /// Rank of the low-rank decomposition
    pub rank: usize,

    /// Scaling factor for `LoRA` updates
    pub alpha: f32,

    /// Dropout probability
    pub dropout: f32,

    /// Which modules to apply `LoRA` to
    pub target_modules: Vec<TargetModule>,
}

impl Default for LoRAAdapterConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            // By default, apply LoRA to Q and V projections (most common)
            target_modules: vec![TargetModule::QProj, TargetModule::VProj],
        }
    }
}

impl LoRAAdapterConfig {
    /// Creates a `LoRAConfig` from this adapter configuration.
    #[must_use]
    pub const fn to_lora_config(&self) -> LoRAConfig {
        LoRAConfig {
            rank: self.rank,
            alpha: self.alpha,
            dropout: self.dropout,
        }
    }

    /// Checks if a module is targeted for `LoRA`.
    #[must_use]
    pub fn is_target(&self, module: &TargetModule) -> bool {
        self.target_modules.contains(module)
    }
}

/// `LoRA` adapter for a transformer model.
///
/// Manages `LoRA` layers applied to specific modules in the model.
/// Each `LoRA` layer adds a trainable low-rank update to a frozen linear layer.
///
/// # Architecture
///
/// For a frozen linear layer with weight W:
/// ```text
/// output = (W + ΔW) @ input
///        = W @ input + ΔW @ input
///        = frozen_output + lora_output
/// ```
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::{LoRAAdapter, LoRAAdapterConfig, TargetModule};
/// use candle_core::Device;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let config = LoRAAdapterConfig::default();
///
/// // Create adapter for a model with hidden_size=768
/// let adapter = LoRAAdapter::new(768, 768, 32, &config, &device)?;
///
/// // Get number of trainable parameters
/// println!("Trainable params: {}", adapter.num_trainable_parameters());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct LoRAAdapter {
    /// `LoRA` layers indexed by (`layer_idx`, `module_name`)
    layers: HashMap<String, LoRALayer>,

    /// Adapter configuration
    config: LoRAAdapterConfig,

    /// Number of transformer layers
    num_layers: usize,
}

impl LoRAAdapter {
    /// Creates a new `LoRA` adapter.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Model hidden dimension
    /// * `intermediate_size` - MLP intermediate dimension (for MLP modules)
    /// * `num_layers` - Number of transformer layers in the model
    /// * `config` - Adapter configuration
    /// * `device` - Device to place tensors on
    ///
    /// # Errors
    ///
    /// Returns an error if `LoRA` layer creation fails.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        config: &LoRAAdapterConfig,
        device: &Device,
    ) -> Result<Self> {
        let lora_config = config.to_lora_config();
        let mut layers = HashMap::new();

        // Create LoRA layers for each target module in each transformer layer
        for layer_idx in 0..num_layers {
            for target in &config.target_modules {
                let (in_features, out_features) = match target {
                    TargetModule::QProj
                    | TargetModule::KProj
                    | TargetModule::VProj
                    | TargetModule::OProj => (hidden_size, hidden_size),
                    TargetModule::GateProj | TargetModule::UpProj => {
                        (hidden_size, intermediate_size)
                    }
                    TargetModule::DownProj => (intermediate_size, hidden_size),
                };

                let lora_layer = LoRALayer::new(in_features, out_features, &lora_config, device)?;

                let key = format!("layers.{}.{}", layer_idx, target.name());
                layers.insert(key, lora_layer);
            }
        }

        Ok(Self {
            layers,
            config: config.clone(),
            num_layers,
        })
    }

    /// Applies `LoRA` to a layer's output.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the transformer layer
    /// * `module` - Which module (`q_proj`, `v_proj`, etc.)
    /// * `input` - Input to the linear layer (before frozen projection)
    ///
    /// # Returns
    ///
    /// The `LoRA` delta to add to the frozen layer output.
    /// Returns `None` if this layer/module doesn't have `LoRA` applied.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(
        &self,
        layer_idx: usize,
        module: &TargetModule,
        input: &Tensor,
    ) -> Result<Option<Tensor>> {
        let key = format!("layers.{}.{}", layer_idx, module.name());

        if let Some(lora_layer) = self.layers.get(&key) {
            let delta = lora_layer.forward(input)?;
            Ok(Some(delta))
        } else {
            Ok(None)
        }
    }

    /// Returns the total number of trainable parameters.
    ///
    /// This is the sum of parameters in all `LoRA` layers.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        self.layers.values().map(LoRALayer::num_parameters).sum()
    }

    /// Returns the number of frozen (non-trainable) parameters.
    ///
    /// This would be all model parameters minus the `LoRA` parameters.
    /// Note: This requires knowing the model's total parameter count.
    #[must_use]
    pub fn num_frozen_parameters(&self, total_model_params: usize) -> usize {
        total_model_params.saturating_sub(self.num_trainable_parameters())
    }

    /// Returns the adapter configuration.
    #[must_use]
    pub const fn config(&self) -> &LoRAAdapterConfig {
        &self.config
    }

    /// Returns the number of transformer layers.
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns an iterator over all `LoRA` layers.
    pub fn layers(&self) -> impl Iterator<Item = (&String, &LoRALayer)> {
        self.layers.iter()
    }

    /// Gets a specific `LoRA` layer by key.
    #[must_use]
    pub fn get_layer(&self, layer_idx: usize, module: &TargetModule) -> Option<&LoRALayer> {
        let key = format!("layers.{}.{}", layer_idx, module.name());
        self.layers.get(&key)
    }

    /// Merges `LoRA` weights back into the base model weights.
    ///
    /// Computes: `W_new = W_base + (B @ A) * scaling`
    ///
    /// This is useful for inference after training, as it eliminates
    /// the overhead of separate `LoRA` computation.
    ///
    /// # Arguments
    ///
    /// * `base_weight` - The frozen base weight matrix (`out_features`, `in_features`)
    /// * `layer_idx` - Index of the transformer layer
    /// * `module` - Which module to merge
    ///
    /// # Returns
    ///
    /// The merged weight matrix, or the original if no `LoRA` is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail.
    pub fn merge_weights(
        &self,
        base_weight: &Tensor,
        layer_idx: usize,
        module: &TargetModule,
    ) -> Result<Tensor> {
        let key = format!("layers.{}.{}", layer_idx, module.name());

        if let Some(lora_layer) = self.layers.get(&key) {
            // Compute ΔW = B @ A * scaling
            let lora_a = lora_layer.lora_a_tensor();
            let lora_b = lora_layer.lora_b_tensor();

            // A: (rank, in_features)
            // B: (out_features, rank)
            // B @ A = (out_features, in_features)
            let delta_w = lora_b.matmul(lora_a)?;

            // Scale by alpha/rank
            let scaling = lora_layer.config().scaling();
            let scaled_delta = (delta_w * f64::from(scaling))?;

            // W_new = W_base + scaled_delta
            let merged = base_weight.add(&scaled_delta)?;
            Ok(merged)
        } else {
            // No LoRA for this layer/module, return base weight unchanged
            Ok(base_weight.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_module_name() {
        assert_eq!(TargetModule::QProj.name(), "q_proj");
        assert_eq!(TargetModule::VProj.name(), "v_proj");
        assert_eq!(TargetModule::GateProj.name(), "gate_proj");
    }

    #[test]
    fn test_target_module_from_name() {
        assert_eq!(TargetModule::from_name("q_proj"), Some(TargetModule::QProj));
        assert_eq!(TargetModule::from_name("v_proj"), Some(TargetModule::VProj));
        assert_eq!(TargetModule::from_name("invalid"), None);
    }

    #[test]
    fn test_lora_adapter_config_default() {
        let config = LoRAAdapterConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.target_modules.len(), 2);
        assert!(config.is_target(&TargetModule::QProj));
        assert!(config.is_target(&TargetModule::VProj));
        assert!(!config.is_target(&TargetModule::KProj));
    }

    #[test]
    fn test_lora_adapter_creation() {
        let device = Device::Cpu;
        let config = LoRAAdapterConfig::default();

        let adapter = LoRAAdapter::new(768, 2048, 4, &config, &device);
        assert!(adapter.is_ok());

        let adapter = adapter.unwrap();
        assert_eq!(adapter.num_layers(), 4);

        // Should have LoRA for 2 modules * 4 layers = 8 LoRA layers
        assert_eq!(adapter.layers.len(), 8);
    }

    #[test]
    fn test_lora_adapter_trainable_parameters() {
        let device = Device::Cpu;
        let config = LoRAAdapterConfig {
            rank: 8,
            target_modules: vec![TargetModule::QProj, TargetModule::VProj],
            ..Default::default()
        };

        let adapter = LoRAAdapter::new(768, 2048, 4, &config, &device).unwrap();

        // Each LoRA layer: rank * (in_features + out_features)
        // q_proj, v_proj: 8 * (768 + 768) = 12,288 params each
        // Total: 2 modules * 4 layers * 12,288 = 98,304 params
        assert_eq!(adapter.num_trainable_parameters(), 98_304);
    }

    #[test]
    fn test_lora_adapter_forward() {
        let device = Device::Cpu;
        let config = LoRAAdapterConfig::default();

        let adapter = LoRAAdapter::new(768, 2048, 2, &config, &device).unwrap();

        // Create input tensor
        let input = Tensor::randn(0f32, 1f32, (2, 16, 768), &device).unwrap();

        // Forward through layer 0, q_proj (should have LoRA)
        let output = adapter.forward(0, &TargetModule::QProj, &input);
        assert!(output.is_ok());
        assert!(output.unwrap().is_some());

        // Forward through layer 0, k_proj (should NOT have LoRA by default)
        let output = adapter.forward(0, &TargetModule::KProj, &input);
        assert!(output.is_ok());
        assert!(output.unwrap().is_none());
    }

    #[test]
    fn test_lora_adapter_get_layer() {
        let device = Device::Cpu;
        let config = LoRAAdapterConfig::default();

        let adapter = LoRAAdapter::new(768, 2048, 2, &config, &device).unwrap();

        // Should find q_proj in layer 0
        assert!(adapter.get_layer(0, &TargetModule::QProj).is_some());

        // Should not find k_proj (not in target modules)
        assert!(adapter.get_layer(0, &TargetModule::KProj).is_none());

        // Should not find q_proj in layer 5 (only 2 layers)
        assert!(adapter.get_layer(5, &TargetModule::QProj).is_none());
    }

    #[test]
    fn test_lora_adapter_merge_weights() {
        let device = Device::Cpu;
        let config = LoRAAdapterConfig::default();

        let adapter = LoRAAdapter::new(768, 2048, 1, &config, &device).unwrap();

        // Create a base weight matrix: (out_features=768, in_features=768)
        let base_weight = Tensor::zeros((768, 768), candle_core::DType::F32, &device).unwrap();

        // Merge with q_proj (should have LoRA)
        let merged = adapter.merge_weights(&base_weight, 0, &TargetModule::QProj);
        assert!(merged.is_ok());

        let merged = merged.unwrap();
        assert_eq!(merged.dims(), &[768, 768]);

        // Merge with k_proj (no LoRA, should return base_weight)
        let merged = adapter.merge_weights(&base_weight, 0, &TargetModule::KProj);
        assert!(merged.is_ok());
    }
}
