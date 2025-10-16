//! Training utilities for `LoRA` fine-tuning.
//!
//! This module provides everything needed to train `LoRA` (Low-Rank Adaptation)
//! adapters for fine-tuning large language models efficiently.
//!
//! # Overview
//!
//! `LoRA` enables efficient fine-tuning by adding trainable low-rank matrices
//! to existing model layers, requiring only a fraction of the parameters
//! compared to full fine-tuning.
//!
//! # Key Components
//!
//! - [`LoRALayer`] - Individual low-rank adaptation layer
//! - [`LoRAConfig`] - Configuration for `LoRA` adapters
//!
//! # Example
//!
//! ```no_run
//! use metal_candle::training::{LoRAConfig, LoRALayer};
//! use candle_core::Device;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::Cpu;
//! let config = LoRAConfig {
//!     rank: 8,
//!     alpha: 16.0,
//!     dropout: 0.0,
//! };
//!
//! // Create LoRA layer for a linear layer with in_features=768, out_features=768
//! let lora = LoRALayer::new(768, 768, &config, &device)?;
//! # Ok(())
//! # }
//! ```

pub mod adapter;
pub mod lora;
pub mod loss;
pub mod optimizer;
pub mod scheduler;
pub mod trainer;

// Re-export main types
pub use adapter::{LoRAAdapter, LoRAAdapterConfig, TargetModule};
pub use lora::{LoRAConfig, LoRALayer};
pub use loss::{cross_entropy_loss, cross_entropy_loss_with_smoothing};
pub use optimizer::{AdamW, AdamWConfig};
pub use scheduler::LRScheduler;
pub use trainer::{StepMetrics, TrainingStep};
