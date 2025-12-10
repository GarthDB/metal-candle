//! Backend abstraction layer for Metal device operations.
//!
//! This module provides high-level abstractions over Candle's Metal backend,
//! making it easier to work with tensors and device operations on Apple Silicon.
//!
//! # Custom Metal Kernels
//!
//! When the `custom-metal` feature is enabled, this module also provides
//! high-performance custom Metal kernels for performance-critical operations.

pub mod device;
pub mod tensor;

//Custom Metal kernel support (feature-gated)
#[cfg(feature = "custom-metal")]
pub mod custom_ops;
#[cfg(feature = "custom-metal")]
pub mod metal_kernels;
#[cfg(feature = "custom-metal")]
pub mod metal_ops;

// Re-export key types for convenience
pub use device::{Device, DeviceInfo, DeviceType};
pub use tensor::TensorExt;

#[cfg(feature = "custom-metal")]
pub use custom_ops::{layer_norm, FusedLoRAOp, FusedRMSNormOp, FusedSoftmaxOp, LayerNormOp};
#[cfg(feature = "custom-metal")]
pub use metal_kernels::MetalKernelCompiler;
#[cfg(feature = "custom-metal")]
pub use metal_ops::CustomMetalOps;
