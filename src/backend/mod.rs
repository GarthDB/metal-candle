//! Backend abstraction layer for Metal device operations.
//!
//! This module provides high-level abstractions over Candle's Metal backend,
//! making it easier to work with tensors and device operations on Apple Silicon.

pub mod device;
pub mod tensor;

// Re-export key types for convenience
pub use device::{Device, DeviceInfo, DeviceType};
pub use tensor::TensorExt;
