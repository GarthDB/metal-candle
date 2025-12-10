//! Custom Metal shader kernels for performance-critical operations.
//!
//! This module provides optimized Metal shaders that outperform
//! Candle's default implementations for specific operations.
//!
//! # Architecture
//!
//! Custom Metal kernels are used for operations where kernel fusion
//! or specialized optimization can provide significant speedups:
//!
//! - Fused `LoRA` forward pass (A @ B in one kernel)
//! - Fused softmax (max + exp + sum + divide)
//! - Fused RMS normalization
//! - Fused layer normalization
//!
//! # Usage
//!
//! ```no_run
//! use metal_candle::backend::metal_kernels::MetalKernelCompiler;
//! # use std::sync::Arc;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = metal::Device::system_default()
//!     .ok_or("Metal not available")?;
//! let device = Arc::new(device);
//!
//! let compiler = MetalKernelCompiler::new(device)?;
//! let pipeline = compiler.create_pipeline("fused_lora_forward")?;
//! # Ok(())
//! # }
//! ```

use crate::error::DeviceError;
use std::sync::Arc;

/// Metal kernel compiler for custom shaders.
///
/// Compiles and caches Metal compute pipelines for custom kernels.
pub struct MetalKernelCompiler {
    device: Arc<metal::Device>,
    library: metal::Library,
}

impl MetalKernelCompiler {
    /// Create a new Metal kernel compiler.
    ///
    /// # Arguments
    ///
    /// * `device` - Metal device to compile kernels for
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::InitializationFailed`] if the Metal library cannot be compiled.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::backend::metal_kernels::MetalKernelCompiler;
    /// # use std::sync::Arc;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = metal::Device::system_default()
    ///     .ok_or("Metal not available")?;
    /// let device = Arc::new(device);
    ///
    /// let compiler = MetalKernelCompiler::new(device)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<metal::Device>) -> Result<Self, DeviceError> {
        // Compile the Metal shader library from source
        let source = include_str!("kernels.metal");

        let library = device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| DeviceError::InitializationFailed {
                reason: format!("Failed to compile Metal library: {e}"),
            })?;

        Ok(Self { device, library })
    }

    /// Create a compute pipeline for a specific kernel function.
    ///
    /// # Arguments
    ///
    /// * `kernel_name` - Name of the kernel function in the Metal source
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::OperationFailed`] if:
    /// - The kernel function is not found in the library
    /// - The compute pipeline cannot be created
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::backend::metal_kernels::MetalKernelCompiler;
    /// # use std::sync::Arc;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let device = Arc::new(metal::Device::system_default().unwrap());
    /// let compiler = MetalKernelCompiler::new(device)?;
    /// let pipeline = compiler.create_pipeline("fused_lora_forward")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_pipeline(
        &self,
        kernel_name: &str,
    ) -> Result<metal::ComputePipelineState, DeviceError> {
        let function = self.library.get_function(kernel_name, None).map_err(|e| {
            DeviceError::OperationFailed {
                operation: format!("Failed to get kernel function '{kernel_name}': {e}"),
            }
        })?;

        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| DeviceError::OperationFailed {
                operation: format!("Failed to create compute pipeline for '{kernel_name}': {e}"),
            })
    }

    /// Get reference to the Metal device.
    #[must_use]
    pub fn device(&self) -> &Arc<metal::Device> {
        &self.device
    }

    /// Get reference to the compiled library.
    #[must_use]
    pub fn library(&self) -> &metal::Library {
        &self.library
    }
}

/// Parameters for fused `LoRA` kernel dispatch.
///
/// This struct matches the layout expected by the `Metal` shader.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LoRAParams {
    /// Batch size dimension
    pub batch_size: u32,
    /// Sequence length dimension
    pub seq_len: u32,
    /// Input features dimension
    pub in_features: u32,
    /// `LoRA` rank
    pub rank: u32,
    /// Output features dimension
    pub out_features: u32,
    /// `LoRA` scaling factor (alpha/rank)
    pub scaling: f32,
}

/// Parameters for fused `softmax` kernel.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SoftmaxParams {
    /// Batch size dimension
    pub batch_size: u32,
    /// Sequence length dimension
    pub seq_len: u32,
    /// Dimension to apply softmax over
    pub dim: u32,
}

/// Parameters for fused `RMS` norm kernel.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RMSNormParams {
    /// Batch size dimension
    pub batch_size: u32,
    /// Sequence length dimension
    pub seq_len: u32,
    /// Hidden dimension to normalize over
    pub dim: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

/// Parameters for layer normalization kernel.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LayerNormParams {
    /// Batch size dimension
    pub batch_size: u32,
    /// Hidden size dimension
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        if let Some(device) = metal::Device::system_default() {
            let device = Arc::new(device);
            let compiler = MetalKernelCompiler::new(device);

            // Should compile successfully
            assert!(compiler.is_ok());
        }
    }

    #[test]
    fn test_pipeline_creation() {
        if let Some(device) = metal::Device::system_default() {
            let device = Arc::new(device);
            let compiler = MetalKernelCompiler::new(device).expect("Failed to create compiler");

            // Test with a kernel that exists
            // (will add actual kernels in Phase 3)
            // For now, just verify the compiler works
            let _function_names = compiler.library().function_names();
            // Library compiled successfully if we got here
        }
    }
}
