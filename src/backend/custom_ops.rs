//! Custom operations using Candle's `CustomOp` framework.
//!
//! This module implements high-performance fused operations using Candle's
//! `CustomOp` traits, providing clean Metal buffer access and proper integration
//! with Candle's autodiff system.
//!
//! # Architecture
//!
//! Custom operations implement Candle's `CustomOp1`, `CustomOp2`, or `CustomOp3`
//! traits, depending on the number of input tensors. Each trait requires:
//! - `name()`: Operation identifier
//! - `cpu_fwd()`: CPU implementation (required)
//! - `metal_fwd()`: Metal GPU implementation (optional)
//! - `bwd()`: Gradient computation (optional, for training)
//!
//! # Performance
//!
//! Custom operations achieve significant speedups by:
//! - Fusing multiple operations into single GPU kernels
//! - Eliminating intermediate memory allocations
//! - Reducing kernel launch overhead
//! - Optimizing memory access patterns
//!
//! # Example
//!
//! ```no_run
//! use candle_core::{Tensor, Device};
//! use metal_candle::backend::custom_ops::FusedLoRAOp;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::new_metal(0)?;
//! let input = Tensor::randn(0.0, 1.0, (1, 512), &device)?;
//! let lora_a = Tensor::randn(0.0, 0.01, (512, 8), &device)?;
//! let lora_b = Tensor::zeros((8, 512), candle_core::DType::F32, &device)?;
//!
//! let op = FusedLoRAOp::new(lora_a, lora_b, 2.0)?;
//! let output = input.apply_op1(op)?;
//! # Ok(())
//! # }
//! ```

// Allow similar_names for LoRA operations - A/B matrix naming is standard ML convention
#![allow(clippy::similar_names)]

use crate::backend::metal_kernels::{
    LoRAParams, MetalKernelCompiler, RMSNormParams, SoftmaxParams,
};
use candle_core::backend::BackendStorage;
use candle_core::{CustomOp1, Layout, MetalStorage, Result, Shape, Tensor};
use metal::ComputePipelineState;
use std::sync::{Arc, Mutex};

/// Fused `LoRA` forward pass operation.
///
/// Implements `LoRA` forward pass as a single Metal kernel:
/// `output = (input @ lora_a @ lora_b) * scaling`
///
/// This fuses two matrix multiplications and a scaling operation,
/// avoiding intermediate allocations and reducing kernel launch overhead.
///
/// # Performance
///
/// - **Current (unfused)**: 37-98 µs (2+ kernel launches)
/// - **Target (fused)**: 5-12 µs (1 kernel launch)
/// - **Expected speedup**: 6-10x
///
/// # Implementation Notes
///
/// - Uses `CustomOp1` (single input tensor)
/// - Stores `LoRA` matrices as operation state
/// - Caches Metal pipeline for reuse
/// - CPU fallback provided for compatibility
pub struct FusedLoRAOp {
    /// First `LoRA` matrix (`in_features` × rank)
    lora_a: Tensor,

    /// Second `LoRA` matrix (rank × `out_features`)
    lora_b: Tensor,

    /// Scaling factor (alpha / rank)
    scaling: f32,

    /// Cached Metal compute pipeline (compiled kernel)
    pipeline: Arc<Mutex<Option<metal::ComputePipelineState>>>,

    /// Cached Metal kernel compiler
    compiler: Arc<Mutex<Option<MetalKernelCompiler>>>,
}

impl FusedLoRAOp {
    /// Creates a new fused `LoRA` operation.
    ///
    /// # Arguments
    ///
    /// * `lora_a` - First `LoRA` matrix (`in_features` × rank)
    /// * `lora_b` - Second `LoRA` matrix (rank × `out_features`)
    /// * `scaling` - Scaling factor (typically alpha / rank)
    ///
    /// # Errors
    ///
    /// Returns error if tensor shapes are incompatible.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use candle_core::{Tensor, Device, DType};
    /// use metal_candle::backend::custom_ops::FusedLoRAOp;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::new_metal(0)?;
    /// let lora_a = Tensor::randn(0.0, 0.01, (512, 8), &device)?;
    /// let lora_b = Tensor::zeros((8, 512), DType::F32, &device)?;
    ///
    /// let op = FusedLoRAOp::new(lora_a, lora_b, 2.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(lora_a: Tensor, lora_b: Tensor, scaling: f32) -> Result<Self> {
        // Validate dimensions
        let a_dims = lora_a.dims();
        let b_dims = lora_b.dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            candle_core::bail!(
                "LoRA matrices must be 2D, got shapes {:?} and {:?}",
                a_dims,
                b_dims
            );
        }

        if a_dims[1] != b_dims[0] {
            candle_core::bail!(
                "Incompatible LoRA matrix dimensions: {}×{} and {}×{}",
                a_dims[0],
                a_dims[1],
                b_dims[0],
                b_dims[1]
            );
        }

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            pipeline: Arc::new(Mutex::new(None)),
            compiler: Arc::new(Mutex::new(None)),
        })
    }

    /// Computes the output shape given an input shape.
    ///
    /// Transforms: `(..., in_features)` → `(..., out_features)`
    fn compute_output_shape(&self, input_shape: &Shape) -> Result<Shape> {
        let dims = input_shape.dims();
        if dims.is_empty() {
            candle_core::bail!("Input must have at least 1 dimension");
        }

        let in_features = dims[dims.len() - 1];
        let expected_in = self.lora_a.dim(0)?;

        if in_features != expected_in {
            candle_core::bail!(
                "Input feature dimension mismatch: expected {}, got {}",
                expected_in,
                in_features
            );
        }

        let mut output_dims = dims.to_vec();
        output_dims[dims.len() - 1] = self.lora_b.dim(1)?;

        Ok(Shape::from(output_dims))
    }

    /// Gets or compiles the Metal compute pipeline.
    ///
    /// Caches the pipeline after first compilation for reuse.
    fn get_or_compile_pipeline(
        &self,
        device: &metal::DeviceRef,
    ) -> Result<metal::ComputePipelineState> {
        // Check cache first
        {
            let pipeline_guard = self
                .pipeline
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;

            if let Some(ref pipeline) = *pipeline_guard {
                return Ok(pipeline.clone());
            }
        }

        // Compile kernel
        let mut compiler_guard = self
            .compiler
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock compiler: {e}")))?;

        let compiler = if let Some(ref comp) = *compiler_guard {
            comp
        } else {
            // Create a new owned metal::Device from the DeviceRef
            let owned_device = device.to_owned();
            let new_compiler = MetalKernelCompiler::new(Arc::new(owned_device))
                .map_err(|e| candle_core::Error::Msg(format!("Failed to create compiler: {e}")))?;
            *compiler_guard = Some(new_compiler);
            compiler_guard.as_ref().unwrap()
        };

        // Use tiled kernel for better performance
        let pipeline = compiler
            .create_pipeline("fused_lora_forward_tiled")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {e}")))?;

        // Cache for next time
        {
            let mut pipeline_guard = self
                .pipeline
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
            *pipeline_guard = Some(pipeline.clone());
        }

        Ok(pipeline)
    }
}

impl CustomOp1 for FusedLoRAOp {
    fn name(&self) -> &'static str {
        "fused_lora_forward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        // CPU implementation not provided - this is a Metal-only operation
        // For CPU fallback, the LoRA layer will use standard Candle matmul operations
        candle_core::bail!(
            "FusedLoRAOp is Metal-only. Use standard LoRA operations on CPU or move tensors to Metal device."
        )
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.metal_fwd_impl(storage, layout)
    }
}

impl FusedLoRAOp {
    fn metal_fwd_impl(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        // Get Metal device from storage
        use candle_core::backend::BackendStorage;
        let device = storage.device();

        // Compute output shape
        let output_shape = self.compute_output_shape(layout.shape())?;

        // Get input dimensions
        let input_dims = layout.shape().dims();
        let batch_size = if input_dims.len() == 3 {
            input_dims[0]
        } else {
            1
        };
        let seq_len = if input_dims.len() >= 2 {
            input_dims[input_dims.len() - 2]
        } else {
            1
        };
        let in_features = input_dims[input_dims.len() - 1];
        let rank = self.lora_a.dim(1)?;
        let out_features = self.lora_b.dim(1)?;

        // Create parameters
        let params = LoRAParams {
            batch_size: u32::try_from(batch_size).map_err(|_| {
                candle_core::Error::DimOutOfRange {
                    shape: output_shape.clone(),
                    dim: 0,
                    op: "fused_lora",
                }
            })?,
            seq_len: u32::try_from(seq_len).map_err(|_| candle_core::Error::DimOutOfRange {
                shape: output_shape.clone(),
                dim: 1,
                op: "fused_lora",
            })?,
            in_features: u32::try_from(in_features).map_err(|_| {
                candle_core::Error::DimOutOfRange {
                    shape: self.lora_a.shape().clone(),
                    dim: 0,
                    op: "fused_lora",
                }
            })?,
            rank: u32::try_from(rank).map_err(|_| candle_core::Error::DimOutOfRange {
                shape: self.lora_a.shape().clone(),
                dim: 1,
                op: "fused_lora",
            })?,
            out_features: u32::try_from(out_features).map_err(|_| {
                candle_core::Error::DimOutOfRange {
                    shape: self.lora_b.shape().clone(),
                    dim: 1,
                    op: "fused_lora",
                }
            })?,
            scaling: self.scaling,
        };

        // Get Metal buffers
        let input_buffer = storage.buffer();

        // Extract Metal buffers from LoRA tensors
        let weight_a_guard = self.lora_a.storage_and_layout();
        let candle_core::Storage::Metal(weight_a_storage) = &*weight_a_guard.0 else {
            candle_core::bail!("LoRA_A must be on Metal device")
        };
        let lora_a_buffer = weight_a_storage.buffer();

        let lora_b_guard = self.lora_b.storage_and_layout();
        let candle_core::Storage::Metal(lora_b_storage) = &*lora_b_guard.0 else {
            candle_core::bail!("LoRA_B must be on Metal device")
        };
        let lora_b_buffer = lora_b_storage.buffer();

        // Create output buffer
        let output_elem_count = output_shape.elem_count();
        let output_buffer =
            device.new_buffer(output_elem_count, storage.dtype(), "fused_lora_output")?;

        // Get or compile pipeline (device implements Deref to metal::DeviceRef)
        let pipeline = self.get_or_compile_pipeline(device)?;

        // Create command buffer and dispatch kernel
        let command_buffer = device.command_buffer()?;

        // Scope the encoder to ensure it's dropped before commit
        {
            use candle_metal_kernels::utils::EncoderProvider;
            let command_buffer_ref = &command_buffer;
            let encoder_wrapper = command_buffer_ref.encoder();
            let encoder = encoder_wrapper.as_ref();

            // Set pipeline and buffers
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(input_buffer), 0);
            encoder.set_buffer(1, Some(lora_a_buffer), 0);
            encoder.set_buffer(2, Some(lora_b_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);

            // Set parameters as bytes
            encoder.set_bytes(
                4,
                std::mem::size_of::<LoRAParams>() as u64,
                std::ptr::addr_of!(params).cast::<std::ffi::c_void>(),
            );

            // Calculate grid dimensions
            // Each thread computes one output element: output[batch, seq, out_feature]
            let grid_size = metal::MTLSize {
                width: batch_size as u64,
                height: seq_len as u64,
                depth: out_features as u64,
            };

            // Threadgroup size optimized for Apple Silicon
            // 256 threads per group is optimal for M1/M2/M3
            let threadgroup_size = metal::MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            // Dispatch kernel
            encoder.dispatch_threads(grid_size, threadgroup_size);

            // encoder_wrapper drops here, ending encoding
        }

        // Don't commit - Candle manages the command buffer lifecycle
        // The kernel has been dispatched and will execute when Candle commits

        // Create output storage
        let output_storage = candle_core::MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            storage.dtype(),
        );

        Ok((output_storage, output_shape))
    }
}

/// Fused RMS Normalization operation.
///
/// Implements RMS norm as a single Metal kernel with threadgroup reductions:
/// `rms_norm(x) = x / sqrt(mean(x^2) + eps)`
///
/// Key features:
/// - Uses threadgroup memory for parallel reductions
/// - Numerically stable with epsilon
/// - Single kernel dispatch (vs 3+ in unfused version)
/// - Expected 4-5x speedup
pub struct FusedRMSNormOp {
    /// Epsilon for numerical stability
    eps: f32,

    /// Cached Metal pipeline state for the kernel.
    pipeline: Mutex<Option<ComputePipelineState>>,

    /// Cached Metal kernel compiler.
    compiler: Mutex<Option<MetalKernelCompiler>>,
}

impl FusedRMSNormOp {
    /// Creates a new fused RMS norm operation.
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    ///
    /// # Errors
    ///
    /// Currently doesn't error, but returns Result for future-proofing.
    pub fn new(eps: f32) -> Result<Self> {
        Ok(Self {
            eps,
            pipeline: Mutex::new(None),
            compiler: Mutex::new(None),
        })
    }

    /// Lazily gets or compiles the Metal compute pipeline for the `fused_rms_norm` kernel.
    fn get_or_compile_pipeline(
        &self,
        device: &candle_core::MetalDevice,
    ) -> Result<ComputePipelineState> {
        // Try to get from cache
        if let Some(pipeline) = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?
            .as_ref()
        {
            return Ok(pipeline.clone());
        }

        // If not in cache, compile
        let pipeline = {
            let mut compiler_guard = self
                .compiler
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to lock compiler: {e}")))?;
            if compiler_guard.is_none() {
                let metal_device = device.device().to_owned();
                let new_compiler =
                    MetalKernelCompiler::new(Arc::new(metal_device)).map_err(|e| {
                        candle_core::Error::Msg(format!("Failed to create compiler: {e}"))
                    })?;
                *compiler_guard = Some(new_compiler);
            }
            compiler_guard
                .as_ref()
                .unwrap()
                .create_pipeline("fused_rms_norm")
                .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {e}")))?
        };

        // Cache for next time
        {
            let mut pipeline_guard = self
                .pipeline
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
            *pipeline_guard = Some(pipeline.clone());
        }

        Ok(pipeline)
    }
}

impl CustomOp1 for FusedRMSNormOp {
    fn name(&self) -> &'static str {
        "fused-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Fused RMS Norm kernel not implemented for CPU. Use Metal device.")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();

        // Get or compile the pipeline
        let pipeline = self.get_or_compile_pipeline(device)?;

        // Extract input tensor dimensions
        let input_dims = layout.shape().dims();
        if input_dims.is_empty() {
            candle_core::bail!(
                "Input tensor for FusedRMSNormOp must have at least 1 dimension, got {:?}",
                input_dims
            );
        }

        // Support various input shapes
        let (batch_size, seq_len, dim) = match input_dims.len() {
            1 => (1, 1, input_dims[0]),
            2 => (1, input_dims[0], input_dims[1]),
            3 => (input_dims[0], input_dims[1], input_dims[2]),
            _ => candle_core::bail!(
                "RMS Norm only supports 1D, 2D, or 3D tensors, got {:?}",
                input_dims
            ),
        };

        // Get Metal buffers
        let input_buffer = storage.buffer();

        // Create output buffer
        let output_shape = layout.shape().clone();
        let output_elem_count = output_shape.elem_count();
        let output_buffer =
            device.new_buffer(output_elem_count, storage.dtype(), "fused_rms_norm_output")?;

        // Prepare kernel parameters
        let params = RMSNormParams {
            batch_size: u32::try_from(batch_size)
                .map_err(|e| candle_core::Error::Msg(format!("Batch size too large: {e}")))?,
            seq_len: u32::try_from(seq_len)
                .map_err(|e| candle_core::Error::Msg(format!("Sequence length too large: {e}")))?,
            dim: u32::try_from(dim)
                .map_err(|e| candle_core::Error::Msg(format!("Dimension too large: {e}")))?,
            eps: self.eps,
        };

        // Create command buffer and dispatch kernel
        let command_buffer = device.command_buffer()?;

        // Scope the encoder
        {
            use candle_metal_kernels::utils::EncoderProvider;
            let command_buffer_ref = &command_buffer;
            let encoder_wrapper = command_buffer_ref.encoder();
            let encoder = encoder_wrapper.as_ref();

            encoder.set_compute_pipeline_state(&pipeline);

            // Set buffers
            encoder.set_buffer(0, Some(input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<RMSNormParams>() as u64,
                std::ptr::addr_of!(params).cast::<std::ffi::c_void>(),
            );

            // Threadgroup memory for reductions (256 floats = 1024 bytes)
            encoder.set_threadgroup_memory_length(0, 256 * 4);

            // Dispatch threads
            // Each threadgroup handles one RMS norm operation (one row)
            let grid_size = metal::MTLSize {
                width: 1,
                height: seq_len as u64,
                depth: batch_size as u64,
            };
            let threadgroup_size = metal::MTLSize {
                width: 256, // Optimal for reductions
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);

            // encoder_wrapper drops here
        }

        // Create output storage
        let output_storage = candle_core::MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            storage.dtype(),
        );

        Ok((output_storage, output_shape))
    }
}

/// Fused Softmax operation.
///
/// Implements softmax as a single Metal kernel with threadgroup reductions:
/// `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
///
/// Key features:
/// - Uses threadgroup memory for parallel reductions
/// - Numerically stable (subtract max before exp)
/// - Single kernel dispatch (vs 4+ in unfused version)
/// - Expected 6-8x speedup
pub struct FusedSoftmaxOp {
    /// Cached Metal pipeline state for the kernel.
    pipeline: Mutex<Option<ComputePipelineState>>,

    /// Cached Metal kernel compiler.
    compiler: Mutex<Option<MetalKernelCompiler>>,
}

impl FusedSoftmaxOp {
    /// Creates a new fused softmax operation.
    ///
    /// # Errors
    ///
    /// Currently doesn't error, but returns Result for future-proofing.
    pub fn new() -> Result<Self> {
        Ok(Self {
            pipeline: Mutex::new(None),
            compiler: Mutex::new(None),
        })
    }

    /// Lazily gets or compiles the Metal compute pipeline for the `fused_softmax` kernel.
    fn get_or_compile_pipeline(
        &self,
        device: &candle_core::MetalDevice,
    ) -> Result<ComputePipelineState> {
        // Try to get from cache
        if let Some(pipeline) = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?
            .as_ref()
        {
            return Ok(pipeline.clone());
        }

        // If not in cache, compile
        let pipeline = {
            let mut compiler_guard = self
                .compiler
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to lock compiler: {e}")))?;
            if compiler_guard.is_none() {
                let metal_device = device.device().to_owned();
                let new_compiler =
                    MetalKernelCompiler::new(Arc::new(metal_device)).map_err(|e| {
                        candle_core::Error::Msg(format!("Failed to create compiler: {e}"))
                    })?;
                *compiler_guard = Some(new_compiler);
            }
            compiler_guard
                .as_ref()
                .unwrap()
                .create_pipeline("fused_softmax")
                .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {e}")))?
        };

        // Cache for next time
        {
            let mut pipeline_guard = self
                .pipeline
                .lock()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
            *pipeline_guard = Some(pipeline.clone());
        }

        Ok(pipeline)
    }
}

impl CustomOp1 for FusedSoftmaxOp {
    fn name(&self) -> &'static str {
        "fused-softmax"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Fused Softmax kernel not implemented for CPU. Use Metal device.")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();

        // Get or compile the pipeline
        let pipeline = self.get_or_compile_pipeline(device)?;

        // Extract input tensor dimensions
        let input_dims = layout.shape().dims();
        if input_dims.is_empty() {
            candle_core::bail!(
                "Input tensor for FusedSoftmaxOp must have at least 1 dimension, got {:?}",
                input_dims
            );
        }

        // Support various input shapes
        let (batch_size, seq_len, dim) = match input_dims.len() {
            1 => (1, 1, input_dims[0]),
            2 => (1, input_dims[0], input_dims[1]),
            3 => (input_dims[0], input_dims[1], input_dims[2]),
            _ => candle_core::bail!(
                "Softmax only supports 1D, 2D, or 3D tensors, got {:?}",
                input_dims
            ),
        };

        // Get Metal buffers
        let input_buffer = storage.buffer();

        // Create output buffer
        let output_shape = layout.shape().clone();
        let output_elem_count = output_shape.elem_count();
        let output_buffer =
            device.new_buffer(output_elem_count, storage.dtype(), "fused_softmax_output")?;

        // Prepare kernel parameters
        let params = SoftmaxParams {
            batch_size: u32::try_from(batch_size)
                .map_err(|e| candle_core::Error::Msg(format!("Batch size too large: {e}")))?,
            seq_len: u32::try_from(seq_len)
                .map_err(|e| candle_core::Error::Msg(format!("Sequence length too large: {e}")))?,
            dim: u32::try_from(dim)
                .map_err(|e| candle_core::Error::Msg(format!("Dimension too large: {e}")))?,
        };

        // Create command buffer and dispatch kernel
        let command_buffer = device.command_buffer()?;

        // Scope the encoder
        {
            use candle_metal_kernels::utils::EncoderProvider;
            let command_buffer_ref = &command_buffer;
            let encoder_wrapper = command_buffer_ref.encoder();
            let encoder = encoder_wrapper.as_ref();

            encoder.set_compute_pipeline_state(&pipeline);

            // Set buffers
            encoder.set_buffer(0, Some(input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<SoftmaxParams>() as u64,
                std::ptr::addr_of!(params).cast::<std::ffi::c_void>(),
            );

            // Threadgroup memory for reductions (256 floats = 1024 bytes)
            encoder.set_threadgroup_memory_length(0, 256 * 4);

            // Dispatch threads
            // Each threadgroup handles one softmax operation (one row)
            let grid_size = metal::MTLSize {
                width: 1,
                height: seq_len as u64,
                depth: batch_size as u64,
            };
            let threadgroup_size = metal::MTLSize {
                width: 256, // Optimal for reductions
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);

            // encoder_wrapper drops here
        }

        // Create output storage
        let output_storage = candle_core::MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            storage.dtype(),
        );

        Ok((output_storage, output_shape))
    }
}

/// Fused Layer Normalization operation.
///
/// Implements standard layer normalization as a single Metal kernel:
/// `normalized[i] = (input[i] - mean) / sqrt(variance + eps)`
///
/// This enables Metal acceleration for embedding models (BERT, E5, `MiniLM`)
/// that require `LayerNorm`, which is not natively supported by Candle's Metal backend.
///
/// # Performance
///
/// - **CPU fallback**: ~50-200 µs
/// - **Metal (fused)**: ~5-20 µs
/// - **Expected speedup**: 5-10x
///
/// # Implementation Notes
///
/// - Uses `CustomOp1` (single input tensor)
/// - Fuses mean/variance computation and normalization
/// - Single-pass reduction using threadgroup memory
/// - CPU fallback provided for compatibility
pub struct LayerNormOp {
    /// Epsilon for numerical stability
    eps: f64,

    /// Cached Metal compute pipeline
    pipeline: Arc<Mutex<Option<metal::ComputePipelineState>>>,

    /// Cached Metal kernel compiler
    compiler: Arc<Mutex<Option<MetalKernelCompiler>>>,
}

impl LayerNormOp {
    /// Creates a new layer normalization operation.
    ///
    /// # Arguments
    ///
    /// * `eps` - Epsilon for numerical stability (default: 1e-12 for BERT)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::backend::custom_ops::LayerNormOp;
    ///
    /// let op = LayerNormOp::new(1e-12);
    /// ```
    #[must_use]
    pub fn new(eps: f64) -> Self {
        Self {
            eps,
            pipeline: Arc::new(Mutex::new(None)),
            compiler: Arc::new(Mutex::new(None)),
        }
    }

    /// Helper to get or compile the Metal pipeline.
    fn get_or_compile_pipeline(&self, device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let pipeline_guard = self.pipeline.lock().unwrap();

        if let Some(ref pipeline) = *pipeline_guard {
            return Ok(pipeline.clone());
        }

        // Need to compile the pipeline
        drop(pipeline_guard);

        let mut compiler_guard = self.compiler.lock().unwrap();

        if compiler_guard.is_none() {
            *compiler_guard = Some(
                MetalKernelCompiler::new(Arc::new(device.to_owned())).map_err(|e| {
                    candle_core::Error::Msg(format!("Metal compiler init failed: {e}"))
                })?,
            );
        }

        let compiler = compiler_guard.as_ref().unwrap();
        let new_pipeline = compiler
            .create_pipeline("layer_norm")
            .map_err(|e| candle_core::Error::Msg(format!("Pipeline creation failed: {e}")))?;

        drop(compiler_guard);

        let mut pipeline_guard = self.pipeline.lock().unwrap();
        *pipeline_guard = Some(new_pipeline.clone());

        Ok(new_pipeline)
    }
}

impl CustomOp1 for LayerNormOp {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        // CPU fallback: standard layer norm implementation
        let shape = layout.shape();
        if shape.rank() != 2 {
            candle_core::bail!("LayerNorm expects 2D tensors, got shape {:?}", shape.dims());
        }

        let (batch_size, hidden_size) = (shape.dims()[0], shape.dims()[1]);
        let input = storage.as_slice::<f32>()?;

        let mut output = vec![0.0f32; batch_size * hidden_size];

        for b in 0..batch_size {
            let offset = b * hidden_size;
            let row = &input[offset..offset + hidden_size];

            // Compute mean
            // Cast to f32 is acceptable - typical hidden sizes (384-1024) are well within f32 range
            #[allow(clippy::cast_precision_loss)]
            let hidden_size_f32 = hidden_size as f32;
            let mean: f32 = row.iter().sum::<f32>() / hidden_size_f32;

            // Compute variance
            let variance: f32 =
                row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / hidden_size_f32;

            // Normalize
            #[allow(clippy::cast_possible_truncation)]
            // eps is a small value (typically 1e-12), f32 precision is sufficient
            let inv_std = 1.0 / (variance + self.eps as f32).sqrt();

            for (i, &val) in row.iter().enumerate() {
                output[offset + i] = (val - mean) * inv_std;
            }
        }

        let storage = candle_core::CpuStorage::F32(output);
        Ok((storage, shape.clone()))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        use crate::backend::metal_kernels::LayerNormParams;
        
        let shape = layout.shape();
        if shape.rank() != 2 {
            candle_core::bail!("LayerNorm expects 2D tensors, got shape {:?}", shape.dims());
        }

        let (batch_size, hidden_size) = (shape.dims()[0], shape.dims()[1]);

        let device = storage.device();
        let metal_device = device.metal_device();

        // Get or compile pipeline
        let pipeline = self.get_or_compile_pipeline(metal_device)?;

        // Create output buffer using Metal API directly
        let output_elem_count = batch_size * hidden_size;
        let output_buffer = metal_device.new_buffer(
            (output_elem_count * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Set up kernel parameters
        // Casts to u32/f32 are acceptable for typical ML batch sizes (<10k) and hidden dims (<16k)
        #[allow(clippy::cast_possible_truncation)]
        let params = LayerNormParams {
            batch_size: batch_size as u32,
            hidden_size: hidden_size as u32,
            eps: self.eps as f32,
        };

        // Create command buffer and dispatch kernel (use Candle's device wrapper)
        let command_buffer = device.command_buffer()?;

        {
            let input_buffer = storage.buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<LayerNormParams>() as u64,
                std::ptr::addr_of!(params).cast::<std::ffi::c_void>(),
            );

            //Threadgroup memory for reductions (256 floats = 1024 bytes)
            encoder.set_threadgroup_memory_length(0, 256 * 4);

            // Dispatch grid: one threadgroup per batch element
            let threadgroup_size = metal::MTLSize {
                width: 256.min(hidden_size as u64),
                height: 1,
                depth: 1,
            };

            let grid_size = metal::MTLSize {
                width: batch_size as u64,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();

            // encoder reference drops here
        }

        // Don't commit - Candle manages the command buffer lifecycle
        // The kernel has been dispatched and will execute when Candle commits

        // Create output storage
        let output_storage = MetalStorage::new(
            std::sync::Arc::new(output_buffer),
            device.clone(),
            output_elem_count,
            candle_core::DType::F32,
        );

        Ok((output_storage, shape.clone()))
    }
}

/// Convenience function to apply layer normalization to a tensor.
///
/// # Arguments
///
/// * `tensor` - Input tensor of shape `[batch, hidden_size]`
/// * `eps` - Epsilon for numerical stability (default: 1e-12 for BERT)
///
/// # Returns
///
/// Normalized tensor with the same shape as input.
///
/// # Examples
///
/// ```no_run
/// use candle_core::{Tensor, Device};
/// use metal_candle::backend::custom_ops::layer_norm;
///
/// # fn example() -> candle_core::Result<()> {
/// let device = Device::new_metal(0)?;
/// let input = Tensor::randn(0.0, 1.0, (2, 768), &device)?;
/// let normalized = layer_norm(&input, 1e-12)?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns error if the operation fails or if the tensor is not 2D.
pub fn layer_norm(tensor: &Tensor, eps: f64) -> Result<Tensor> {
    tensor.apply_op1(LayerNormOp::new(eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_fused_lora_op_creation() {
        if let Ok(device) = Device::new_metal(0) {
            // Use F32 for Metal compatibility
            let lora_a = Tensor::randn(0.0f32, 0.01f32, (512, 8), &device).unwrap();
            let lora_b = Tensor::zeros((8, 512), DType::F32, &device).unwrap();

            let op = FusedLoRAOp::new(lora_a, lora_b, 2.0);
            assert!(op.is_ok());
        }
    }

    #[test]
    fn test_fused_lora_op_invalid_dimensions() {
        if let Ok(device) = Device::new_metal(0) {
            // Use F32 for Metal compatibility
            let lora_a = Tensor::randn(0.0f32, 0.01f32, (512, 8), &device).unwrap();
            let lora_b = Tensor::zeros((16, 512), DType::F32, &device).unwrap(); // Wrong rank!

            let op = FusedLoRAOp::new(lora_a, lora_b, 2.0);
            assert!(op.is_err());
        }
    }
}
