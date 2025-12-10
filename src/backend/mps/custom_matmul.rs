//! MPS matrix multiplication as a Candle CustomOp.

use super::{tensor_to_mps_matrix, MPSMatrixMultiplication};
use crate::error::{Result, TrainingError};
use candle_core::{CustomOp2, DType, Layout, MetalStorage, Result as CandleResult, Shape, Tensor};
use metal::foreign_types::ForeignTypeRef;
use std::sync::{OnceLock, Mutex};
use std::collections::HashMap;

/// Global command queue pool for MPS operations.
/// 
/// Creating a Metal command queue is expensive (~300µs), so we create one
/// per device and reuse it across all MPS operations.
static MPS_COMMAND_QUEUE: OnceLock<metal::CommandQueue> = OnceLock::new();

/// Cache for MPSMatrixMultiplication objects by dimensions (m, n, k).
/// 
/// Creating MPSMatrixMultiplication is expensive (~30-60µs), so we cache
/// them by their dimensions and reuse across operations.
static MPS_MATMUL_CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), MPSMatrixMultiplication>>> = OnceLock::new();

/// Custom operation for MPS matrix multiplication.
pub struct MPSMatMulOp;

impl CustomOp2 for MPSMatMulOp {
    fn name(&self) -> &'static str {
        "mps-matmul"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &Layout,
    ) -> CandleResult<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("MPS matmul not implemented for CPU. Use Metal device.")
    }

    fn metal_fwd(
        &self,
        left_storage: &MetalStorage,
        left_layout: &Layout,
        right_storage: &MetalStorage,
        right_layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        
        // Validate contiguity - MPS requires row-major, contiguous tensors
        // If not contiguous, caller must call .contiguous() first
        if !left_layout.is_contiguous() {
            candle_core::bail!("MPS matmul requires left tensor to be contiguous. Call .contiguous() first.");
        }
        if !right_layout.is_contiguous() {
            candle_core::bail!("MPS matmul requires right tensor to be contiguous. Call .contiguous() first.");
        }
        if left_layout.start_offset() != 0 {
            candle_core::bail!("MPS matmul requires left tensor start_offset to be 0");
        }
        if right_layout.start_offset() != 0 {
            candle_core::bail!("MPS matmul requires right tensor start_offset to be 0");
        }
        
        let device = left_storage.device();
        
        // Get dimensions
        let left_shape = left_layout.shape();
        let right_shape = right_layout.shape();
        
        if left_shape.rank() != 2 || right_shape.rank() != 2 {
            candle_core::bail!("MPS matmul requires 2D tensors");
        }
        
        let m = left_shape.dims()[0];
        let k = left_shape.dims()[1];
        let n = right_shape.dims()[1];
        
        if right_shape.dims()[0] != k {
            candle_core::bail!(
                "Matrix multiplication dimension mismatch: ({}, {}) @ ({}, {})",
                m, k, right_shape.dims()[0], n
            );
        }
        
        // Now safe to use buffers directly (validated as contiguous)
        let left_buffer = left_storage.buffer();
        let right_buffer = right_storage.buffer();
        
        // MPS expects row_bytes to be the number of BYTES per row, not elements
        // For row-major layout: row_bytes = columns * sizeof(element)
        let element_size = super::ffi::MPSDataType::Float32.size_in_bytes();
        
        // Create MPS descriptors with correct row_bytes
        let left_desc = super::ffi::MPSMatrixDescriptor::new(
            m,                  // rows
            k,                  // columns
            k * element_size,   // row_bytes (bytes per row)
            super::ffi::MPSDataType::Float32,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Left descriptor creation failed: {e}")))?;
        
        let right_desc = super::ffi::MPSMatrixDescriptor::new(
            k,                  // rows
            n,                  // columns  
            n * element_size,   // row_bytes (bytes per row)
            super::ffi::MPSDataType::Float32,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Right descriptor creation failed: {e}")))?;
        
        // Create MPS matrices
        let mps_left = super::matrix::MPSMatrix::new(left_buffer, &left_desc)
            .map_err(|e| candle_core::Error::Msg(format!("MPS left creation failed: {e}")))?;
        let mps_right = super::matrix::MPSMatrix::new(right_buffer, &right_desc)
            .map_err(|e| candle_core::Error::Msg(format!("MPS right creation failed: {e}")))?;
        
        // Create output buffer
        let output_elem_count = m * n;
        let output_buffer = device.new_buffer(
            output_elem_count,
            DType::F32,
            "mps_matmul_output",
        )?;
        
        // Create MPS descriptor for output
        let output_desc = super::ffi::MPSMatrixDescriptor::new(
            m,
            n,
            n * super::ffi::MPSDataType::Float32.size_in_bytes(),
            super::ffi::MPSDataType::Float32,
        )
        .map_err(|e| candle_core::Error::Msg(format!("MPS descriptor creation failed: {e}")))?;
        
        // Wrap output buffer in MPS matrix
        let mps_output = super::matrix::MPSMatrix::new(&output_buffer, &output_desc)
            .map_err(|e| candle_core::Error::Msg(format!("MPS matrix creation failed: {e}")))?;
        
        // Get or create cached MPSMatrixMultiplication kernel
        let cache = MPS_MATMUL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache_guard = cache.lock().unwrap();
        
        // Check if we have a cached matmul for these dimensions
        if !cache_guard.contains_key(&(m, n, k)) {
            let matmul = MPSMatrixMultiplication::new(
                device,
                false, // transpose_left
                false, // transpose_right
                m,     // result_rows
                n,     // result_columns
                k,     // interior_columns
                1.0,   // alpha
                0.0,   // beta
            )
            .map_err(|e| candle_core::Error::Msg(format!("MPS matmul creation failed: {e}")))?;
            
            cache_guard.insert((m, n, k), matmul);
        }
        
        let matmul = cache_guard.get(&(m, n, k)).unwrap();
        
        // Use pooled command queue for MPS operations
        // This avoids the ~300µs overhead of creating a new queue each time
        let metal_device = device.device();
        
        let queue = MPS_COMMAND_QUEUE.get_or_init(|| {
            metal_device.new_command_queue()
        });
        
        let mps_cmd_buffer = queue.new_command_buffer();
        
        // Encode MPS matmul to the command buffer
        let cmd_ptr = mps_cmd_buffer.as_ptr() as *mut objc::runtime::Object;
        matmul.encode(cmd_ptr, &mps_left, &mps_right, &mps_output);
        
        // Release cache lock before committing
        drop(cache_guard);
        
        // Execute and wait for completion
        // Note: Async execution would be faster but breaks correctness
        // CustomOp2's synchronous API requires the result to be ready
        mps_cmd_buffer.commit();
        mps_cmd_buffer.wait_until_completed();
        
        // Create output storage
        let output_storage = MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            DType::F32,
        );
        
        let output_shape = Shape::from_dims(&[m, n]);
        
        Ok((output_storage, output_shape))
    }
}

/// Perform matrix multiplication using MPS.
///
/// This is a thin wrapper around the MPS CustomOp.
///
/// # Errors
///
/// Returns error if operation fails.
pub fn mps_matmul(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    left.apply_op2(right, MPSMatMulOp)
        .map_err(|e| TrainingError::Failed {
            reason: format!("MPS matmul failed: {e}"),
        }.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mps_matmul_basic() {
        if let Ok(device) = Device::new_metal(0) {
            let candle_device = device.as_candle_device();

            // Create simple test matrices
            let a = Tensor::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0],
                &[2, 2],
                candle_device,
            )
            .unwrap();
            let b = Tensor::from_slice(
                &[5.0f32, 6.0, 7.0, 8.0],
                &[2, 2],
                candle_device,
            )
            .unwrap();

            // MPS matmul
            let result = mps_matmul(&a, &b);
            assert!(result.is_ok(), "MPS matmul should succeed");

            let result = result.unwrap();
            assert_eq!(result.dims(), &[2, 2]);
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mps_matmul_correctness() {
        if let Ok(device) = Device::new_metal(0) {
            let candle_device = device.as_candle_device();

            // Create test matrices
            let a = Tensor::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[2, 3],
                candle_device,
            )
            .unwrap();
            let b = Tensor::from_slice(
                &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
                &[3, 2],
                candle_device,
            )
            .unwrap();

            // Expected result: [1*7+2*9+3*11, 1*8+2*10+3*12]
            //                  [4*7+5*9+6*11, 4*8+5*10+6*12]
            //                = [58, 64]
            //                  [139, 154]

            // MPS result
            let mps_result = mps_matmul(&a, &b).unwrap();

            // Candle reference
            let candle_result = a.matmul(&b).unwrap();


            // Compare
            let diff = (mps_result - candle_result).unwrap().abs().unwrap();
            // Get max value across all elements
            let max_diff_scalar = diff.flatten_all().unwrap().max(0).unwrap().to_vec0::<f32>().unwrap();

            assert!(
                max_diff_scalar < 1e-5,
                "MPS result should match Candle within 1e-5, got diff: {}",
                max_diff_scalar
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mps_matmul_dimension_mismatch() {
        if let Ok(device) = Device::new_metal(0) {
            let candle_device = device.as_candle_device();

            let a = Tensor::zeros((2, 3), DType::F32, candle_device).unwrap();
            let b = Tensor::zeros((4, 5), DType::F32, candle_device).unwrap();

            let result = mps_matmul(&a, &b);
            assert!(result.is_err(), "Should error on dimension mismatch");
        }
    }
}

