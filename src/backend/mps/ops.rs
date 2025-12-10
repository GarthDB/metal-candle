//! High-level MPS operations.
//!
//! This module provides end-to-end operations using Metal Performance Shaders,
//! handling tensor conversion, MPS dispatch, and result extraction.

use super::{tensor_to_mps_matrix, MPSMatrixMultiplication};
use crate::error::{Result, TrainingError};
use candle_core::{DType, Storage, Tensor};
use metal::foreign_types::ForeignTypeRef;

/// Perform matrix multiplication using MPS.
///
/// Computes `output = left @ right` using Apple's optimized Metal Performance Shaders.
///
/// # Performance
///
/// Expected to be 5-10x faster than custom Metal kernels and competitive with MLX.
///
/// # Requirements
///
/// - Both tensors must be on Metal device
/// - Both tensors must be 2D
/// - Both tensors must be F32 dtype
/// - Both tensors must be contiguous
/// - Inner dimensions must match: `(M, K) @ (K, N) = (M, N)`
///
/// # Examples
///
/// ```no_run
/// use metal_candle::backend::mps::mps_matmul;
/// use candle_core::{Device, Tensor, DType};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::new_metal(0)?;
/// let a = Tensor::randn(0.0f32, 1.0, (512, 256), &device)?;
/// let b = Tensor::randn(0.0f32, 1.0, (256, 512), &device)?;
///
/// let result = mps_matmul(&a, &b)?;
/// assert_eq!(result.dims(), &[512, 512]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns error if:
/// - Tensors don't meet MPS requirements
/// - Dimension mismatch (K dimensions don't match)
/// - MPS operation fails
pub fn mps_matmul(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    // Validate inputs
    super::validate_tensor_for_mps(left)?;
    super::validate_tensor_for_mps(right)?;

    let left_dims = left.dims();
    let right_dims = right.dims();

    // Validate shapes: (M, K) @ (K, N) = (M, N)
    if left_dims[1] != right_dims[0] {
        return Err(TrainingError::Failed {
            reason: format!(
                "Matrix multiplication dimension mismatch: ({}, {}) @ ({}, {})",
                left_dims[0], left_dims[1], right_dims[0], right_dims[1]
            ),
        }
        .into());
    }

    let m = left_dims[0];
    let k = left_dims[1];
    let n = right_dims[1];

    // Get Metal device from Candle device
    let candle_device = left.device();
    let metal_device = match candle_device {
        candle_core::Device::Metal(metal_dev) => metal_dev,
        _ => {
            return Err(TrainingError::Failed {
                reason: "Tensor must be on Metal device".to_string(),
            }
            .into())
        }
    };

    // Convert input tensors to MPS matrices
    let mps_left = tensor_to_mps_matrix(left)?;
    let mps_right = tensor_to_mps_matrix(right)?;

    // Create output buffer FIRST (before wrapping in tensor)
    // This avoids lifetime issues with Rust borrows
    let output_elem_count = m * n;
    let output_buffer = metal_device.new_buffer(
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
    )?;

    // Wrap output buffer in MPS matrix
    let mps_output = super::matrix::MPSMatrix::new(
        &output_buffer,
        &output_desc,
    )?;

    // Create MPS matrix multiplication kernel
    // MetalDevice derefs to metal::DeviceRef, we need the owned Device
    let matmul = MPSMatrixMultiplication::new(
        metal_device,
        false, // transpose_left
        false, // transpose_right
        m,     // result_rows
        n,     // result_columns
        k,     // interior_columns
        1.0,   // alpha
        0.0,   // beta
    )?;

    // Create command buffer and encode operation
    let command_buffer = metal_device.command_buffer()?;

    // Encode MPS matmul to command buffer
    matmul.encode(command_buffer.as_ref(), &mps_left, &mps_right, &mps_output);

    // Execute and wait for completion
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Now wrap the filled buffer in a Candle tensor
    // The buffer has been filled by MPS, we can safely create the tensor
    let output_storage = candle_core::MetalStorage::new(
        output_buffer,
        metal_device.clone(),
        output_elem_count,
        DType::F32,
    );

    // Use unsafe to create tensor from raw parts (like CustomOp does)
    let output_tensor = unsafe {
        Tensor::from_storage(
            candle_core::Storage::Metal(output_storage),
            candle_core::Shape::from_dims(&[m, n]),
            false, // is_variable (not for training)
        )
    };

    Ok(output_tensor)
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
            let max_diff = diff.max(0).unwrap().max(1).unwrap();
            let max_diff_scalar = max_diff.to_vec0::<f32>().unwrap();

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

