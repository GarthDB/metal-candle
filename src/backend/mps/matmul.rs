//! MPS matrix multiplication implementation.

use super::matrix::MPSMatrix;
use crate::error::Result;
use metal;
use metal::foreign_types::ForeignTypeRef;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

/// Wrapper for MPSMatrixMultiplication.
///
/// Performs highly optimized matrix multiplication on Apple Silicon GPUs.
///
/// # Operation
///
/// Computes: `result = alpha * (left × right) + beta * result`
///
/// # Performance
///
/// Expected to be 5-10x faster than custom Metal kernels, matching or exceeding MLX.
pub struct MPSMatrixMultiplication {
    inner: *mut Object,
}

unsafe impl Send for MPSMatrixMultiplication {}
unsafe impl Sync for MPSMatrixMultiplication {}

impl MPSMatrixMultiplication {
    /// Create a new matrix multiplication operation.
    ///
    /// # Arguments
    ///
    /// * `device` - Metal device to use
    /// * `transpose_left` - Whether to transpose the left matrix
    /// * `transpose_right` - Whether to transpose the right matrix
    /// * `result_rows` - Number of rows in result (M dimension)
    /// * `result_columns` - Number of columns in result (N dimension)
    /// * `interior_columns` - Interior dimension (K), must match for A×B
    /// * `alpha` - Scaling factor for the product
    /// * `beta` - Scaling factor for result accumulation (usually 0.0)
    ///
    /// # Errors
    ///
    /// Returns error if operation creation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &candle_core::MetalDevice,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        unsafe {
            let class = class!(MPSMatrixMultiplication);
            let kernel: *mut Object = msg_send![class, alloc];
            let kernel: *mut Object = msg_send![
                kernel,
                initWithDevice: device.as_ptr()
                transposeLeft: transpose_left
                transposeRight: transpose_right
                resultRows: result_rows as u64
                resultColumns: result_columns as u64
                interiorColumns: interior_columns as u64
                alpha: alpha
                beta: beta
            ];

            if kernel.is_null() {
                return Err(crate::error::TrainingError::Failed {
                    reason: "Failed to create MPSMatrixMultiplication".to_string(),
                }.into());
            }

            // Retain for ownership
            let _: () = msg_send![kernel, retain];

            Ok(Self { inner: kernel })
        }
    }

    /// Encode the matrix multiplication to a command buffer.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - Command buffer to encode to (as raw pointer)
    /// * `left_matrix` - Left matrix (A)
    /// * `right_matrix` - Right matrix (B)
    /// * `result_matrix` - Result matrix (C)
    ///
    /// # Panics
    ///
    /// May panic if matrix dimensions are incompatible (should be validated before calling).
    pub fn encode(
        &self,
        command_buffer: *mut Object,
        left_matrix: &MPSMatrix,
        right_matrix: &MPSMatrix,
        result_matrix: &MPSMatrix,
    ) {
        unsafe {
            let _: () = msg_send![
                self.inner,
                encodeToCommandBuffer: command_buffer
                leftMatrix: left_matrix.as_ptr()
                rightMatrix: right_matrix.as_ptr()
                resultMatrix: result_matrix.as_ptr()
            ];
        }
    }
}

impl Drop for MPSMatrixMultiplication {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, release];
        }
    }
}

impl std::fmt::Debug for MPSMatrixMultiplication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MPSMatrixMultiplication").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mps_matmul_creation() {
        if let Ok(device) = Device::new_metal(0) {
            // Get Candle's MetalDevice
            match device.as_candle_device() {
                candle_core::Device::Metal(metal_dev) => {
                    let matmul = MPSMatrixMultiplication::new(
                        metal_dev,
                        false,
                        false,
                        512,
                        512,
                        256,
                        1.0,
                        0.0,
                    );
                    assert!(matmul.is_ok());
                }
                _ => panic!("Expected Metal device"),
            }
        }
    }
}

