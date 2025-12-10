//! MPSMatrix wrapper and conversion utilities.

use super::ffi::{MPSDataType, MPSMatrixDescriptor};
use crate::error::Result;
use candle_core::{DType, Storage, Tensor};
use metal;
use metal::foreign_types::ForeignTypeRef;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

/// Wrapper for MPSMatrix.
///
/// An MPSMatrix wraps a Metal buffer with shape and type metadata for use
/// in MPS operations.
pub struct MPSMatrix {
    inner: *mut Object,
}

unsafe impl Send for MPSMatrix {}
unsafe impl Sync for MPSMatrix {}

impl MPSMatrix {
    /// Create a new MPSMatrix from a Metal buffer and descriptor.
    ///
    /// # Errors
    ///
    /// Returns error if matrix creation fails.
    pub fn new(buffer: &metal::Buffer, descriptor: &MPSMatrixDescriptor) -> Result<Self> {
        unsafe {
            let class = class!(MPSMatrix);
            let matrix: *mut Object = msg_send![class, alloc];
            let matrix: *mut Object = msg_send![
                matrix,
                initWithBuffer: buffer.as_ptr()
                descriptor: descriptor.as_ptr()
            ];

            if matrix.is_null() {
                return Err(crate::error::TrainingError::Failed {
                    reason: "Failed to create MPSMatrix".to_string(),
                }.into());
            }

            // Retain for ownership
            let _: () = msg_send![matrix, retain];

            Ok(Self { inner: matrix })
        }
    }

    /// Get the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        unsafe {
            let rows: u64 = msg_send![self.inner, rows];
            rows as usize
        }
    }

    /// Get the number of columns.
    #[must_use]
    pub fn columns(&self) -> usize {
        unsafe {
            let columns: u64 = msg_send![self.inner, columns];
            columns as usize
        }
    }

    /// Get the raw pointer (for FFI use).
    #[must_use]
    pub(crate) fn as_ptr(&self) -> *mut Object {
        self.inner
    }
}

impl Drop for MPSMatrix {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, release];
        }
    }
}

impl std::fmt::Debug for MPSMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MPSMatrix")
            .field("rows", &self.rows())
            .field("columns", &self.columns())
            .finish()
    }
}

/// Validate that a tensor meets MPS requirements.
///
/// # Errors
///
/// Returns error if tensor doesn't meet requirements.
pub fn validate_tensor_for_mps(tensor: &Tensor) -> Result<()> {
    // Check device type
    if !matches!(tensor.device(), candle_core::Device::Metal(_)) {
        return Err(crate::error::TrainingError::Failed {
            reason: format!(
                "Tensor must be on Metal device, got {:?}",
                tensor.device()
            ),
        }.into());
    }

    // Check contiguous
    let layout = tensor.layout();
    if !layout.is_contiguous() {
        return Err(crate::error::TrainingError::Failed {
            reason: "Tensor must be contiguous for MPS operations. Call .contiguous() first."
                .to_string(),
        }.into());
    }

    // Check 2D
    if tensor.dims().len() != 2 {
        return Err(crate::error::TrainingError::Failed {
            reason: format!(
                "MPS operations currently only support 2D tensors, got {:?}D",
                tensor.dims().len()
            ),
        }.into());
    }

    // Check dtype
    if tensor.dtype() != DType::F32 {
        return Err(crate::error::TrainingError::Failed {
            reason: format!(
                "MPS operations currently only support F32, got {:?}",
                tensor.dtype()
            ),
        }.into());
    }

    Ok(())
}

/// Convert a Candle tensor to an MPSMatrix.
///
/// # Errors
///
/// Returns error if tensor doesn't meet MPS requirements or conversion fails.
pub fn tensor_to_mps_matrix(tensor: &Tensor) -> Result<MPSMatrix> {
    // Validate tensor
    validate_tensor_for_mps(tensor)?;

    // Extract Metal buffer
    let storage_guard = tensor.storage_and_layout();
    let Storage::Metal(metal_storage) = &*storage_guard.0 else {
        return Err(crate::error::TrainingError::Failed {
            reason: "Tensor must have Metal storage".to_string(),
        }.into());
    };

    let buffer = metal_storage.buffer();
    let dims = tensor.dims();

    // Create descriptor
    let rows = dims[0];
    let columns = dims[1];
    let row_bytes = columns * MPSDataType::Float32.size_in_bytes();

    let descriptor = MPSMatrixDescriptor::new(rows, columns, row_bytes, MPSDataType::Float32)?;

    // Create matrix
    MPSMatrix::new(buffer, &descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;

    #[test]
    fn test_validate_tensor_cpu_fails() {
        let device = candle_core::Device::Cpu;
        let tensor = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        assert!(validate_tensor_for_mps(&tensor).is_err());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_validate_tensor_metal_passes() {
        if let Ok(device) = Device::new_metal(0) {
            let candle_device = device.as_candle_device();
            let tensor = Tensor::zeros((2, 2), DType::F32, candle_device).unwrap();
            assert!(validate_tensor_for_mps(&tensor).is_ok());
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_validate_tensor_3d_fails() {
        if let Ok(device) = Device::new_metal(0) {
            let candle_device = device.as_candle_device();
            let tensor = Tensor::zeros((2, 2, 2), DType::F32, candle_device).unwrap();
            assert!(validate_tensor_for_mps(&tensor).is_err());
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_tensor_to_mps_matrix() {
        if let Ok(device) = Device::new_metal(0) {
            let candle_device = device.as_candle_device();
            let tensor = Tensor::zeros((512, 256), DType::F32, candle_device).unwrap();

            let mps_matrix = tensor_to_mps_matrix(&tensor);
            assert!(mps_matrix.is_ok());

            let mps_matrix = mps_matrix.unwrap();
            assert_eq!(mps_matrix.rows(), 512);
            assert_eq!(mps_matrix.columns(), 256);
        }
    }
}

