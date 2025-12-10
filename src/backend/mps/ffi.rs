//! Low-level FFI bindings to Metal Performance Shaders.
//!
//! This module provides type-safe wrappers around MPS Objective-C APIs.
//! All memory management (retain/release) is handled automatically via Drop.

use crate::error::Result;
use metal;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

/// MPS data type enumeration.
///
/// Corresponds to MPSDataType in Metal Performance Shaders.
/// Values from MPSDataType.h: 0x10000000 | size_in_bytes
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MPSDataType {
    /// 16-bit floating point (0x10000010)
    Float16 = 268435472,
    /// 32-bit floating point (0x10000020) - our primary type
    Float32 = 268435488,
    /// 32-bit signed integer (0x20000020)
    Int32 = 536870944,
}

impl MPSDataType {
    /// Get the size in bytes for this data type.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 => 2,
        }
    }
}

/// Wrapper for MPSMatrixDescriptor.
///
/// Describes the shape and data type of a matrix for MPS operations.
pub struct MPSMatrixDescriptor {
    inner: *mut Object,
}

unsafe impl Send for MPSMatrixDescriptor {}
unsafe impl Sync for MPSMatrixDescriptor {}

impl MPSMatrixDescriptor {
    /// Create a new matrix descriptor.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `columns` - Number of columns
    /// * `row_bytes` - Number of bytes per row (usually columns * data_type.size_in_bytes())
    /// * `data_type` - Data type of matrix elements
    ///
    /// # Errors
    ///
    /// Returns error if descriptor creation fails.
    pub fn new(
        rows: usize,
        columns: usize,
        row_bytes: usize,
        data_type: MPSDataType,
    ) -> Result<Self> {
        unsafe {
            let class = class!(MPSMatrixDescriptor);
            let descriptor: *mut Object = msg_send![
                class,
                matrixDescriptorWithRows: rows as u64
                columns: columns as u64
                rowBytes: row_bytes as u64
                dataType: data_type as u64
            ];

            if descriptor.is_null() {
                return Err(crate::error::TrainingError::Failed {
                    reason: "Failed to create MPSMatrixDescriptor".to_string(),
                }.into());
            }

            // Retain for ownership
            let _: () = msg_send![descriptor, retain];

            Ok(Self { inner: descriptor })
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

    /// Get the row bytes.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        unsafe {
            let row_bytes: u64 = msg_send![self.inner, rowBytes];
            row_bytes as usize
        }
    }

    /// Get the raw pointer (for FFI use).
    #[must_use]
    pub(crate) fn as_ptr(&self) -> *mut Object {
        self.inner
    }
}

impl Drop for MPSMatrixDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, release];
        }
    }
}

impl Clone for MPSMatrixDescriptor {
    fn clone(&self) -> Self {
        unsafe {
            let _: () = msg_send![self.inner, retain];
            Self { inner: self.inner }
        }
    }
}

impl std::fmt::Debug for MPSMatrixDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MPSMatrixDescriptor")
            .field("rows", &self.rows())
            .field("columns", &self.columns())
            .field("row_bytes", &self.row_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_data_type_size() {
        assert_eq!(MPSDataType::Float32.size_in_bytes(), 4);
        assert_eq!(MPSDataType::Float16.size_in_bytes(), 2);
        assert_eq!(MPSDataType::Int32.size_in_bytes(), 4);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mps_descriptor_creation() {
        let descriptor = MPSMatrixDescriptor::new(512, 512, 512 * 4, MPSDataType::Float32);
        assert!(descriptor.is_ok());

        let descriptor = descriptor.unwrap();
        assert_eq!(descriptor.rows(), 512);
        assert_eq!(descriptor.columns(), 512);
        assert_eq!(descriptor.row_bytes(), 512 * 4);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_mps_descriptor_clone() {
        let descriptor =
            MPSMatrixDescriptor::new(256, 256, 256 * 4, MPSDataType::Float32).unwrap();
        let cloned = descriptor.clone();

        assert_eq!(cloned.rows(), 256);
        assert_eq!(cloned.columns(), 256);
    }
}

