//! Error types for metal-candle.
//!
//! This module defines the error types used throughout the crate.
//! We use `thiserror` for ergonomic error handling in library code.

use thiserror::Error;

/// Result type alias for metal-candle operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The main error type for metal-candle operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Error related to model operations (loading, validation, etc.)
    #[error("model error: {0}")]
    Model(#[from] ModelError),

    /// Error related to training operations
    #[error("training error: {0}")]
    Training(#[from] TrainingError),

    /// Error related to inference/generation operations
    #[error("inference error: {0}")]
    Inference(#[from] InferenceError),

    /// Error related to checkpoint operations
    #[error("checkpoint error: {0}")]
    Checkpoint(#[from] CheckpointError),

    /// Error related to device/backend operations
    #[error("device error: {0}")]
    Device(#[from] DeviceError),

    /// IO errors
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Candle framework errors
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Errors related to model operations.
#[derive(Error, Debug)]
pub enum ModelError {
    /// Model file not found at the specified path
    #[error("model file not found: {path}")]
    FileNotFound {
        /// The path that was not found
        path: std::path::PathBuf,
    },

    /// Invalid model format or corrupted file
    #[error("invalid model format: {reason}")]
    InvalidFormat {
        /// Description of the format issue
        reason: String,
    },

    /// Model version incompatibility
    #[error("incompatible model version: expected {expected}, found {found}")]
    IncompatibleVersion {
        /// Expected version
        expected: String,
        /// Found version
        found: String,
    },

    /// Invalid model configuration
    #[error("invalid model configuration: {reason}")]
    InvalidConfig {
        /// Description of the configuration issue
        reason: String,
    },

    /// Tensor shape mismatch
    #[error("tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },
}

/// Errors related to training operations.
#[derive(Error, Debug)]
pub enum TrainingError {
    /// Invalid `LoRA` configuration
    #[error("invalid LoRA configuration: {reason}")]
    InvalidLoRAConfig {
        /// Description of the configuration issue
        reason: String,
    },

    /// Invalid training configuration
    #[error("invalid training configuration: {reason}")]
    InvalidConfig {
        /// Description of the configuration issue
        reason: String,
    },

    /// Training failed
    #[error("training failed: {reason}")]
    Failed {
        /// Description of the failure
        reason: String,
    },

    /// Gradient computation error
    #[error("gradient error: {reason}")]
    Gradient {
        /// Description of the gradient issue
        reason: String,
    },

    /// Training state error
    #[error("training state error: {reason}")]
    StateError {
        /// Description of the state issue
        reason: String,
    },
}

/// Errors related to inference operations.
#[derive(Error, Debug)]
pub enum InferenceError {
    /// Invalid generation configuration
    #[error("invalid generation configuration: {reason}")]
    InvalidConfig {
        /// Description of the configuration issue
        reason: String,
    },

    /// Generation failed
    #[error("generation failed: {reason}")]
    Failed {
        /// Description of the failure
        reason: String,
    },

    /// Invalid sampling parameters
    #[error("invalid sampling parameters: {reason}")]
    InvalidSampling {
        /// Description of the sampling issue
        reason: String,
    },

    /// KV-cache is full
    #[error("KV-cache full: position {position} exceeds max length {max_len}")]
    CacheFull {
        /// Current cache position
        position: usize,
        /// Maximum cache length
        max_len: usize,
    },

    /// Token sampling failed
    #[error("token sampling failed: {reason}")]
    SamplingError {
        /// Description of the sampling error
        reason: String,
    },

    /// End-of-sequence reached
    #[error("end of sequence reached at position {position}")]
    EndOfSequence {
        /// Position where EOS was reached
        position: usize,
    },
}

/// Errors related to checkpoint operations.
#[derive(Error, Debug)]
pub enum CheckpointError {
    /// Failed to save checkpoint
    #[error("failed to save checkpoint: {reason}")]
    SaveFailed {
        /// Description of the save failure
        reason: String,
    },

    /// Failed to load checkpoint
    #[error("failed to load checkpoint: {reason}")]
    LoadFailed {
        /// Description of the load failure
        reason: String,
    },

    /// Checkpoint format is invalid or corrupted
    #[error("invalid checkpoint format: {reason}")]
    InvalidFormat {
        /// Description of the format issue
        reason: String,
    },
}

/// Errors related to device/backend operations.
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Metal device not available
    #[error("Metal device not available: {reason}")]
    MetalUnavailable {
        /// Description of why Metal is unavailable
        reason: String,
    },

    /// Device initialization failed
    #[error("device initialization failed: {reason}")]
    InitializationFailed {
        /// Description of the initialization failure
        reason: String,
    },

    /// Memory allocation failed
    #[error("memory allocation failed: requested {requested_bytes} bytes")]
    AllocationFailed {
        /// Number of bytes requested
        requested_bytes: usize,
    },

    /// Tensor operation failed
    #[error("tensor operation failed: {operation}")]
    OperationFailed {
        /// Name of the failed operation
        operation: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::Model(ModelError::FileNotFound {
            path: std::path::PathBuf::from("/path/to/model.safetensors"),
        });
        assert!(err.to_string().contains("model file not found"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn test_model_error_types() {
        let err = ModelError::ShapeMismatch {
            expected: vec![1, 2, 3],
            actual: vec![1, 2, 4],
        };
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn test_training_error_types() {
        let err = TrainingError::InvalidLoRAConfig {
            reason: "rank must be > 0".to_string(),
        };
        assert!(err.to_string().contains("invalid LoRA configuration"));

        let err = TrainingError::InvalidConfig {
            reason: "invalid".to_string(),
        };
        assert!(err.to_string().contains("invalid training configuration"));

        let err = TrainingError::StateError {
            reason: "state error".to_string(),
        };
        assert!(err.to_string().contains("training state error"));
    }

    #[test]
    fn test_device_error_types() {
        let err = DeviceError::MetalUnavailable {
            reason: "not running on Apple Silicon".to_string(),
        };
        assert!(err.to_string().contains("Metal device not available"));
    }
}
