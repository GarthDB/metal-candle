//! Model loading from safetensors format.
//!
//! This module provides utilities for loading ML model weights from safetensors files,
//! with validation and proper error handling.

use crate::backend::Device;
use crate::error::{ModelError, Result};
use candle_core::{DType, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// A loader for ML models from safetensors format.
///
/// `ModelLoader` handles loading model weights from safetensors files with validation,
/// shape checking, and conversion to the appropriate device and dtype.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::{Device, models::ModelLoader};
///
/// let device = Device::new_with_fallback(0);
/// let loader = ModelLoader::new(device);
/// let tensors = loader.load("model.safetensors")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ModelLoader {
    device: Device,
    dtype: Option<DType>,
}

impl ModelLoader {
    /// Creates a new model loader for the specified device.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::{Device, models::ModelLoader};
    ///
    /// let device = Device::new_cpu();
    /// let loader = ModelLoader::new(device);
    /// ```
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            device,
            dtype: None,
        }
    }

    /// Sets the target dtype for loaded tensors.
    ///
    /// If specified, tensors will be converted to this dtype after loading.
    /// If not specified, tensors keep their original dtype from the file.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::{Device, models::ModelLoader};
    /// use candle_core::DType;
    ///
    /// let device = Device::new_cpu();
    /// let loader = ModelLoader::new(device)
    ///     .with_dtype(DType::F16);
    /// ```
    #[must_use]
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Loads model weights from a safetensors file.
    ///
    /// Returns a map of tensor names to tensors loaded on the specified device.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::{Device, models::ModelLoader};
    ///
    /// let device = Device::new_with_fallback(0);
    /// let loader = ModelLoader::new(device);
    /// let tensors = loader.load("model.safetensors")?;
    ///
    /// // Access specific weights
    /// if let Some(embeddings) = tensors.get("embed_tokens.weight") {
    ///     println!("Embeddings shape: {:?}", embeddings.shape());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::FileNotFound`] if the file doesn't exist.
    /// Returns [`ModelError::InvalidFormat`] if:
    /// - The file is not a valid safetensors file
    /// - The file is corrupted
    /// - Tensor shapes or types are invalid
    pub fn load(&self, path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
        let path = path.as_ref();

        // Check if file exists
        if !path.exists() {
            return Err(ModelError::FileNotFound {
                path: path.to_path_buf(),
            }
            .into());
        }

        // Load safetensors file
        let tensors = self.load_safetensors(path)?;

        Ok(tensors)
    }

    /// Loads a safetensors file and returns the tensors.
    fn load_safetensors(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        // Use candle's built-in safetensors support
        let tensors = candle_core::safetensors::load(path, self.device.as_ref()).map_err(|e| {
            ModelError::InvalidFormat {
                reason: format!("Failed to load safetensors file: {e}"),
            }
        })?;

        // Convert dtype if specified
        let tensors = if let Some(target_dtype) = self.dtype {
            tensors
                .into_iter()
                .map(|(name, tensor)| {
                    let converted = if tensor.dtype() == target_dtype {
                        tensor
                    } else {
                        tensor
                            .to_dtype(target_dtype)
                            .map_err(|e| ModelError::InvalidFormat {
                                reason: format!(
                                    "Failed to convert tensor '{name}' to {target_dtype:?}: {e}"
                                ),
                            })?
                    };
                    Ok((name, converted))
                })
                .collect::<Result<HashMap<_, _>>>()?
        } else {
            tensors
        };

        Ok(tensors)
    }

    /// Loads model weights and validates against expected tensor names and shapes.
    ///
    /// This is useful when you know the expected structure of the model.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file
    /// * `expected` - Map of expected tensor names to their expected shapes
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::{Device, models::ModelLoader};
    /// use std::collections::HashMap;
    ///
    /// let device = Device::new_with_fallback(0);
    /// let loader = ModelLoader::new(device);
    ///
    /// let mut expected = HashMap::new();
    /// expected.insert("embed_tokens.weight".to_string(), vec![32000, 768]);
    ///
    /// let tensors = loader.load_with_validation("model.safetensors", &expected)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if tensor shapes don't match expectations.
    /// Returns [`ModelError::InvalidFormat`] if expected tensors are missing.
    pub fn load_with_validation(
        &self,
        path: impl AsRef<Path>,
        expected: &HashMap<String, Vec<usize>>,
    ) -> Result<HashMap<String, Tensor>> {
        let tensors = self.load(path)?;

        // Validate all expected tensors are present
        for (name, expected_shape) in expected {
            match tensors.get(name) {
                Some(tensor) => {
                    let actual_shape: Vec<usize> = tensor.shape().dims().to_vec();
                    if &actual_shape != expected_shape {
                        return Err(ModelError::ShapeMismatch {
                            expected: expected_shape.clone(),
                            actual: actual_shape,
                        }
                        .into());
                    }
                }
                None => {
                    return Err(ModelError::InvalidFormat {
                        reason: format!("Expected tensor '{name}' not found in model file"),
                    }
                    .into());
                }
            }
        }

        Ok(tensors)
    }

    /// Returns information about tensors in a safetensors file without loading them.
    ///
    /// This is useful for inspecting model structure without loading all weights into memory.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::{Device, models::ModelLoader};
    ///
    /// let device = Device::new_cpu();
    /// let loader = ModelLoader::new(device);
    /// let info = loader.inspect("model.safetensors")?;
    ///
    /// for (name, shape) in &info {
    ///     println!("{}: {:?}", name, shape);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::FileNotFound`] if the file doesn't exist.
    /// Returns [`ModelError::InvalidFormat`] if the file is not a valid safetensors file.
    pub fn inspect(&self, path: impl AsRef<Path>) -> Result<HashMap<String, Vec<usize>>> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ModelError::FileNotFound {
                path: path.to_path_buf(),
            }
            .into());
        }

        // Read safetensors metadata without loading tensors
        // We read the file into memory since we only need the header (small)
        let buffer = std::fs::read(path)?;

        let tensors = safetensors::SafeTensors::deserialize(&buffer).map_err(|e| {
            ModelError::InvalidFormat {
                reason: format!("Failed to parse safetensors metadata: {e}"),
            }
        })?;

        let info = tensors
            .tensors()
            .iter()
            .map(|(name, view)| (name.clone(), view.shape().to_vec()))
            .collect();

        Ok(info)
    }

    /// Returns the device this loader will place tensors on.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the target dtype if set.
    #[must_use]
    pub const fn dtype(&self) -> Option<DType> {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_loader_creation() {
        let device = Device::new_cpu();
        let loader = ModelLoader::new(device);
        assert_eq!(loader.dtype(), None);
    }

    #[test]
    fn test_loader_with_dtype() {
        let device = Device::new_cpu();
        let loader = ModelLoader::new(device).with_dtype(DType::F16);
        assert_eq!(loader.dtype(), Some(DType::F16));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let device = Device::new_cpu();
        let loader = ModelLoader::new(device);
        let result = loader.load("nonexistent.safetensors");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::Error::Model(ModelError::FileNotFound { .. })
        ));
    }

    #[test]
    fn test_inspect_nonexistent_file() {
        let device = Device::new_cpu();
        let loader = ModelLoader::new(device);
        let result = loader.inspect("nonexistent.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_loader_device_accessor() {
        let device = Device::new_cpu();
        let loader = ModelLoader::new(device.clone());
        assert!(loader.device().is_cpu());
    }

    // Integration test for actual file loading would go in tests/models/loading.rs
    // with test fixtures
}
