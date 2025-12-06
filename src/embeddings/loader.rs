//! Model loading and caching utilities.
//!
//! This module handles downloading embedding models from `HuggingFace` Hub
//! and caching them locally for fast subsequent loads.

use std::path::PathBuf;

use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::Config as BertConfig;
use hf_hub::{api::sync::Api, Repo};

use crate::embeddings::config::EmbeddingModelType;
use crate::error::{EmbeddingError, Result};

/// Get the cache directory for models.
///
/// Returns `~/.cache/ferris/models/` on Unix systems, or equivalent on other platforms.
/// Falls back to `.cache/` in the current directory if the system cache directory
/// cannot be determined.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::embeddings::loader::get_cache_dir;
///
/// let cache_dir = get_cache_dir();
/// println!("Models cached in: {:?}", cache_dir);
/// ```
///
/// # Note
///
/// This function is provided for informational purposes. The `HuggingFace` Hub API
/// manages caching automatically, but this can be useful for cleanup or debugging.
#[must_use]
#[allow(dead_code)] // Public utility function for users to query cache location
pub fn get_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("ferris")
        .join("models")
}

/// Download a model from `HuggingFace` Hub if not already cached.
///
/// This function downloads the model configuration, tokenizer, and weights
/// from `HuggingFace` Hub. Files are cached locally to avoid re-downloading
/// on subsequent calls.
///
/// # Arguments
///
/// * `model_type` - The embedding model to download
///
/// # Returns
///
/// Path to the directory containing the model files.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::embeddings::loader::download_model;
/// use metal_candle::embeddings::EmbeddingModelType;
///
/// let model_dir = download_model(EmbeddingModelType::E5SmallV2)?;
/// println!("Model downloaded to: {:?}", model_dir);
/// # Ok::<(), metal_candle::error::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`EmbeddingError::DownloadFailed`] if:
/// - Network connection fails
/// - Model files are not found on `HuggingFace` Hub
/// - Download is interrupted or corrupted
///
/// Returns [`EmbeddingError::ModelNotFound`] if the model does not exist on the Hub.
pub fn download_model(model_type: EmbeddingModelType) -> Result<PathBuf> {
    let api = Api::new().map_err(|e| EmbeddingError::DownloadFailed {
        reason: format!("Failed to initialize HuggingFace API: {e}"),
    })?;

    // Use Repo::model() for default branch (simpler and more reliable)
    let repo = Repo::model(model_type.model_id().to_string());

    let repo_api = api.repo(repo);

    // Download required files
    let config_path = repo_api
        .get("config.json")
        .map_err(|e| EmbeddingError::DownloadFailed {
            reason: format!("Failed to download config.json: {e}"),
        })?;

    let _tokenizer_path =
        repo_api
            .get("tokenizer.json")
            .map_err(|e| EmbeddingError::DownloadFailed {
                reason: format!("Failed to download tokenizer.json: {e}"),
            })?;

    // Try safetensors first, fall back to pytorch_model.bin
    let _weights_path = repo_api
        .get("model.safetensors")
        .or_else(|_| repo_api.get("pytorch_model.bin"))
        .map_err(|e| EmbeddingError::DownloadFailed {
            reason: format!("Failed to download model weights: {e}"),
        })?;

    // Return the directory containing all files
    Ok(config_path
        .parent()
        .ok_or_else(|| EmbeddingError::DownloadFailed {
            reason: "Invalid model path structure".to_string(),
        })?
        .to_path_buf())
}

/// Load BERT configuration from a model directory.
///
/// # Arguments
///
/// * `model_dir` - Path to the directory containing `config.json`
///
/// # Returns
///
/// The parsed BERT configuration.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::embeddings::loader::{download_model, load_config};
/// use metal_candle::embeddings::EmbeddingModelType;
///
/// let model_dir = download_model(EmbeddingModelType::E5SmallV2)?;
/// let config = load_config(&model_dir)?;
/// println!("Hidden size: {}", config.hidden_size);
/// # Ok::<(), metal_candle::error::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - `config.json` file does not exist
/// - File cannot be read
/// - JSON parsing fails
/// - Configuration is invalid
pub fn load_config(model_dir: &std::path::Path) -> Result<BertConfig> {
    let config_path = model_dir.join("config.json");

    if !config_path.exists() {
        return Err(EmbeddingError::DownloadFailed {
            reason: format!("config.json not found in {}", model_dir.display()),
        }
        .into());
    }

    let config_str = std::fs::read_to_string(&config_path)?;
    serde_json::from_str(&config_str).map_err(|e| {
        EmbeddingError::InvalidConfig {
            reason: format!("Failed to parse config.json: {e}"),
        }
        .into()
    })
}

/// Load model weights as a `VarBuilder`.
///
/// Loads weights from `model.safetensors` if available, otherwise falls back
/// to `pytorch_model.bin` (though `PyTorch` weight conversion is not yet implemented).
///
/// # Arguments
///
/// * `model_dir` - Path to the directory containing model weights
/// * `device` - Device to load weights onto (CPU or Metal)
///
/// # Returns
///
/// A `VarBuilder` for constructing the BERT model.
///
/// # Examples
///
/// ```no_run
/// use candle_core::Device;
/// use metal_candle::embeddings::loader::{download_model, load_weights};
/// use metal_candle::embeddings::EmbeddingModelType;
///
/// let model_dir = download_model(EmbeddingModelType::E5SmallV2)?;
/// let device = Device::Cpu;
/// let vb = load_weights(&model_dir, &device)?;
/// # Ok::<(), metal_candle::error::Error>(())
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Neither `model.safetensors` nor `pytorch_model.bin` exists
/// - Weight files are corrupted or invalid
/// - Loading to the specified device fails
///
/// # Panics
///
/// Panics if `PyTorch` weights are found but conversion is not yet implemented
/// (this will be fixed in a future version).
pub fn load_weights<'a>(model_dir: &std::path::Path, device: &'a Device) -> Result<VarBuilder<'a>> {
    use candle_core::safetensors::load as load_safetensors;
    use std::collections::HashMap;

    let safetensors_path = model_dir.join("model.safetensors");
    let pytorch_path = model_dir.join("pytorch_model.bin");

    if safetensors_path.exists() {
        // Load from safetensors
        let tensors: HashMap<String, candle_core::Tensor> =
            load_safetensors(&safetensors_path, device).map_err(|e| {
                EmbeddingError::DownloadFailed {
                    reason: format!("Failed to load safetensors weights: {e}"),
                }
            })?;

        Ok(VarBuilder::from_tensors(
            tensors,
            candle_core::DType::F32,
            device,
        ))
    } else if pytorch_path.exists() {
        // PyTorch weight conversion not yet implemented
        Err(EmbeddingError::InvalidConfig {
            reason: "PyTorch weight conversion not yet implemented. Please use a model with safetensors weights.".to_string(),
        }
        .into())
    } else {
        Err(EmbeddingError::DownloadFailed {
            reason: format!(
                "No model weights found in {}. Expected model.safetensors or pytorch_model.bin",
                model_dir.display()
            ),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.to_string_lossy().contains("ferris"));
        assert!(cache_dir.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_model_type_ids() {
        assert_eq!(
            EmbeddingModelType::E5SmallV2.model_id(),
            "intfloat/e5-small-v2"
        );
        assert_eq!(
            EmbeddingModelType::AllMiniLmL6V2.model_id(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
    }

    // Note: Integration tests that download models are in tests/embeddings/
    // to avoid network calls during unit testing
}
