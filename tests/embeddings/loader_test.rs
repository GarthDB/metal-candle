//! Tests for embeddings loader module.

#![cfg(feature = "embeddings")]

use metal_candle::embeddings::loader::{get_cache_dir, load_config, load_weights};
use candle_core::Device;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_get_cache_dir() {
    let cache_dir = get_cache_dir();
    assert!(cache_dir.to_string_lossy().contains("ferris"));
    assert!(cache_dir.to_string_lossy().contains("models"));
}

#[test]
fn test_load_config_success() -> Result<(), Box<dyn std::error::Error>> {
    // Use the test fixture
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embeddings");
    
    let config = load_config(&fixture_path)?;
    
    // Verify config was loaded correctly
    assert_eq!(config.hidden_size, 384);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_hidden_layers, 12);
    assert_eq!(config.vocab_size, 30522);
    
    Ok(())
}

#[test]
fn test_load_config_file_not_found() {
    let temp_dir = TempDir::new().unwrap();
    let non_existent_path = temp_dir.path().join("nonexistent");
    
    let result = load_config(&non_existent_path);
    assert!(result.is_err());
    
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("config.json not found") || err_msg.contains("No such file"));
}

#[test]
fn test_load_config_invalid_json() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embeddings");
    
    // Create temp dir and copy invalid config
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("config.json");
    
    fs::copy(
        fixture_path.join("config_invalid.json"),
        &config_path,
    ).unwrap();
    
    let result = load_config(temp_dir.path());
    assert!(result.is_err());
    
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("parse") || err_msg.contains("JSON") || err_msg.contains("Invalid"));
}

#[test]
fn test_load_weights_safetensors() -> Result<(), Box<dyn std::error::Error>> {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embeddings");
    
    let device = Device::Cpu;
    
    // This should load without error (even though it's a minimal file)
    let result = load_weights(&fixture_path, &device);
    
    // The minimal safetensors file we created should load successfully
    // (it has valid header but no tensors)
    assert!(result.is_ok());
    
    Ok(())
}

#[test]
fn test_load_weights_no_files() {
    let temp_dir = TempDir::new().unwrap();
    let device = Device::Cpu;
    
    let result = load_weights(temp_dir.path(), &device);
    assert!(result.is_err());
    
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("No model weights found") || err_msg.contains("model"));
}

#[test]
fn test_load_weights_pytorch_not_implemented() {
    let temp_dir = TempDir::new().unwrap();
    let device = Device::Cpu;
    
    // Copy only the pytorch_model.bin file
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embeddings");
    
    fs::copy(
        fixture_path.join("pytorch_model.bin"),
        temp_dir.path().join("pytorch_model.bin"),
    ).unwrap();
    
    let result = load_weights(temp_dir.path(), &device);
    assert!(result.is_err());
    
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("PyTorch") || err_msg.contains("not yet implemented"));
}

// Note: download_model tests are marked #[ignore] because they require network access
// Run them manually with: cargo test --features embeddings -- --ignored

#[test]
#[ignore]
fn test_download_model_e5_small() -> Result<(), Box<dyn std::error::Error>> {
    use metal_candle::embeddings::config::EmbeddingModelType;
    use metal_candle::embeddings::loader::download_model;
    
    // This will actually download the model (only run manually)
    let model_dir = download_model(EmbeddingModelType::E5SmallV2)?;
    
    assert!(model_dir.exists());
    assert!(model_dir.join("config.json").exists());
    assert!(model_dir.join("tokenizer.json").exists());
    
    Ok(())
}



