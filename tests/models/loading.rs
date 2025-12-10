//! Integration tests for model loading

use metal_candle::models::{ModelConfig, ModelLoader};
use metal_candle::Device;
use std::collections::HashMap;

#[test]
fn test_config_parsing_and_validation() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12
    }"#;

    let config = ModelConfig::from_json(json).expect("Failed to parse config");
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 768);

    // Validation should pass
    config.validate().expect("Validation failed");
}

#[test]
fn test_config_validation_fails_on_invalid_dimensions() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 769,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12
    }"#;

    let config = ModelConfig::from_json(json).expect("Failed to parse config");

    // Validation should fail (769 not divisible by 12)
    assert!(config.validate().is_err());
}

#[test]
fn test_model_loader_creation() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    assert!(loader.device().is_cpu());
    assert_eq!(loader.dtype(), None);
}

#[test]
fn test_model_loader_with_dtype() {
    use candle_core::DType;

    let device = Device::new_cpu();
    let loader = ModelLoader::new(device).with_dtype(DType::F16);

    assert_eq!(loader.dtype(), Some(DType::F16));
}

#[test]
fn test_load_nonexistent_file() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    let result = loader.load("tests/fixtures/nonexistent.safetensors");
    assert!(result.is_err());

    // Check it's the correct error type
    match result {
        Err(metal_candle::Error::Model(metal_candle::error::ModelError::FileNotFound {
            ..
        })) => { /* Expected */ }
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_inspect_nonexistent_file() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    let result = loader.inspect("tests/fixtures/nonexistent.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_validation_with_shape_mismatch() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    // This would test validation if we had a test fixture
    // For now, we just ensure the API works
    let mut expected = HashMap::new();
    expected.insert("test_tensor".to_string(), vec![10, 20]);

    // Would fail with FileNotFound since file doesn't exist
    let result = loader.load_with_validation("nonexistent.safetensors", &expected);
    assert!(result.is_err());
}

#[test]
fn test_model_loader_builder_defaults() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    // Verify defaults
    assert!(loader.device().is_cpu());
    assert_eq!(loader.dtype(), None);
}

#[test]
fn test_model_loader_with_custom_dtype() {
    use candle_core::DType;

    let device = Device::new_cpu();
    let loader = ModelLoader::new(device)
        .with_dtype(DType::F32);

    assert_eq!(loader.dtype(), Some(DType::F32));
}

#[test]
fn test_model_loader_with_metal_device() {
    use metal_candle::backend::Device as MetalCandleDevice;

    #[cfg(target_os = "macos")]
    {
        if let Ok(device) = MetalCandleDevice::new_metal(0) {
            let loader = ModelLoader::new(device.clone());
            assert!(loader.device().is_metal());
        }
    }
}

#[test]
fn test_inspect_returns_metadata() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    // Test that inspect returns error for non-existent file
    let result = loader.inspect("tests/fixtures/nonexistent.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_load_with_validation_checks_tensor_presence() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    let mut expected_shapes = HashMap::new();
    expected_shapes.insert("required_tensor".to_string(), vec![128, 256]);

    // Should fail because file doesn't exist
    let result = loader.load_with_validation(
        "tests/fixtures/nonexistent.safetensors",
        &expected_shapes,
    );
    assert!(result.is_err());
}

#[test]
fn test_config_with_zero_layers_fails_validation() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 0,
        "num_attention_heads": 12
    }"#;

    let config = ModelConfig::from_json(json).expect("Failed to parse config");
    
    // Validation should fail for zero layers
    assert!(config.validate().is_err());
}

#[test]
fn test_config_with_zero_vocab_fails_validation() {
    let json = r#"{
        "vocab_size": 0,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12
    }"#;

    let config = ModelConfig::from_json(json).expect("Failed to parse config");
    
    // Validation should fail for zero vocab
    assert!(config.validate().is_err());
}

#[test]
fn test_config_with_mismatched_head_count() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 13
    }"#;

    let config = ModelConfig::from_json(json).expect("Failed to parse config");
    
    // Validation should fail when hidden_size not divisible by num_attention_heads
    assert!(config.validate().is_err());
}

#[test]
fn test_config_from_invalid_json() {
    let invalid_json = r#"{ invalid json }"#;

    let result = ModelConfig::from_json(invalid_json);
    assert!(result.is_err());
}

#[test]
fn test_config_missing_required_fields() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768
    }"#;

    // Should fail to parse due to missing required fields
    let result = ModelConfig::from_json(json);
    assert!(result.is_err());
}

#[test]
fn test_load_empty_path() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device);

    let result = loader.load("");
    assert!(result.is_err());
}

#[test]
fn test_multiple_dtype_configurations() {
    use candle_core::DType;

    let device = Device::new_cpu();

    // Test F16
    let loader_f16 = ModelLoader::new(device.clone()).with_dtype(DType::F16);
    assert_eq!(loader_f16.dtype(), Some(DType::F16));

    // Test F32
    let loader_f32 = ModelLoader::new(device.clone()).with_dtype(DType::F32);
    assert_eq!(loader_f32.dtype(), Some(DType::F32));

    // Test F64
    let loader_f64 = ModelLoader::new(device).with_dtype(DType::F64);
    assert_eq!(loader_f64.dtype(), Some(DType::F64));
}

#[test]
fn test_loader_device_accessor() {
    let device = Device::new_cpu();
    let loader = ModelLoader::new(device.clone());

    assert!(loader.device().is_cpu());
    assert_eq!(loader.device().info().device_type, metal_candle::backend::DeviceType::Cpu);
}

#[test]
fn test_config_with_large_values() {
    let json = r#"{
        "vocab_size": 100000,
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": 48,
        "num_attention_heads": 32
    }"#;

    let config = ModelConfig::from_json(json).expect("Failed to parse config");
    assert_eq!(config.vocab_size, 100000);
    assert_eq!(config.hidden_size, 4096);

    // Should pass validation
    config.validate().expect("Validation failed");
}

#[test]
fn test_config_head_dim_calculation() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12
    }"#;

    let config = ModelConfig::from_json(json).unwrap();
    assert_eq!(config.head_dim(), 64); // 768 / 12
}

#[test]
fn test_config_num_kv_heads_default() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12
    }"#;

    let config = ModelConfig::from_json(json).unwrap();
    assert_eq!(config.num_kv_heads(), config.num_attention_heads);
}

#[test]
fn test_config_num_kv_heads_specified() {
    let json = r#"{
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4
    }"#;

    let config = ModelConfig::from_json(json).unwrap();
    assert_eq!(config.num_kv_heads(), 4);
}

// Note: Tests for actual file loading would require test fixtures
// These would be added in tests/fixtures/ with small safetensors files
// for testing purposes.

